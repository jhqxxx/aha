use std::io::{Read, Seek};

use ahash::AHashMap;
use anyhow::{Result, anyhow};
use candle_core::{
    DType, Device, Tensor,
    quantized::{
        QMatMul, QTensor,
        gguf_file::{self, Value},
    },
};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, LayerNorm, Linear, Module, RmsNorm, VarBuilder, linear_b,
};
use tokenizers::{self, AddedToken, Tokenizer, models::bpe::BPE};

use crate::tokenizer::TokenizerModel;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn get_matedata(&self, name: &str) -> Result<Value> {
        match self.ct.metadata.get(name) {
            None => Err(anyhow!("cannot find {name} in metadata")),
            Some(v) => Ok(v.clone()),
        }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        Ok(QMatMul::from_qtensor(ws)?)
    }

    pub fn quantize_linear(&mut self, prefix: &str, bias: bool) -> Result<QuantizedLinear> {
        let weight = self.qmatmul(&format!("{prefix}.weight"))?;
        let bias = if bias {
            self.get_dequantized(&format!("{prefix}.bias")).ok()
        } else {
            None
        };
        Ok(QuantizedLinear::new(weight, bias))
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?;
        Ok(RmsNorm::new(weight, eps))
    }

    pub fn layer_norm(&mut self, prefix: &str, eps: f64) -> Result<LayerNorm> {
        let weight = self
            .ct
            .tensor(&mut self.reader, &format!("{prefix}.weight"), &self.device)?;
        let weight = weight.dequantize(&self.device)?;
        let bias = self
            .ct
            .tensor(&mut self.reader, &format!("{prefix}.bias"), &self.device);
        let bias = match bias {
            Ok(bias) => bias.dequantize(&self.device).ok(),
            Err(_) => None,
        };
        match bias {
            Some(bias) => Ok(LayerNorm::new(weight, bias, eps)),
            None => Ok(LayerNorm::new_no_bias(weight, eps)),
        }
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        Ok(self.ct.tensor(&mut self.reader, name, &self.device)?)
    }

    pub fn get_dequantized(&mut self, name: &str) -> Result<Tensor> {
        Ok(self.tensor(name)?.dequantize(&self.device)?)
    }

    pub fn get_dequantized_f16(&mut self, name: &str) -> Result<Tensor> {
        Ok(self.tensor(name)?.dequantize_f16(&self.device)?)
    }

    pub fn conv1d(
        &mut self,
        prefix: &str,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Conv1d> {
        let weight = self.get_dequantized(&format!("{prefix}.weight"))?;
        let bias = if bias {
            self.get_dequantized(&format!("{prefix}.bias")).ok()
        } else {
            None
        };
        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        Ok(Conv1d::new(weight, bias, cfg))
    }

    pub fn build_tokenizer(
        &self,
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
    ) -> Result<TokenizerModel> {
        let model_type = self
            .get_matedata("tokenizer.ggml.model")?
            .to_string()?
            .clone();
        match model_type.as_str() {
            "gpt2" | "llama" => {
                let vocab = self
                    .get_matedata("tokenizer.ggml.tokens")?
                    .to_vec()?
                    .clone();
                let vocab: Vec<String> = vocab
                    .into_iter()
                    .map(|tokens| tokens.to_string().cloned())
                    .collect::<Result<Vec<String>, candle_core::Error>>()?;
                let mut vocab_map = AHashMap::new();
                for (id, token) in vocab.iter().enumerate() {
                    vocab_map.insert(token.clone(), id as u32);
                }

                let merges = self
                    .get_matedata("tokenizer.ggml.merges")?
                    .to_vec()?
                    .clone();
                let merges: Vec<String> = merges
                    .into_iter()
                    .map(|tokens| tokens.to_string().cloned())
                    .collect::<Result<Vec<String>, candle_core::Error>>()?;
                let merges: Vec<(String, String)> = merges
                    .into_iter()
                    .map(|token_merge| {
                        let merge: Vec<&str> = token_merge.split(" ").collect();
                        if merge.len() != 2 {
                            // 处理格式不正确的merge规则
                            return ("".to_string(), "".to_string());
                        }
                        (merge[0].to_string(), merge[1].to_string())
                    })
                    .filter(|(a, b)| !a.is_empty() && !b.is_empty())
                    .collect();
                let bpe_model = BPE::new(vocab_map, merges);
                let mut tokenizer = Tokenizer::new(bpe_model);
                let add_prefix_space = add_prefix_space.unwrap_or(false);
                let trim_offsets = trim_offsets.unwrap_or(false);
                let use_regex = use_regex.unwrap_or(false);
                let pre_byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::default()
                    .add_prefix_space(add_prefix_space) // 是否在文本开头添加空格，gpt-2默认是true
                    .trim_offsets(trim_offsets) // 是否删除首尾空白字符
                    .use_regex(use_regex); // 是否使用正则表达式来分割特殊字符
                tokenizer.with_pre_tokenizer(Some(pre_byte_level));
                let dec_byte_level = tokenizers::decoders::byte_level::ByteLevel::default();
                tokenizer.with_decoder(Some(dec_byte_level));
                let token_types = self
                    .get_matedata("tokenizer.ggml.token_type")?
                    .to_vec()?
                    .clone();
                let token_types = token_types
                    .into_iter()
                    .map(|types| types.to_i32())
                    .collect::<Result<Vec<i32>, candle_core::Error>>()?;

                let mut add_tokens = vec![];
                for (id, type_) in token_types.into_iter().enumerate() {
                    if type_ == 3
                        && let Some(token_str) = vocab.get(id)
                    {
                        let add_token = AddedToken::from(token_str.clone(), true);
                        add_tokens.push(add_token);
                    } else if type_ == 4
                        && let Some(token_str) = vocab.get(id)
                    {
                        let add_token = AddedToken::from(token_str.clone(), false);
                        add_tokens.push(add_token);
                    }
                }
                let _ = tokenizer.add_special_tokens(&add_tokens);
                let tokenizer_model = TokenizerModel::new(tokenizer);
                Ok(tokenizer_model)
            }
            _ => Err(anyhow!("Unsupported tokenizer model type: {model_type}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    pub fn new(inner: QMatMul, bias: Option<Tensor>) -> Self {
        Self { inner, bias }
    }

    pub fn inner_dequantize(&self) -> Result<Tensor> {
        Ok(self.inner.dequantize_f16()?)
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = if xs.dtype() == DType::F16 {
            self.inner.forward_via_f16(xs)?
        } else {
            self.inner.forward(xs)?
        };
        if let Some(bias) = &self.bias {
            xs.broadcast_add(&bias.to_dtype(xs.dtype())?)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProjKind {
    QuantizedProj(QuantizedLinear),
    LinearProj(Linear),
}

impl Module for ProjKind {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            ProjKind::QuantizedProj(q) => q.forward(xs),
            ProjKind::LinearProj(l) => l.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateUpDownMLPGguf {
    gate_proj: ProjKind,
    up_proj: ProjKind,
    down_proj: ProjKind,
    act: Activation,
}

impl GateUpDownMLPGguf {
    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        bias: bool,
        gate_name: Option<&str>,
        up_name: Option<&str>,
        down_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let gate_name = gate_name.unwrap_or("ffn_gate");
        let up_name = up_name.unwrap_or("ffn_up");
        let down_name = down_name.unwrap_or("ffn_down");
        let gate_proj = gguf.qmatmul(&format!("{prefix}.{gate_name}.weight"))?;
        let gate_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{gate_name}.bias"))
                .ok()
        } else {
            None
        };
        let up_proj = gguf.qmatmul(&format!("{prefix}.{up_name}.weight"))?;
        let up_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{up_name}.bias"))
                .ok()
        } else {
            None
        };
        let down_proj = gguf.qmatmul(&format!("{prefix}.{down_name}.weight"))?;
        let down_bias = if bias {
            gguf.get_dequantized(&format!("{prefix}.{down_name}.bias"))
                .ok()
        } else {
            None
        };
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            gate_proj: ProjKind::QuantizedProj(QuantizedLinear::new(gate_proj, gate_bias)),
            up_proj: ProjKind::QuantizedProj(QuantizedLinear::new(up_proj, up_bias)),
            down_proj: ProjKind::QuantizedProj(QuantizedLinear::new(down_proj, down_bias)),
            act,
        })
    }
    pub fn new_from_vb(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        gate_pp_name: Option<&str>,
        up_pp_name: Option<&str>,
        down_pp_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let gate_pp_name = gate_pp_name.unwrap_or("gate_proj");
        let up_pp_name = up_pp_name.unwrap_or("up_proj");
        let down_pp_name = down_pp_name.unwrap_or("down_proj");
        let gate_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(gate_pp_name))?;
        let up_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(up_pp_name))?;
        let down_proj = linear_b(intermediate_size, hidden_size, bias, vb.pp(down_pp_name))?;
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            gate_proj: ProjKind::LinearProj(gate_proj),
            up_proj: ProjKind::LinearProj(up_proj),
            down_proj: ProjKind::LinearProj(down_proj),
            act,
        })
    }
}

impl Module for GateUpDownMLPGguf {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let w1 = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let w3 = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(w1 * w3)?)
    }
}

pub struct TwoLinearMLPGguf {
    linear1: ProjKind,
    linear2: ProjKind,
    act: Activation,
}

impl TwoLinearMLPGguf {
    pub fn new(
        vb: VarBuilder,
        // embedding_dim: usize,
        // mlp_dim: usize,
        in_dim: usize,
        middle_dim: usize,
        out_dim: usize,
        act: Activation,
        bias: bool,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
    ) -> Result<Self> {
        let linear1 = linear_b(in_dim, middle_dim, bias, vb.pp(linear1_pp_name))?;
        let linear2 = linear_b(middle_dim, out_dim, bias, vb.pp(linear2_pp_name))?;

        Ok(Self {
            linear1: ProjKind::LinearProj(linear1),
            linear2: ProjKind::LinearProj(linear2),
            act,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        bias: bool,
        linear1_name: Option<&str>,
        linear2_name: Option<&str>,
        act: Option<Activation>,
    ) -> Result<Self> {
        let linear1_name = linear1_name.unwrap_or("ffn_up");
        let linear2_name = linear2_name.unwrap_or("ffn_down");
        let linear1 = gguf.quantize_linear(&format!("{prefix}.{linear1_name}"), bias)?;
        let linear2 = gguf.quantize_linear(&format!("{prefix}.{linear2_name}"), bias)?;
        let act = act.unwrap_or(Activation::Silu);
        Ok(Self {
            linear1: ProjKind::QuantizedProj(linear1),
            linear2: ProjKind::QuantizedProj(linear2),
            act,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}
