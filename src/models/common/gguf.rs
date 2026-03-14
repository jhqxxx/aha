use std::io::{Read, Seek};

use ahash::AHashMap;
use anyhow::{Result, anyhow};
use candle_core::{
    Device, Tensor,
    quantized::{
        QMatMul, QTensor,
        gguf_file::{self, Value},
    },
};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, RmsNorm, VarBuilder, linear_b};
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

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?;
        Ok(RmsNorm::new(weight, eps))
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
pub enum ProjKind {
    QuantizedProj(QMatMul),
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
    gate_proj: ProjKind, // ffn_gate.weight
    up_proj: ProjKind,   // ffn_up.weight
    down_proj: ProjKind, // ffn_down.weight
}

impl GateUpDownMLPGguf {
    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gguf.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gguf.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gguf.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_proj: ProjKind::QuantizedProj(gate_proj),
            up_proj: ProjKind::QuantizedProj(up_proj),
            down_proj: ProjKind::QuantizedProj(down_proj),
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
    ) -> Result<Self> {
        let gate_pp_name = gate_pp_name.unwrap_or("gate_proj");
        let up_pp_name = up_pp_name.unwrap_or("up_proj");
        let down_pp_name = down_pp_name.unwrap_or("down_proj");
        let gate_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(gate_pp_name))?;
        let up_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(up_pp_name))?;
        let down_proj = linear_b(intermediate_size, hidden_size, bias, vb.pp(down_pp_name))?;
        Ok(Self {
            gate_proj: ProjKind::LinearProj(gate_proj),
            up_proj: ProjKind::LinearProj(up_proj),
            down_proj: ProjKind::LinearProj(down_proj),
        })
    }
}

impl Module for GateUpDownMLPGguf {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let w1 = self.gate_proj.forward(xs)?;
        let w3 = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}
