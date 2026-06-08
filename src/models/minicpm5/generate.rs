use crate::models::common::generate::{GenerationDataProvider, PrepareData};
use crate::models::llama::LlamaForCausalLM;
use crate::models::minicpm5::config::MiniCPM5Config;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

pub struct MiniCPM5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    model: LlamaForCausalLM,
    device: Device,
    model_name: String,
}

impl<'a> MiniCPM5GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: MiniCPM5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let model = LlamaForCausalLM::new(
            vb,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            Some(cfg.num_key_value_heads),
            Some(cfg.head_dim),
            false,
            "self_attn",
            Some("o_proj"),
            cfg.intermediate_size,
            cfg.hidden_act,
            false,
            "mlp",
            cfg.rms_norm_eps,
            "input_layernorm",
            "post_attention_layernorm",
            cfg.rope_theta,
            cfg.eos_token_id.clone(),
        )?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("minicpm5")
            .to_string();
        Ok(MiniCPM5GenerateModel {
            chat_template,
            tokenizer,
            model,
            device: device.clone(),
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for MiniCPM5GenerateModel<'a> {
    fn get_data(&self, mes: &crate::params::chat::ChatCompletionParameters) -> Result<PrepareData> {
        let mes_render = self.chat_template.apply_chat_template(mes)?;
        let in_reasoning = self.is_in_reasoning(&mes_render);
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let multi_model_data = self.get_multi_model_data();
        Ok(PrepareData {
            in_reasoning,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(MiniCPM5GenerateModel<'a>);
