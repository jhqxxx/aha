use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device};

use crate::{
    models::index_tts2::{config::IndexTTS2Config, processor::IndexTTS2Processor},
    utils::{get_default_save_dir, get_device, get_dtype},
};
pub struct IndexTTS2Generate {
    processor: IndexTTS2Processor,
    config: IndexTTS2Config,
}

impl IndexTTS2Generate {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let config_path = path.to_string() + "/config.yaml";
        let save_dir = get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
        let config: IndexTTS2Config = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, "bf16");
        let processor = IndexTTS2Processor::new(path, &save_dir, &config, &device, dtype)?;

        Ok(Self { config, processor })
    }
    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<()> {
        let _ = self.processor.process_info(&mes)?;
        Ok(())
    }
}
