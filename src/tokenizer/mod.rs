use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use sentencepiece::SentencePieceProcessor;
use serde_json::Value;
use tokenizers::{
    AddedToken, Tokenizer, decoders::byte_level::ByteLevel as ByteLevelDecoder, models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel,
};

pub struct TokenizerModel {
    pub tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn init(path: &str) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let tokenizer_file = path.clone() + "/tokenizer.json";
        let tokenizer = if std::path::Path::new(&tokenizer_file).exists() {
            Tokenizer::from_file(tokenizer_file)
                .map_err(|e| anyhow!(format!("tokenizer from file error{}", e)))?
        } else {
            // 如果不存在 tokenizer.json，尝试使用 vocab.json 和 merges.txt
            let vocab_file = path.clone() + "/vocab.json";
            let merges_file = path.clone() + "/merges.txt";
            let config_file = path.clone() + "/tokenizer_config.json";

            if !std::path::Path::new(&vocab_file).exists() {
                return Err(anyhow!(
                    "Neither tokenizer.json nor vocab.json found in model path"
                ));
            }

            if !std::path::Path::new(&merges_file).exists() {
                return Err(anyhow!(
                    "Neither tokenizer.json nor merges.txt found in model path"
                ));
            }
            // 创建 BPE 模型
            let bpe = BPE::from_file(&vocab_file, &merges_file)
                .build()
                .map_err(|e| anyhow!(format!("failed to build BPE tokenizer: {}", e)))?;

            // 创建分词器
            let mut tokenizer = Tokenizer::new(bpe);
            // 添加字节级预分词器，这会处理换行符等特殊字符
            let byte_level_pre_tokenizer = ByteLevel::new(false, true, false);
            tokenizer.with_pre_tokenizer(Some(byte_level_pre_tokenizer));
            tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
            if std::path::Path::new(&config_file).exists() {
                let config_content = std::fs::read_to_string(&config_file)?;
                let config: Value = serde_json::from_str(&config_content)?;
                if let Some(added_tokens_decoder) = config.get("added_tokens_decoder") {
                    let mut special_tokens = Vec::new();

                    if let Value::Object(tokens_map) = added_tokens_decoder {
                        for (_, token_info) in tokens_map {
                            if let Value::Object(token_obj) = token_info
                                && let Some(content_val) = token_obj.get("content")
                                && let Some(content) = content_val.as_str()
                            {
                                let special = token_obj
                                    .get("special")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);

                                let added_token = AddedToken::from(content.to_string(), special);
                                special_tokens.push(added_token);
                            }
                        }
                    }

                    // 添加所有特殊标记
                    if !special_tokens.is_empty() {
                        tokenizer.add_special_tokens(&special_tokens);
                    }
                }
            }
            tokenizer
        };
        Ok(Self { tokenizer })
    }

    pub fn text_encode_vec(&self, text: String, add_special_token: bool) -> Result<Vec<u32>> {
        let token_id = self
            .tokenizer
            .encode(text, add_special_token)
            .map_err(|e| anyhow!(format!("tokenizer encode error: {}", e)))?
            .get_ids()
            .to_vec();
        Ok(token_id)
    }
    pub fn text_encode(&self, text: String, device: &Device) -> Result<Tensor> {
        // let token_id = self
        //     .tokenizer
        //     .encode(text, true)
        //     .map_err(|e| anyhow!(format!("tokenizer encode error: {}", e)))?
        //     .get_ids()
        //     .to_vec();
        let token_id = self.text_encode_vec(text, true)?;
        let token_tensor = Tensor::from_slice(&token_id, (1, token_id.len()), device)?;
        Ok(token_tensor)
    }

    pub fn token_decode(&self, tokens: Vec<u32>) -> Result<String> {
        let decode = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!(format!("tokenizer encode error{}", e)))?;
        Ok(decode)
    }
}

pub fn sentencepiece_encode(
    text: &str,
    tokenizer: &SentencePieceProcessor,
    device: &Device,
) -> Result<Tensor> {
    let tokens = tokenizer
        .encode(text)
        .map_err(|e| anyhow!(format!("tokenizer encode error:{}", e)))?;
    let token_ids = tokens.iter().map(|p| p.id).collect::<Vec<u32>>();
    let tokens_t = Tensor::new(token_ids, device)?.unsqueeze(0)?;
    Ok(tokens_t)
}
