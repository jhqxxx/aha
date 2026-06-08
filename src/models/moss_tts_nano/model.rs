use crate::{
    models::{
        common::sample::simple_sample, gpt2::GPT2Model,
        moss_audio_tokenizer_nano::MossAudioTokenizer, moss_tts_nano::config::MossTTSConfig,
    },
    utils::audio_utils::save_wav,
};
use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear_no_bias};
// use candle_transformers::generation::LogitsProcessor;

#[derive(PartialEq, Clone, Debug)]
pub enum MossTTSMode {
    Continuation,
    VoiceClone,
}

pub struct MossTTSModel {
    transformer: GPT2Model,
    audio_embeddings: Vec<Embedding>,
    text_lm_head: Linear,
    audio_lm_heads: Vec<Linear>,
    local_transformer: GPT2Model,
    audio_assistant_slot_token_id: usize,
    audio_end_token_id: usize,
    n_vq: usize,
    audio_pad_token_id_tensor: Tensor,
    audio_codebook_sizes: Vec<usize>,
    audio_temperature: f64,
    audio_top_k: usize,
    audio_top_p: f32,
    audio_repetition_penalty: f32,
    // audio_processor: LogitsProcessor,
}

impl MossTTSModel {
    pub fn new(vb: VarBuilder, cfg: &MossTTSConfig) -> Result<Self> {
        let transformer = GPT2Model::new(
            vb.pp("transformer"),
            cfg.gpt2_config.n_embd,
            cfg.gpt2_config.n_head,
            cfg.gpt2_config.n_layer,
            cfg.gpt2_config.vocab_size,
            // cfg.gpt2_config.n_positions,
        )?;
        let mut audio_embeddings = vec![];
        let audio_embed_vb = vb.pp("audio_embeddings");
        for i in 0..cfg.n_vq {
            let embed = embedding(
                cfg.audio_codebook_sizes[i],
                cfg.gpt2_config.n_embd,
                audio_embed_vb.pp(i),
            )?;
            audio_embeddings.push(embed);
        }
        let text_lm_head = linear_no_bias(
            cfg.gpt2_config.n_embd,
            cfg.gpt2_config.vocab_size,
            vb.pp("text_lm_head"),
        )?;

        let mut audio_lm_heads = vec![];
        let audio_lm_vb = vb.pp("audio_lm_heads");
        for i in 0..cfg.n_vq {
            let layer = linear_no_bias(
                cfg.gpt2_config.n_embd,
                cfg.audio_codebook_sizes[i],
                audio_lm_vb.pp(i),
            )?;
            audio_lm_heads.push(layer);
        }

        let mut local_gpt2_cfg = cfg.gpt2_config.clone();
        local_gpt2_cfg.n_layer = cfg.local_transformer_layers;
        local_gpt2_cfg.n_positions = cfg.n_vq + 1;
        local_gpt2_cfg.n_ctx = cfg.n_vq + 1;
        let local_transformer = GPT2Model::new_without_wte(
            vb.pp("local_transformer"),
            local_gpt2_cfg.n_embd,
            local_gpt2_cfg.n_head,
            local_gpt2_cfg.n_layer,
            local_gpt2_cfg.vocab_size,
            // local_gpt2_cfg.n_positions,
        )?;
        let audio_pad_token_id_tensor = Tensor::new(cfg.audio_pad_token_id, vb.device())?;
        // let audio_processor = get_logit_processor(Some(0.8), Some(0.95), Some(25), 34562);
        Ok(Self {
            transformer,
            audio_embeddings,
            text_lm_head,
            audio_lm_heads,
            local_transformer,
            audio_assistant_slot_token_id: cfg.audio_assistant_slot_token_id as usize,
            audio_end_token_id: cfg.audio_end_token_id as usize,
            n_vq: cfg.n_vq,
            audio_pad_token_id_tensor,
            audio_codebook_sizes: cfg.audio_codebook_sizes.clone(),
            audio_temperature: 0.8,
            audio_top_k: 25,
            audio_top_p: 0.95,
            audio_repetition_penalty: 1.2,
            // audio_processor,
        })
    }

    fn build_inputs_embeds(&self, input_ids: &Tensor) -> Result<Tensor> {
        let text_ids = input_ids.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let mut inputs_embeds = if let Some(wte) = &self.transformer.wte {
            wte.forward(&text_ids)?
        } else {
            return Err(anyhow!("MossTTS transformer wte can not be none"));
        };
        for (channel_index, embedding) in self.audio_embeddings.iter().enumerate() {
            let channel_ids = input_ids
                .narrow(D::Minus1, channel_index + 1, 1)?
                .squeeze(D::Minus1)?;
            let valid_mask = channel_ids.ne(&self
                .audio_pad_token_id_tensor
                .broadcast_as(channel_ids.shape())?)?;
            let invalid_mask = channel_ids.lt(&channel_ids.zeros_like()?)?;
            let embedding_nums = Tensor::new(
                self.audio_codebook_sizes[channel_index] as u32,
                input_ids.device(),
            )?;
            let invalid_mask1 =
                channel_ids.ge(&embedding_nums.broadcast_as(channel_ids.shape())?)?;
            let invalid_mask = valid_mask
                .minimum(&invalid_mask.maximum(&invalid_mask1)?)?
                .to_dtype(candle_core::DType::U32)?;
            if invalid_mask.sum_all()?.to_scalar::<u32>()? > 0 {
                return Err(anyhow!("Found out-of-range audio token ids for channel"));
            }
            let safe_ids = valid_mask.where_cond(&channel_ids, &channel_ids.zeros_like()?)?;
            let audio_embeds = embedding.forward(&safe_ids)?;
            let audio_embeds = audio_embeds.broadcast_mul(
                &valid_mask
                    .unsqueeze(D::Minus1)?
                    .to_dtype(audio_embeds.dtype())?,
            )?;
            inputs_embeds = inputs_embeds.add(&audio_embeds)?;
        }
        Ok(inputs_embeds)
    }

    fn sample_next_assistant_text_token(&self, logits: &Tensor) -> Result<usize> {
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let slot_token_id_logit = logits
            .i(self.audio_assistant_slot_token_id)?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        let end_token_id_logit = logits
            .i(self.audio_end_token_id)?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        let logits = Tensor::new(&[slot_token_id_logit, end_token_id_logit], logits.device())?;
        let token = simple_sample(&logits, true, None, None, None, None, 1.0)?;
        if token == 0 {
            Ok(self.audio_assistant_slot_token_id)
        } else {
            Ok(self.audio_end_token_id)
        }
    }

    fn build_generation_row(&self, audio_token_ids: &Tensor) -> Result<Tensor> {
        let slot = Tensor::from_slice(
            &[self.audio_assistant_slot_token_id as u32],
            (1, 1, 1),
            audio_token_ids.device(),
        )?;
        let audio_token_ids = audio_token_ids.unsqueeze(0)?.unsqueeze(0)?;
        Ok(Tensor::cat(&[&slot, &audio_token_ids], D::Minus1)?)
    }

    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        audio_tokenizer: &MossAudioTokenizer,
    ) -> Result<()> {
        let sample_len = 100;
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let mut generated_frames = vec![];
        let mut current_model_input_ids = input_ids.clone();
        for _ in 0..sample_len {
            let inputs_embeds = self.build_inputs_embeds(&current_model_input_ids)?;
            let outputs = self.transformer.forward(&inputs_embeds, seqlen_offset)?;
            let outputs_len = outputs.dim(1)?;
            let global_hidden_state = outputs.narrow(1, outputs_len - 1, 1)?;
            let mut local_positions = 0usize;
            let local_outputs = self
                .local_transformer
                .forward(&global_hidden_state, local_positions)?;
            let local_len = local_outputs.dim(1)?;
            let local_hidden_states = local_outputs.narrow(1, local_len - 1, 1)?;
            let text_logits = self.text_lm_head.forward(&local_hidden_states)?;
            let next_text_token = self.sample_next_assistant_text_token(&text_logits)?;
            if next_text_token == self.audio_end_token_id {
                self.local_transformer.clear_kv_cache();
                break;
            }
            let mut next_frame_tokens = vec![];
            let mut current_local_input = if let Some(wte) = &self.transformer.wte {
                wte.forward(&Tensor::from_slice(
                    &[next_text_token as u32],
                    (1, 1),
                    input_ids.device(),
                )?)?
            } else {
                return Err(anyhow!("MossTTS GPT2 wte can not be none"));
            };
            for channel_index in 0..self.n_vq {
                local_positions += 1;
                let local_outputs = self
                    .local_transformer
                    .forward(&current_local_input, local_positions)?;
                let local_len = local_outputs.dim(1)?;
                let local_hidden_states = local_outputs.narrow(1, local_len - 1, 1)?;
                let channel_logits = self.audio_lm_heads[channel_index]
                    .forward(&local_hidden_states)?
                    .squeeze(0)?
                    .squeeze(0)?;
                // let channel_token = self.audio_processor.sample(&channel_logits)?;
                let channel_token = simple_sample(
                    &channel_logits,
                    true,
                    Some(self.audio_temperature),
                    Some(self.audio_top_k),
                    Some(self.audio_top_p),
                    Some(&next_frame_tokens),
                    self.audio_repetition_penalty,
                )?;
                next_frame_tokens.push(channel_token);
                current_local_input = self.audio_embeddings[channel_index].forward(
                    &Tensor::from_slice(&[channel_token], (1, 1), input_ids.device())?,
                )?;
            }
            self.local_transformer.clear_kv_cache();
            let next_frame = Tensor::new(next_frame_tokens, input_ids.device())?;
            current_model_input_ids = self.build_generation_row(&next_frame)?;
            seqlen_offset += seq_len;
            seq_len = 1;
            generated_frames.push(next_frame);
        }
        let audio_token_ids = Tensor::stack(&generated_frames, 0)?;
        let waveform = audio_tokenizer
            .decode_audio_token_ids_to_waveform(&audio_token_ids)?
            .squeeze(0)?;
        save_wav(
            &waveform,
            "./demo.wav",
            2,
            audio_tokenizer.sampling_rate as u32,
        )?;
        Ok(())
    }

    pub fn decode(
        &self,
        prompt_audio_code: Option<&Tensor>,
        audio_tokenizer: &MossAudioTokenizer,
    ) -> Result<()> {
        if let Some(audio) = prompt_audio_code {
            let waveform = audio_tokenizer
                .decode_audio_token_ids_to_waveform(audio)?
                .squeeze(0)?;
            save_wav(
                &waveform,
                "./demo.wav",
                2,
                audio_tokenizer.sampling_rate as u32,
            )?;
        }
        Ok(())
    }
}
