use crate::{
    models::{
        moss_audio_tokenizer_nano::MossAudioTokenizer,
        moss_tts_nano::{config::MossTTSConfig, model::MossTTSMode},
    },
    tokenizer::sentencepiece_encode_vec,
    utils::{audio_utils::load_audio_with_resample, prepare_tts_text},
};
use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use sentencepiece::SentencePieceProcessor;

pub struct MossTTSProcessor {
    target_sample_rate: usize,
    target_channels: usize,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_user_slot_token_id: u32,
    audio_assistant_slot_token_id: u32,
    audio_pad_token_id: u32,
    n_vq: usize,
    prompt_token_ids: Vec<u32>,
    user_after_ids: Vec<u32>,
    assistant_ids: Vec<u32>,
    none_ids: Vec<u32>,
}

impl MossTTSProcessor {
    pub fn new(
        tts_cfg: &MossTTSConfig,
        target_sample_rate: usize,
        target_channels: usize,
        text_tokenizer: &SentencePieceProcessor,
    ) -> Result<Self> {
        let mut prompt_token_ids = vec![tts_cfg.im_start_token_id];
        let user_role_ids = sentencepiece_encode_vec("user\n", text_tokenizer)?;
        prompt_token_ids.extend_from_slice(&user_role_ids);
        let user_template_pre_ids =
            sentencepiece_encode_vec("<user_inst>\n- Reference(s):\n", text_tokenizer)?;
        prompt_token_ids.extend_from_slice(&user_template_pre_ids);

        let user_after_ids = sentencepiece_encode_vec(
            "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n",
            text_tokenizer,
        )?;
        let mut assistant_ids = vec![];
        let user_suffix = sentencepiece_encode_vec("\n</user_inst>", text_tokenizer)?;
        assistant_ids.extend_from_slice(&user_suffix);
        assistant_ids.push(tts_cfg.im_end_token_id);
        let assistant_turn_ids = sentencepiece_encode_vec("\n", text_tokenizer)?;
        assistant_ids.extend_from_slice(&assistant_turn_ids);
        assistant_ids.push(tts_cfg.im_start_token_id);
        let assistant_role_ids = sentencepiece_encode_vec("assistant\n", text_tokenizer)?;
        assistant_ids.extend_from_slice(&assistant_role_ids);

        let none_ids = sentencepiece_encode_vec("None", text_tokenizer)?;

        Ok(Self {
            target_sample_rate,
            target_channels,
            audio_start_token_id: tts_cfg.audio_start_token_id,
            audio_end_token_id: tts_cfg.audio_end_token_id,
            audio_user_slot_token_id: tts_cfg.audio_user_slot_token_id,
            audio_assistant_slot_token_id: tts_cfg.audio_assistant_slot_token_id,
            audio_pad_token_id: tts_cfg.audio_pad_token_id,
            n_vq: tts_cfg.n_vq,
            prompt_token_ids,
            user_after_ids,
            assistant_ids,
            none_ids,
        })
    }

    pub fn resolved_mode(
        &self,
        mode: Option<MossTTSMode>,
        has_prompt_text: bool,
        has_prompt_audio: bool,
    ) -> Result<MossTTSMode> {
        let normalized_mode = mode.unwrap_or(MossTTSMode::VoiceClone);
        if normalized_mode == MossTTSMode::VoiceClone {
            if !has_prompt_audio {
                return Err(anyhow!("voice_clone mode requires prompt_audio_path"));
            }
            if has_prompt_text {
                println!("voice_clone mode does not accept prompt_text");
            }
        } else {
            if has_prompt_text != has_prompt_audio {
                return Err(anyhow!(
                    "continuation mode accepts either target text only, or prompt_text and prompt_audio_path together."
                ));
            }
        }
        Ok(normalized_mode)
    }

    pub fn build_inference_input_ids(
        &self,
        text: &str,
        prompt_audio_path: Option<&str>,
        prompt_text: Option<&str>,
        mode: MossTTSMode,
        audio_tokenizer: &MossAudioTokenizer,
        text_tokenizer: &SentencePieceProcessor,
        device: &Device,
    ) -> Result<Tensor> {
        let audio_code = if let Some(audio_path) = prompt_audio_path {
            let audio = load_audio_with_resample(
                audio_path,
                device,
                Some(self.target_sample_rate),
                Some(self.target_channels),
            )?;
            Some(audio_tokenizer.encode_one(&audio)?)
        } else {
            None
        };
        let text = &prepare_tts_text(text)?;
        let prompt_text = if let Some(prompt_text) = prompt_text {
            Some(prepare_tts_text(prompt_text)?)
        } else {
            None
        };
        // TODO: 长文本段切分
        if mode == MossTTSMode::VoiceClone
            && let Some(prompt_audio_codes) = &audio_code
        {
            let mut prompt_token_ids = vec![];
            prompt_token_ids.extend_from_slice(&self.prompt_token_ids);
            prompt_token_ids.push(self.audio_start_token_id);
            let prompt_ids_tensor = Self::build_text_raw(
                &prompt_token_ids,
                self.audio_pad_token_id,
                self.n_vq,
                device,
            )?;
            let text_token_ids = sentencepiece_encode_vec(text, text_tokenizer)?;
            let mut suffix_token_ids = vec![self.audio_end_token_id];
            suffix_token_ids.extend_from_slice(&self.user_after_ids);
            suffix_token_ids.extend_from_slice(&text_token_ids);
            suffix_token_ids.extend_from_slice(&self.assistant_ids);
            suffix_token_ids.push(self.audio_start_token_id);
            let audio_prefix_rows = Self::build_audio_prefix_rows(
                prompt_audio_codes,
                self.audio_user_slot_token_id,
                device,
            )?;
            let suffix_rows = Self::build_text_raw(
                &suffix_token_ids,
                self.audio_pad_token_id,
                self.n_vq,
                device,
            )?;
            let input_ids =
                Tensor::cat(&[&prompt_ids_tensor, &audio_prefix_rows, &suffix_rows], 0)?
                    .unsqueeze(0)?;
            Ok(input_ids)
        } else {
            let text = if let Some(prompt_text) = prompt_text {
                prompt_text + text
            } else {
                text.to_string()
            };
            let text_token_ids = sentencepiece_encode_vec(&text, text_tokenizer)?;
            let mut prompt_ids = vec![];
            prompt_ids.extend_from_slice(&self.prompt_token_ids);
            prompt_ids.extend_from_slice(&self.none_ids);
            prompt_ids.extend_from_slice(&self.user_after_ids);
            prompt_ids.extend_from_slice(&text_token_ids);
            prompt_ids.extend_from_slice(&self.assistant_ids);
            prompt_ids.push(self.audio_start_token_id);
            let mut input_ids =
                Self::build_text_raw(&prompt_ids, self.audio_pad_token_id, self.n_vq, device)?;
            if let Some(prompt_audio_codes) = &audio_code {
                let audio_prefix_rows = Self::build_audio_prefix_rows(
                    prompt_audio_codes,
                    self.audio_assistant_slot_token_id,
                    device,
                )?;
                input_ids = Tensor::cat(&[&input_ids, &audio_prefix_rows], 0)?;
            }
            input_ids = input_ids.unsqueeze(0)?;
            Ok(input_ids)
        }
    }

    fn build_audio_prefix_rows(
        prompt_audio_codes: &Tensor,
        slot_token_id: u32,
        device: &Device,
    ) -> Result<Tensor> {
        let audio_len = prompt_audio_codes.dim(0)?;
        let pad_tensor = Tensor::new(slot_token_id, device)?.broadcast_as((audio_len, 1))?;
        let rows = Tensor::cat(&[&pad_tensor, prompt_audio_codes], 1)?;
        Ok(rows)
    }

    fn build_text_raw(
        token_ids: &[u32],
        audio_pad_token_id: u32,
        n_vq: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let id_len = token_ids.len();
        //(1, len) -> (len, 1)
        let text_tensor = Tensor::from_slice(token_ids, (id_len, 1), device)?;
        // (len, n_vq)
        let pad_tensor = Tensor::new(audio_pad_token_id, device)?.broadcast_as((id_len, n_vq))?;
        let rows = Tensor::cat(&[text_tensor, pad_tensor], 1)?;
        Ok(rows)
    }
}
