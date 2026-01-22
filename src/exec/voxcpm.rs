//! VoxCPM exec implementation for CLI `run` subcommand

use crate::exec::ExecModel;
use crate::models::voxcpm::generate::VoxCPMGenerate;
use anyhow::{Ok, Result};
use std::time::Instant;

pub struct VoxCPMExec;

impl ExecModel for VoxCPMExec {
    fn run(input: &str, output: Option<&str>, weight_path: &str) -> Result<()> {
        let target_text = if input.starts_with("file://") {
            let path = &input[7..];
            std::fs::read_to_string(path)?
        } else {
            input.to_string()
        };

        let i_start = Instant::now();
        let mut voxcpm_generate = VoxCPMGenerate::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let i_start = Instant::now();
        let audio = voxcpm_generate.inference(
            target_text,
            Some("啥子小师叔，打狗还要看主人，你再要继续，我就是你的对手".to_string()), //todo args
            Some("file://./assets/audio/voice_01.wav".to_string()),  //todo args
            2,
            100,   // max_len (voxcpm uses 100 vs voxcpm1.5's 4096)
            10,
            2.0,
            6.0,
        )?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        let output_path = if let Some(out) = output {
            out.to_string()
        } else {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            format!("voxcpm_{}.wav", timestamp)
        };

        let sample_rate = voxcpm_generate.sample_rate();
        crate::utils::audio_utils::save_wav(&audio, &output_path, sample_rate as u32)?;

        println!("Output saved to: {}", output_path);

        Ok(())
    }
}
