use anyhow::Result;

use crate::exec::ExecModel;
use crate::models::{LoadSpec, all_minilm_l6_v2::generate::AllMiniLML6V2Model};
use crate::utils::get_file_path;

pub struct AllMiniLML6V2Exec;

impl AllMiniLML6V2Exec {
    pub fn run_with_spec(input: &[String], output: Option<&str>, spec: &LoadSpec) -> Result<()> {
        let input_text = input
            .first()
            .ok_or_else(|| anyhow::anyhow!("embedding run requires one text input"))?;
        let text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let mut model = AllMiniLML6V2Model::init_from_spec(spec, None, None)?;
        let embedding = model.embed(&[text])?;

        let output_json = serde_json::to_string_pretty(&embedding)?;
        println!("{}", output_json);
        if let Some(out) = output {
            std::fs::write(out, output_json)?;
            println!("Output saved to: {}", out);
        }
        Ok(())
    }
}

impl ExecModel for AllMiniLML6V2Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(crate::models::WhichModel::AllMiniLML6V2, weight_path);
        Self::run_with_spec(input, output, &spec)
    }
}
