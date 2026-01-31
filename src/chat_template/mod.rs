use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use minijinja::{Environment, Value as MiniJinjaValue, context};

use crate::utils::string_to_static_str;

pub fn get_template(path: String) -> Result<String> {
    let tokenizer_config_file = path.clone() + "/tokenizer_config.json";
    assert!(
        std::path::Path::new(&tokenizer_config_file).exists(),
        "tokenizer_config.json not exists in model path"
    );
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tokenizer_config_file)?)
            .map_err(|e| anyhow!(format!("load tokenizer_config file error:{}", e)))?;
    let chat_template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or(anyhow!(format!("chat_template to str error")))?;
    // 修复模板中的问题行
    let fixed_template = chat_template
        .replace(
            "content.startswith('<tool_response>')",
            "content is startingwith('<tool_response>')", // 使用minijinja中的 is startingwith 替换
        )
        .replace(
            "content.endswith('</tool_response>')",
            "content is endingwith('</tool_response>')", // 使用minijinja中的 is endingwith 替换
        )
        .replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "((content | split('</think>'))[0] | rstrip('\\n') | split('<think>'))[-1] | lstrip('\\n')", // 使用自定义的split, rstrip, lstrip过滤器替换
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "(content | split('</think>'))[-1] | lstrip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "reasoning_content.strip('\\n')",
            "reasoning_content | strip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "content.lstrip('\\n')",
            "content | lstrip('\\n')", // 使用自定义的过滤器替换
        );
    Ok(fixed_template)
}

pub struct ChatTemplate<'a> {
    env: Environment<'a>,
}

impl<'a> ChatTemplate<'a> {
    pub fn init(path: &str) -> Result<Self> {
        let path: String = path.to_string();
        if !std::path::Path::new(&path).exists() {
            return Err(anyhow!("model path not found"));
        }
        let template = match get_template(path.clone()) {
            Ok(template) => template,
            Err(e) => {
                let jinja_path = path + "/chat_template.jinja";
                if !std::path::Path::new(&jinja_path).exists() {
                    return Err(anyhow!(
                        "get_template err {e} and chat_template.jinja not found"
                    ));
                }
                std::fs::read_to_string(&jinja_path)
                    .map_err(|e| anyhow!("Failed to read chat_template.jinja: {}", e))?
            }
        };
        let template = string_to_static_str(template);
        // 加载jinjaenv处理chat_template
        let mut env = Environment::new();
        // 添加自定义过滤器
        env.add_filter("tojson", |v: MiniJinjaValue| {
            serde_json::to_string(&v).unwrap()
        });

        env.add_filter("split", |s: String, delimiter: String| {
            s.split(&delimiter)
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
        });

        // 添加 lstrip 过滤器
        env.add_filter("lstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_start_matches(chars_str.as_str()).to_string(),
            None => s.trim_start().to_string(),
        });

        // 添加 rstrip 过滤器
        env.add_filter("rstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_end_matches(chars_str.as_str()).to_string(),
            None => s.trim_end().to_string(),
        });
        // let template = get_template(path.to_string())?;
        let _ = env.add_template("chat", template);

        Ok(Self { env })
    }

    pub fn apply_chat_template(&self, messages: &ChatCompletionParameters) -> Result<String> {
        let context = context! {
            messages => &messages.messages,
            tools => &messages.tools.as_ref(),
            add_generation_prompt => true,
        };
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        let message_str = template
            .render(context)
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        Ok(message_str)
    }
}
