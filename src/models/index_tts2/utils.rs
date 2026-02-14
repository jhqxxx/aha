use anyhow::{Result, anyhow};
use regex::Regex;

use crate::utils::{download_model, get_default_save_dir};

pub async fn download_index_tts2_need_model(save_dir: Option<&str>) -> Result<()> {
    let save_dir = match save_dir {
        Some(dir) => dir.to_string(),
        None => get_default_save_dir().expect("Failed to get home directory"),
    };

    let w2v_bert2_0 = "facebook/w2v-bert-2.0";
    let mask_gct = "amphion/MaskGCT";
    // let campplus= "funasr/campplus"; // huggingface
    let campplus = "iic/speech_campplus_sv_zh-cn_16k-common"; // modelscope
    // let bigvgan = "nvidia/bigvgan_v2_22khz_80band_256x"; // huggingface
    let bigvgan = "nv-community/bigvgan_v2_22khz_80band_256x"; // modelscope
    download_model(w2v_bert2_0, &save_dir, 3).await?;
    download_model(mask_gct, &save_dir, 3).await?;
    download_model(campplus, &save_dir, 3).await?;
    download_model(bigvgan, &save_dir, 3).await?;

    Ok(())
}

#[derive(Debug, Clone)]
pub struct TextNormalizer {
    // char_rep_map: HashMap<String, String>,
    // zh_char_rep_map: HashMap<String, String>,
    pinyin_tone_pattern: Regex,
    // name_pattern: Regex,
    // tech_term_pattern: Regex,
    // english_contraction_pattern: Regex,
    email_pattern: Regex,
    cjk_range_pattern: Regex,
}

impl TextNormalizer {
    pub fn new() -> Result<Self> {
        let pinyin_tone_pattern: Regex = Regex::new(
                // r"(?i)(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüvAEIOUV]|[aeAE]i|u[aiouAIUO]|aoAO|ouOU|i[aeuAEU]|[uüvUÜV]e|[uvüUVÜ]ang?|uaiUAI|[aeiuvAEIUV]n|[aeioAEIO]ng|ia[noNAO]|i[aA][oO]ng)|ngNG|erER)([1-5])"
                r"(?i)((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüvAEIOUV]|[aeAE]i|u[aiouAIUO]|aoAO|ouOU|i[aeuAEU]|[uüvUÜV]e|[uvüUVÜ]ang?|uaiUAI|[aeiuvAEIUV]n|[aeioAEIO]ng|ia[noNAO]|i[aA][oO]ng)|ngNG|erER)([1-5])"
            ).map_err(|e| anyhow!(format!("new pinyin_tone_pattern regex error:{}", e)))?;
        // let name_pattern: Regex =
        //     Regex::new(r"[\u{4e00}-\u{9fff}]+(?:[-·—][\u{4e00}-\u{9fff}]+){1,2}")
        //         .map_err(|e| anyhow!(format!("new name_pattern regex error:{}", e)))?;
        // let tech_term_pattern: Regex = Regex::new(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")
        //     .map_err(|e| anyhow!(format!("new tech_term_pattern regex error:{}", e)))?;
        // let english_contraction_pattern: Regex = Regex::new(
        //     r"(?i)(what|where|who|which|how|t?here|it|s?he|that|this)'s",
        // )
        // .map_err(|e| anyhow!(format!("new english_contraction_pattern regex error:{}", e)))?;
        let email_pattern: Regex = Regex::new(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$")
            .map_err(|e| anyhow!(format!("new email_pattern regex error:{}", e)))?;
        let cjk_range_pattern: Regex = Regex::new(r"([\u{1100}-\u{11ff}\u{2e80}-\u{a4cf}\u{a840}-\u{d7af}\u{f900}-\u{faff}\u{fe30}-\u{fe4f}\u{ff65}-\u{ffdc}\u{20000}-\u{2ffff}])")
            .map_err(|e| anyhow!(format!("new cjk_range_pattern regex error:{}", e)))?;
        // let mut char_rep_map = HashMap::new();
        // char_rep_map.insert("：".to_string(), ",".to_string());
        // char_rep_map.insert("；".to_string(), ",".to_string());
        // char_rep_map.insert(";".to_string(), ",".to_string());
        // char_rep_map.insert("，".to_string(), ",".to_string());
        // char_rep_map.insert("。".to_string(), ".".to_string());
        // char_rep_map.insert("！".to_string(), "!".to_string());
        // char_rep_map.insert("？".to_string(), "?".to_string());
        // char_rep_map.insert("\n".to_string(), " ".to_string());
        // char_rep_map.insert("·".to_string(), "-".to_string());
        // char_rep_map.insert("、".to_string(), ",".to_string());
        // char_rep_map.insert("...".to_string(), "…".to_string());
        // char_rep_map.insert(",,,".to_string(), "…".to_string());
        // char_rep_map.insert("，，，".to_string(), "…".to_string());
        // char_rep_map.insert("……".to_string(), "…".to_string());
        // char_rep_map.insert("“".to_string(), "'".to_string());
        // char_rep_map.insert("”".to_string(), "'".to_string());
        // char_rep_map.insert("\"".to_string(), "'".to_string());
        // char_rep_map.insert("‘".to_string(), "'".to_string());
        // char_rep_map.insert("’".to_string(), "'".to_string());
        // char_rep_map.insert("（".to_string(), "'".to_string());
        // char_rep_map.insert("）".to_string(), "'".to_string());
        // char_rep_map.insert("(".to_string(), "'".to_string());
        // char_rep_map.insert(")".to_string(), "'".to_string());
        // char_rep_map.insert("《".to_string(), "'".to_string());
        // char_rep_map.insert("》".to_string(), "'".to_string());
        // char_rep_map.insert("【".to_string(), "'".to_string());
        // char_rep_map.insert("】".to_string(), "'".to_string());
        // char_rep_map.insert("[".to_string(), "'".to_string());
        // char_rep_map.insert("]".to_string(), "'".to_string());
        // char_rep_map.insert("—".to_string(), "-".to_string());
        // char_rep_map.insert("～".to_string(), "-".to_string());
        // char_rep_map.insert("~".to_string(), "-".to_string());
        // char_rep_map.insert("「".to_string(), "'".to_string());
        // char_rep_map.insert("」".to_string(), "'".to_string());
        // char_rep_map.insert(":".to_string(), ",".to_string());

        // let mut zh_char_rep_map = char_rep_map.clone();
        // zh_char_rep_map.insert("$".to_string(), ".".to_string());
        Ok(Self {
            // char_rep_map,
            // zh_char_rep_map,
            pinyin_tone_pattern,
            // name_pattern,
            // tech_term_pattern,
            // english_contraction_pattern,
            email_pattern,
            cjk_range_pattern,
        })
    }

    pub fn match_email(&self, email: &str) -> bool {
        self.email_pattern.is_match(email)
    }

    pub fn use_chinese(&self, s: &str) -> bool {
        let has_chinese = s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        let has_alpha = s.chars().any(|c| c.is_alphabetic());
        let is_email = self.match_email(s);

        if has_chinese || !has_alpha || is_email {
            return true;
        }

        self.pinyin_tone_pattern.is_match(s)
    }

    pub fn tokenize_by_cjk_char(&self, line: &str, do_upper_case: bool) -> String {
        // Split the line by CJK characters
        let parts: Vec<&str> = self.cjk_range_pattern.split(line.trim()).collect();
        // Process each part and join with spaces
        let mut result_parts = Vec::new();
        for part in parts {
            if !part.trim().is_empty() {
                if do_upper_case {
                    result_parts.push(part.trim().to_uppercase());
                } else {
                    result_parts.push(part.trim().to_string());
                }
            }
        }
        // Join the parts with spaces
        result_parts.join(" ")
    }
}

pub fn tokenize_by_cjk_char(line: &str, do_upper_case: bool) -> String {
    let mut result_parts = Vec::new();
    for ch in line.chars() {
        if ('\u{1100}'..='\u{11ff}').contains(&ch)
            || ('\u{2e80}'..='\u{a4cf}').contains(&ch)
            || ('\u{a840}'..='\u{d7af}').contains(&ch)
            || ('\u{f900}'..='\u{faff}').contains(&ch)
            || ('\u{fe30}'..='\u{fe4f}').contains(&ch)
            || ('\u{ff65}'..='\u{ffdc}').contains(&ch)
            || ('\u{20000}'..='\u{2ffff}').contains(&ch)
            || ('\u{4e00}'..='\u{9fff}').contains(&ch)
        {
            // CJK 字符
            if do_upper_case {
                result_parts.push(ch.to_uppercase().collect::<String>());
            } else {
                result_parts.push(ch.to_string());
            }
        } else {
            // 非 CJK 字符，保持在一起
            if do_upper_case {
                result_parts.push(ch.to_uppercase().to_string());
            } else {
                result_parts.push(ch.to_string());
            }
        }
    }

    result_parts.join(" ")
}
