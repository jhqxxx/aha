use crate::utils::{download_model, get_default_save_dir};

pub async fn download_index_tts2_need_model(save_dir: Option<&str>) -> anyhow::Result<()> {
    let save_dir = match save_dir {
        Some(dir) => dir.to_string(),
        None => get_default_save_dir().expect("Failed to get home directory"),
    };

    let w2v_bert2_0 = "facebook/w2v-bert-2.0";
    let mask_gct = "amphion/MaskGCT";
    // let campplus= "funasr/campplus"; // huggingface
    let campplus = "iic/speech_campplus_sv_zh-cn_16k-common"; // modelscope
    download_model(w2v_bert2_0, &save_dir, 3).await?;
    download_model(mask_gct, &save_dir, 3).await?;
    download_model(campplus, &save_dir, 3).await?;

    Ok(())
}
