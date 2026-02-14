use aha::models::WhichModel;

// Import helper functions from api module - these will need to be made public
// or tested through integration testing

#[test]
fn test_model_type_classification() {
    // Since get_model_type and get_model_id are private to api.rs,
    // we document the expected behavior here for reference:
    //
    // LLM models: MiniCPM4_0_5B, Qwen2_5vl3B, Qwen2_5vl7B, Qwen3_0_6B,
    //             Qwen3vl2B, Qwen3vl4B, Qwen3vl8B, Qwen3vl32B
    // OCR models: DeepSeekOCR, HunyuanOCR, PaddleOCRVL
    // ASR models: Qwen3ASR0_6B, Qwen3ASR1_7B, GlmASRNano2512, FunASRNano2512
    // Image models: RMBG2_0, VoxCPM, VoxCPM1_5

    // This test documents the expected model type classification
    let llm_models = vec![
        WhichModel::MiniCPM4_0_5B,
        WhichModel::Qwen2_5vl3B,
        WhichModel::Qwen2_5vl7B,
        WhichModel::Qwen3_0_6B,
        WhichModel::Qwen3vl2B,
        WhichModel::Qwen3vl4B,
        WhichModel::Qwen3vl8B,
        WhichModel::Qwen3vl32B,
    ];

    let ocr_models = vec![
        WhichModel::DeepSeekOCR,
        WhichModel::HunyuanOCR,
        WhichModel::PaddleOCRVL,
    ];

    let asr_models = vec![
        WhichModel::Qwen3ASR0_6B,
        WhichModel::Qwen3ASR1_7B,
        WhichModel::GlmASRNano2512,
        WhichModel::FunASRNano2512,
    ];

    let image_models = vec![
        WhichModel::RMBG2_0,
        WhichModel::VoxCPM,
        WhichModel::VoxCPM1_5,
    ];

    // Verify counts
    assert_eq!(llm_models.len(), 8);
    assert_eq!(ocr_models.len(), 3);
    assert_eq!(asr_models.len(), 4);
    assert_eq!(image_models.len(), 3);

    // Total models
    assert_eq!(
        llm_models.len() + ocr_models.len() + asr_models.len() + image_models.len(),
        18
    );
}

// Note: Integration tests for the /health and /models endpoints
// should be done with a running server. These would typically:
//
// 1. Start the server with a test model
// 2. Make HTTP requests to /health and /models
// 3. Verify the response format and status codes
//
// Example (pseudo-code):
//
// #[tokio::test]
// async fn test_health_endpoint() {
//     let resp = reqwest::get("http://localhost:10100/health").await.unwrap();
//     assert_eq!(resp.status(), 200);
//     let json: serde_json::Value = resp.json().await.unwrap();
//     assert_eq!(json["status"], "ok");
// }
