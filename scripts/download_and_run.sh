#!/bin/bash
set -e

# Default values
# MIRROR_URL="https://hf-mirror.com"
DEFAULT_SAVE_DIR="$HOME/.aha"
MODEL_ALIAS=""

# Help function
show_help() {
    echo "Usage: $0 [model_alias]"
    echo ""
    echo "Arguments:"
    echo "  model_alias       The model alias to download (e.g., voxcpm, qwen2.5vl-3b)"
    echo ""
    echo "Available models:"
    echo "  minicpm4-0.5b"
    echo "  qwen2.5vl-3b"
    echo "  qwen2.5vl-7b" 
    echo "  qwen3-0.6b" 
    echo "  qwen3-reranker-0.6b"
    echo "  qwen3-reranker-4b"
    echo "  qwen3-reranker-8b"
    echo "  qwen3asr-0.6b"
    echo "  qwen3asr-1.7b"
    echo "  qwen3vl-2b"
    echo "  qwen3vl-4b"
    echo "  qwen3vl-8b"
    echo "  qwen3vl-32b"
    echo "  deepseek-ocr"
    echo "  hunyuan-ocr"
    echo "  paddleocr-vl"
    echo "  rmbg2.0"
    echo "  voxcpm"
    echo "  voxcpm1.5"
    echo "  glm-asr-nano-2512"
    echo "  fun-asr-nano-2512"
    echo ""
    exit 1
}

# Check if model alias is provided
if [ -z "$1" ]; then
    show_help
fi

MODEL_ALIAS=$1

# Map alias to Repo ID
MODEL_ID=""
case $MODEL_ALIAS in
    "minicpm4-0.5b")
        MODEL_ID="OpenBMB/MiniCPM4-0.5B"
        ;;
    "qwen2.5vl-3b")
        MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
        ;;
    "qwen2.5vl-7b")
        MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    "qwen3-0.6b")
        MODEL_ID="Qwen/Qwen3-0.6B"
        ;;
    "qwen3-reranker-0.6b")
        MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
        ;;
    "qwen3-reranker-4b")
        MODEL_ID="Qwen/Qwen3-Reranker-4B"
        ;;
    "qwen3-reranker-8b")
        MODEL_ID="Qwen/Qwen3-Reranker-8B"
        ;;
    "qwen3asr-0.6b")
        MODEL_ID="Qwen/Qwen3-ASR-0.6B"
        ;;
    "qwen3asr-1.7b")
        MODEL_ID="Qwen/Qwen3-ASR-1.7B"
        ;;
    "qwen3vl-2b")
        MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
        ;;
    "qwen3vl-4b")
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
        ;;
    "qwen3vl-8b")
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
        ;;
    "qwen3vl-32b")
        MODEL_ID="Qwen/Qwen3-VL-32B-Instruct"
        ;;
    "deepseek-ocr")
        MODEL_ID="deepseek-ai/DeepSeek-OCR"
        ;;
    "hunyuan-ocr")
        MODEL_ID="Tencent-Hunyuan/HunyuanOCR"
        ;;
    "paddleocr-vl")
        MODEL_ID="PaddlePaddle/PaddleOCR-VL"
        ;;
    "rmbg2.0")
        MODEL_ID="briaai/RMBG-2.0"
        ;;
    "voxcpm")
        MODEL_ID="openbmb/VoxCPM-0.5B"
        ;;
    "voxcpm1.5")
        MODEL_ID="openbmb/VoxCPM1.5"
        ;;
    "glm-asr-nano-2512")
        MODEL_ID="zai-org/GLM-ASR-Nano-2512"
        ;;
    "fun-asr-nano-2512")
        MODEL_ID="FunAudioLLM/Fun-ASR-Nano-2512"
        ;;
    *)
        echo "Error: Unknown model alias '$MODEL_ALIAS'"
        show_help
        ;;
esac

echo "Selected Model: $MODEL_ALIAS"
echo "Repo ID: $MODEL_ID"
echo "Target Directory: $DEFAULT_SAVE_DIR/$MODEL_ID"

# Prepare environment for acceleration (Default to mirror if not set)
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT=$MIRROR_URL
    echo "Using default HF Mirror: $HF_ENDPOINT"
else
    echo "Using custom HF Endpoint: $HF_ENDPOINT"
fi

# Check if huggingface-cli and hf_transfer are installed
CLI_CMD=""
if command -v huggingface-cli &> /dev/null; then
    CLI_CMD="huggingface-cli"
elif command -v hf &> /dev/null; then
    CLI_CMD="hf"
else
    echo "huggingface-cli not found. Installing via pip..."
    if command -v pip3 &> /dev/null; then
        pip3 install -U "huggingface_hub[cli]" hf_transfer
    elif command -v pip &> /dev/null; then
        pip install -U "huggingface_hub[cli]" hf_transfer
    else
        echo "Error: pip is not available. Please install python and pip."
        exit 1
    fi
    CLI_CMD="huggingface-cli"
fi

# Try to install hf_transfer if simple python check fails (optional but recommended for speed)
if ! python3 -c "import hf_transfer" &> /dev/null; then
     echo "Installing hf_transfer for faster downloads..."
     pip3 install -U hf_transfer || echo "Warning: hf_transfer install failed, falling back to standard download."
fi

# Enable HF Transfer (Rust-based downloader) - Default to 1 (On) unless explicitly disabled
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
if [ "$HF_HUB_ENABLE_HF_TRANSFER" == "1" ]; then
    echo "Enabled HF_HUB_ENABLE_HF_TRANSFER for high-speed download"
else
    echo "HF_HUB_ENABLE_HF_TRANSFER disabled. Using standard Python downloader."
fi

# Download model
echo "Downloading model with acceleration via $CLI_CMD..."
# Note: --resume-download is deprecated/implicit in newer versions, removing it. 
# --local-dir-use-symlinks matches standard CLI if version is recent. 
if ! $CLI_CMD download "$MODEL_ID" --local-dir "$DEFAULT_SAVE_DIR/$MODEL_ID" --local-dir-use-symlinks False $TOKEN_ARG; then
    echo ""
    echo "Error: Download failed."
    echo "If you received a 429 Rate Limit error, please provide a Hugging Face Token."
    echo "Usage: ./download_and_run.sh $MODEL_ALIAS [hf_token]"
    echo "Or set the HF_TOKEN environment variable."
    exit 1
fi

# Run cargo
echo "Starting application with cargo..."
# Detect if on Mac to add 'metal' feature
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected MacOS. Running with 'metal' feature..."
    cargo run -r --features metal -- --model "$MODEL_ALIAS" --weight-path "$DEFAULT_SAVE_DIR/$MODEL_ID"
else
    echo "Running with default features (cuda)..."
    cargo run -r --features cuda -- --model "$MODEL_ALIAS" --weight-path "$DEFAULT_SAVE_DIR/$MODEL_ID"
fi
