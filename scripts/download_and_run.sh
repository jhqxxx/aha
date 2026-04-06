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
    echo "  model_alias       The model alias to download (e.g., voxcpm, Qwen/Qwen2.5-VL-3B-Instruct)"
    echo ""
    echo "Available models:"
    echo "  sentence-transformers/all-MiniLM-L6-v2"
    echo "  LiquidAI/LFM2-1.2B"
    echo "  LiquidAI/LFM2.5-1.2B-Instruct"
    echo "  LiquidAI/LFM2.5-VL-1.6B"
    echo "  LiquidAI/LFM2-VL-1.6B"
    echo "  OpenBMB/MiniCPM4-0.5B"
    echo "  Qwen/Qwen2.5-VL-3B-Instruct"
    echo "  Qwen/Qwen2.5-VL-7B-Instruct" 
    echo "  Qwen/Qwen3-0.6B" 
    echo "  Qwen/Qwen3-1.7B" 
    echo "  Qwen/Qwen3-4B" 
    echo "  Qwen/Qwen3.5-0.8B" 
    echo "  Qwen/Qwen3.5-2B" 
    echo "  Qwen/Qwen3.5-4B" 
    echo "  Qwen/Qwen3.5-9B" 
    echo "  Qwen/Qwen3-ASR-0.6B"
    echo "  Qwen/Qwen3-ASR-1.7B"
    echo "  Qwen/Qwen3-Embedding-0.6B"
    echo "  Qwen/Qwen3-Embedding-4B"
    echo "  Qwen/Qwen3-Embedding-8B"
    echo "  Qwen/Qwen3-Reranker-0.6B"
    echo "  Qwen/Qwen3-Reranker-4B"
    echo "  Qwen/Qwen3-Reranker-8B"
    echo "  Qwen/Qwen3-VL-2B-Instruct"
    echo "  Qwen/Qwen3-VL-4B-Instruct"
    echo "  Qwen/Qwen3-VL-8B-Instruct"
    echo "  Qwen/Qwen3-VL-32B-Instruct"
    echo "  deepseek-ai/DeepSeek-OCR"
    echo "  deepseek-ai/DeepSeek-OCR-2"
    echo "  Tencent-Hunyuan/HunyuanOCR"
    echo "  PaddlePaddle/PaddleOCR-VL"
    echo "  PaddlePaddle/PaddleOCR-VL-1.5"
    echo "  AI-ModelScope/RMBG-2.0"
    echo "  OpenBMB/VoxCPM-0.5B"
    echo "  OpenBMB/VoxCPM1.5"
    echo "  ZhipuAI/GLM-ASR-Nano-2512"
    echo "  FunAudioLLM/Fun-ASR-Nano-2512"
    echo "  ZhipuAI/GLM-OCR"
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
    "sentence-transformers/all-MiniLM-L6-v2")
        MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
        ;;
    "LiquidAI/LFM2-1.2B")
        MODEL_ID="LiquidAI/LFM2-1.2B"
        ;;
    "LiquidAI/LFM2.5-1.2B-Instruct")
        MODEL_ID="LiquidAI/LFM2.5-1.2B-Instruct"
        ;;
    "LiquidAI/LFM2.5-VL-1.6B")
        MODEL_ID="LiquidAI/LFM2.5-VL-1.6B"
        ;;
    "LiquidAI/LFM2-VL-1.6B")
        MODEL_ID="LiquidAI/LFM2-VL-1.6B"
        ;;
    "OpenBMB/MiniCPM4-0.5B")
        MODEL_ID="OpenBMB/MiniCPM4-0.5B"
        ;;
    "Qwen/Qwen2.5-VL-3B-Instruct")
        MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
        ;;
    "Qwen/Qwen2.5-VL-7B-Instruct")
        MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
        ;; 
    "Qwen/Qwen3-0.6B")
        MODEL_ID="Qwen/Qwen3-0.6B"
        ;;
    "Qwen/Qwen3-1.7B")
        MODEL_ID="Qwen/Qwen3-1.7B"
        ;;
    "Qwen/Qwen3-4B")
        MODEL_ID="Qwen/Qwen3-4B"
        ;;
    "Qwen/Qwen3.5-0.8B")
        MODEL_ID="Qwen/Qwen3.5-0.8B"
        ;;
    "Qwen/Qwen3.5-2B")
        MODEL_ID="Qwen/Qwen3.5-2B"
        ;;
    "Qwen/Qwen3.5-4B")
        MODEL_ID="Qwen/Qwen3.5-4B"
        ;;
    "Qwen/Qwen3.5-9B")
        MODEL_ID="Qwen/Qwen3.5-9B"
        ;;
    "Qwen/Qwen3-ASR-0.6B")
        MODEL_ID="Qwen/Qwen3-ASR-0.6B"
        ;;
    "Qwen/Qwen3-ASR-1.7B")
        MODEL_ID="Qwen/Qwen3-ASR-1.7B"
        ;;
    "Qwen/Qwen3-Embedding-0.6B")
        MODEL_ID="Qwen/Qwen3-Embedding-0.6B"
        ;;
    "Qwen/Qwen3-Embedding-4B")
        MODEL_ID="Qwen/Qwen3-Embedding-4B"
        ;;
    "Qwen/Qwen3-Embedding-8B")
        MODEL_ID="Qwen/Qwen3-Embedding-8B"
        ;;
    "Qwen/Qwen3-Reranker-0.6B")
        MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
        ;;
    "Qwen/Qwen3-Reranker-4B")
        MODEL_ID="Qwen/Qwen3-Reranker-4B"
        ;;
    "Qwen/Qwen3-Reranker-8B")
        MODEL_ID="Qwen/Qwen3-Reranker-8B"
        ;;
    "Qwen/Qwen3-VL-2B-Instruct")
        MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
        ;;
    "Qwen/Qwen3-VL-4B-Instruct")
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
        ;;
    "Qwen/Qwen3-VL-8B-Instruct")
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
        ;;
    "Qwen/Qwen3-VL-32B-Instruct")
        MODEL_ID="Qwen/Qwen3-VL-32B-Instruct"
        ;;
    "deepseek-ai/DeepSeek-OCR")
        MODEL_ID="deepseek-ai/DeepSeek-OCR"
        ;;
    "deepseek-ai/DeepSeek-OCR-2")
        MODEL_ID="deepseek-ai/DeepSeek-OCR-2"
        ;;
    "Tencent-Hunyuan/HunyuanOCR")
        MODEL_ID="Tencent-Hunyuan/HunyuanOCR"
        ;;
    "PaddlePaddle/PaddleOCR-VL")
        MODEL_ID="PaddlePaddle/PaddleOCR-VL"
        ;;
    "PaddlePaddle/PaddleOCR-VL-1.5")
        MODEL_ID="PaddlePaddle/PaddleOCR-VL-1.5"
        ;;
    "AI-ModelScope/RMBG-2.0")
        MODEL_ID="briaai/RMBG-2.0"
        ;;
    "OpenBMB/VoxCPM-0.5B")
        MODEL_ID="openbmb/VoxCPM-0.5B"
        ;;
    "OpenBMB/VoxCPM1.5")
        MODEL_ID="openbmb/VoxCPM1.5"
        ;;
    "ZhipuAI/GLM-ASR-Nano-2512")
        MODEL_ID="zai-org/GLM-ASR-Nano-2512"
        ;;
    "FunAudioLLM/Fun-ASR-Nano-2512")
        MODEL_ID="FunAudioLLM/Fun-ASR-Nano-2512"
        ;;
    "ZhipuAI/GLM-OCR")
        MODEL_ID="zai-org/GLM-OCR"
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
