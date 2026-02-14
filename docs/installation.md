# Installation Guide

This guide covers installing and setting up AHA on your system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Feature Flags](#feature-flags)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **Rust toolchain**: Rust 1.85 or later (edition 2024)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **Git**: For cloning the repository
  ```bash
  # Ubuntu/Debian
  sudo apt-get install git
  
  # macOS
  brew install git
  
  # Windows
  # Download from https://git-scm.com/download/win
  ```

### Optional (for FFmpeg feature)

- **FFmpeg development libraries**: Required for audio/video processing

## Installation Methods

### Method 1: Build from Source

Clone the repository and build:

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha

# Build release version
cargo build --release

# The binary will be at target/release/aha
```

### Method 2: Install from Crates.io (when available)

```bash
cargo install aha
```

### Method 3: Install with Features

Build with specific features enabled:

```bash
# With CUDA support (NVIDIA GPUs)
cargo build --release --features cuda

# With Metal support (Apple Silicon)
cargo build --release --features metal

# With Flash Attention
cargo build --release --features cuda,flash-attn

# With FFmpeg support
cargo build --release --features ffmpeg
```

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git clang cmake

# For FFmpeg feature
sudo apt-get install -y ffmpeg libavutil-dev libavcodec-dev \
    libavformat-dev libavfilter-dev libavdevice-dev \
    libswresample-dev libswscale-dev

# For CUDA support, install CUDA toolkit
# See https://developer.nvidia.com/cuda-downloads
```

#### Fedora/RHEL

```bash
# Install build dependencies
sudo dnf install gcc gcc-c++ make git clang pkg-config

# For FFmpeg feature
sudo dnf install ffmpeg-devel

# For CUDA support
sudo dnf install cuda-devel
```

### macOS

#### Apple Silicon (M1/M2/M3/M4)

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install command line tools
xcode-select --install

# For FFmpeg feature
brew install ffmpeg

# Build with Metal support for GPU acceleration
cargo build --release --features metal
```

#### Intel Mac

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install command line tools
xcode-select --install

# For FFmpeg feature
brew install ffmpeg

# For CUDA support (if you have NVIDIA GPU)
# Install CUDA from https://developer.nvidia.com/cuda-downloads
cargo build --release --features cuda
```

### Windows

#### Using MSVC

```bash
# Install Rust from https://rustup.rs/
# Install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/

# For FFmpeg feature
# Download FFmpeg from https://ffmpeg.org/download.html
# Set FFMPEG_DIR environment variable to your FFmpeg installation

# Build
cargo build --release
```

#### Using WSL2 (Recommended)

```bash
# Follow Linux instructions inside WSL2
wsl
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git clang cmake
```

## Feature Flags

aha supports several optional features:

### cuda

Enables CUDA support for NVIDIA GPU acceleration.

```bash
cargo build --release --features cuda
```

**Requirements**:
- NVIDIA GPU
- CUDA Toolkit 11.0 or later
- cuDNN library

**Benefits**:
- 10-50x faster inference
- Support for larger models
- Lower CPU usage

### metal

Enables Metal support for Apple Silicon GPU acceleration.

```bash
cargo build --release --features metal
```

**Requirements**:
- Apple Silicon (M1/M2/M3/M4)
- macOS 11.0 or later

**Benefits**:
- 5-20x faster inference
- Lower power consumption
- Support for larger models

### flash-attn

Enables Flash Attention for optimized long-sequence processing.

```bash
cargo build --release --features cuda,flash-attn
```

**Requirements**:
- CUDA feature enabled
- Supported GPU architecture (compute capability 7.0+)

**Benefits**:
- Reduced memory usage
- Faster inference for long sequences
- Especially beneficial for vision models

**Note**: Must be used with `cuda` feature.

### ffmpeg

Enables FFmpeg support for audio/video processing.

```bash
cargo build --release --features ffmpeg
```

**Requirements**:
- FFmpeg development libraries
- Platform-specific (see above)

**Benefits**:
- Extended audio format support (MP3, AAC, etc.)
- Video processing capabilities
- Better audio resampling

### Combining Features

You can combine multiple features:

```bash
# Maximum performance on NVIDIA GPU
cargo build --release --features cuda,flash-attn

# Apple Silicon with audio support
cargo build --release --features metal,ffmpeg

# Everything enabled
cargo build --release --features cuda,flash-attn,ffmpeg
```

## Verification

After installation, verify that AHA is working:

```bash
# Check version
./target/release/aha --version

# List supported models
./target/release/aha list

# (Or if installed to PATH)
aha --version
aha list
```

Expected output for `aha list`:

```
#Supported models:
 
 Available models:

Model Name                     ModelScope ID
-----------------------------------------------------------
minicpm4-0.5b                  OpenBMB/MiniCPM4-0.5B
qwen2.5vl-3b                   Qwen/Qwen2.5-VL-3B-Instruct
qwen2.5vl-7b                   Qwen/Qwen2.5-VL-7B-Instruct
qwen3-0.6b                     Qwen/Qwen3-0.6B
qwen3asr-0.6b                  Qwen/Qwen3-ASR-0.6B
qwen3asr-1.7b                  Qwen/Qwen3-ASR-1.7B
qwen3vl-4b                     Qwen/Qwen3-VL-2B-Instruct
qwen3vl-4b                     Qwen/Qwen3-VL-4B-Instruct
qwen3vl-8b                     Qwen/Qwen3-VL-8B-Instruct
qwen3vl-32b                    Qwen/Qwen3-VL-32B-Instruct
deepseek-ocr                   deepseek-ai/DeepSeek-OCR
hunyuan-ocr                    Tencent-Hunyuan/HunyuanOCR
paddleocr-vl                   PaddlePaddle/PaddleOCR-VL
rmbg2.0                        AI-ModelScope/RMBG-2.0
voxcpm                         OpenBMB/VoxCPM-0.5B
voxcpm1.5                      OpenBMB/VoxCPM1.5
glm-asr-nano-2512              ZhipuAI/GLM-ASR-Nano-2512
fun-asr-nano-2512              FunAudioLLM/Fun-ASR-Nano-2512

```

## Troubleshooting

### Build Errors

#### "error: linking with cc failed"

This usually indicates missing system dependencies.

**Solution**: Install required build tools for your platform (see Platform-Specific Instructions).

#### "error: CUDA not found"

CUDA feature is enabled but CUDA toolkit is not installed.

**Solution**: 
- Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads
- Or build without CUDA: `cargo build --release`

#### "error: Metal not available"

Metal feature is enabled but not on supported hardware.

**Solution**:
- Ensure you're on Apple Silicon
- Or build without Metal: `cargo build --release`

### Runtime Errors

#### "error while loading shared libraries"

Missing runtime libraries.

**Solution**: Install required libraries (see Platform-Specific Instructions).

#### "Out of memory"

Model is too large for available RAM/VRAM.

**Solution**:
- Use a smaller model
- Close other applications
- Enable GPU acceleration for better memory efficiency

#### "Model download failed"

Network issue or insufficient disk space.

**Solution**:
- Check internet connection
- Ensure sufficient disk space in `~/.aha/`
- Try again: download will resume if interrupted

### Performance Issues

#### Slow inference

**Solutions**:
1. Enable GPU acceleration: `--features cuda` or `--features metal`
2. Enable Flash Attention: `--features "cuda,flash-attn"`
3. Use a smaller model
4. Check if GPU is being used (should see GPU usage in monitoring tools)

#### High CPU usage

**Solutions**:
1. Enable GPU acceleration
2. Reduce batch size
3. Use model with lower precision

## System Requirements
*Different models require different hardware and software, for reference.*

### Minimum Requirements

- **CPU**: x86_64 or ARM64
- **RAM**: 8 GB (16 GB recommended)
- **Disk**: 10 GB for models (varies by model)
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 32 GB or more
- **GPU**: NVIDIA GPU (with CUDA) or Apple Silicon
- **Disk**: SSD with 50+ GB free space
- **OS**: Linux (Ubuntu 22.04+) or macOS (Monterey+)

## Model Sizes

Approximate download sizes for popular models:

| Model | Size | RAM Usage |
|-------|------|-----------|
| qwen3-0.6b | ~1.2 GB | ~2 GB |
| qwen3vl-2b | ~4 GB | ~6 GB |
| qwen3vl-8b | ~16 GB | ~20 GB |
| qwen3vl-32b | ~64 GB | ~70 GB |

## Next Steps

After successful installation:

1. Read the [Getting Started Guide](./getting-started.md)
2. Download your first model: `aha download -m qwen3-0.6b`
3. Start the service: `aha cli -m qwen3-0.6b`
4. Explore the [API Reference](./api.md)

## See Also

- [Getting Started](./getting-started.md) - Quick start guide
- [CLI Reference](./cli.md) - Command-line usage
- [API Reference](./api.md) - REST API documentation
- [Development](./development.md) - Contributing guide
