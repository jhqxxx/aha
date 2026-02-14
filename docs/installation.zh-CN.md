# 安装指南

本指南介绍如何在您的系统上安装和设置 AHA。

## 目录

- [前置要求](#前置要求)
- [安装方法](#安装方法)
- [平台特定说明](#平台特定说明)
- [功能特性](#功能特性)
- [验证安装](#验证安装)
- [故障排除](#故障排除)

## 前置要求

### 必需

- **Rust 工具链**：Rust 1.85 或更高版本（edition 2024）
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **Git**：用于克隆仓库
  ```bash
  # Ubuntu/Debian
  sudo apt-get install git
  
  # macOS
  brew install git
  
  # Windows
  # 从 https://git-scm.com/download/win 下载
  ```

### 可选（用于 FFmpeg 功能）

- **FFmpeg 开发库**：音频/视频处理所需

## 安装方法

### 方法 1：从源码构建

克隆仓库并构建：

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha

# 构建发布版本
cargo build --release

# 二进制文件位于 target/release/aha
```

### 方法 2：从 Crates.io 安装（当可用时）

```bash
cargo install aha
```

### 方法 3：启用特定功能构建

使用特定功能构建：

```bash
# 启用 CUDA 支持（NVIDIA GPU）
cargo build --release --features cuda

# 启用 Metal 支持（Apple Silicon）
cargo build --release --features metal

# 启用 Flash Attention
cargo build --release --features cuda,flash-attn

# 启用 FFmpeg 支持
cargo build --release --features ffmpeg
```

## 平台特定说明

### Linux

#### Ubuntu/Debian

```bash
# 安装构建依赖
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git clang cmake

# FFmpeg 功能所需
sudo apt-get install -y ffmpeg libavutil-dev libavcodec-dev \
    libavformat-dev libavfilter-dev libavdevice-dev \
    libswresample-dev libswscale-dev

# CUDA 支持，从 https://developer.nvidia.com/cuda-downloads 安装 CUDA toolkit
```

#### Fedora/RHEL

```bash
# 安装构建依赖
sudo dnf install gcc gcc-c++ make git clang pkg-config

# FFmpeg 功能所需
sudo dnf install ffmpeg-devel

# CUDA 支持
sudo dnf install cuda-devel
```

### macOS

#### Apple Silicon (M1/M2/M3/M4)

```bash
# 安装 Rust（如果尚未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装命令行工具
xcode-select --install

# FFmpeg 功能所需
brew install ffmpeg

# 启用 Metal 支持以进行 GPU 加速
cargo build --release --features metal
```

#### Intel Mac

```bash
# 安装 Rust（如果尚未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装命令行工具
xcode-select --install

# FFmpeg 功能所需
brew install ffmpeg

# CUDA 支持（如果您有 NVIDIA GPU）
# 从 https://developer.nvidia.com/cuda-downloads 安装 CUDA
cargo build --release --features cuda
```

### Windows

#### 使用 MSVC

```bash
# 从 https://rustup.rs/ 安装 Rust
# 从 https://visualstudio.microsoft.com/downloads/ 安装 Visual Studio Build Tools

# FFmpeg 功能
# 从 https://ffmpeg.org/download.html 下载 FFmpeg
# 设置 FFMPEG_DIR 环境变量指向您的 FFmpeg 安装目录

# 构建
cargo build --release
```

#### 使用 WSL2（推荐）

```bash
# 在 WSL2 中按照 Linux 说明操作
wsl
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git clang cmake
```

## 功能特性

AHA 支持多个可选功能：

### cuda

启用 NVIDIA GPU 加速的 CUDA 支持。

```bash
cargo build --release --features cuda
```

**要求**：
- NVIDIA GPU
- CUDA Toolkit 11.0 或更高版本
- cuDNN 库

**优势**：
- 推理速度提升 10-50 倍
- 支持更大的模型
- 降低 CPU 使用率

### metal

启用 Apple Silicon GPU 加速的 Metal 支持。

```bash
cargo build --release --features metal
```

**要求**：
- Apple Silicon (M1/M2/M3/M4)
- macOS 11.0 或更高版本

**优势**：
- 推理速度提升 5-20 倍
- 更低的功耗
- 支持更大的模型

### flash-attn

启用 Flash Attention 以优化长序列处理。

```bash
cargo build --release --features cuda,flash-attn
```

**要求**：
- 启用 CUDA 功能
- 支持的 GPU 架构（计算能力 7.0+）

**优势**：
- 减少内存使用
- 长序列推理更快
- 对视觉模型特别有益

**注意**：必须与 `cuda` 功能一起使用。

### ffmpeg

启用 FFmpeg 支持以进行音频/视频处理。

```bash
cargo build --release --features ffmpeg
```

**要求**：
- FFmpeg 开发库
- 特定平台（见上文）

**优势**：
- 扩展的音频格式支持（MP3、AAC 等）
- 视频处理能力
- 更好的音频重采样

### 组合功能

您可以组合多个功能：

```bash
# NVIDIA GPU 上的最佳性能
cargo build --release --features cuda,flash-attn

# 带音频支持的 Apple Silicon
cargo build --release --features metal,ffmpeg

# 启用所有功能
cargo build --release --features cuda,flash-attn,ffmpeg
```

## 验证安装

安装后，验证 AHA 是否正常工作：

```bash
# 检查版本
./target/release/aha --version

# 列出支持的模型
./target/release/aha list

# （如果已安装到 PATH）
aha --version
aha list
```

`aha list` 的预期输出：

```shell
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

## 故障排除

### 构建错误

#### "error: linking with cc failed"

通常表示缺少系统依赖。

**解决方案**：为您的平台安装所需的构建工具（见平台特定说明）。

#### "error: CUDA not found"

启用了 CUDA 功能但未安装 CUDA toolkit。

**解决方案**：
- 从 https://developer.nvidia.com/cuda-downloads 安装 CUDA toolkit
- 或不使用 CUDA 构建：`cargo build --release`

#### "error: Metal not available"

启用了 Metal 功能但硬件不支持。

**解决方案**：
- 确保您在 Apple Silicon 上
- 或不使用 Metal 构建：`cargo build --release`

### 运行时错误

#### "error while loading shared libraries"

缺少运行时库。

**解决方案**：安装所需的库（见平台特定说明）。

#### "Out of memory"

模型对于可用 RAM/VRAM 太大。

**解决方案**：
- 使用更小的模型
- 关闭其他应用程序
- 启用 GPU 加速以获得更好的内存效率

#### "Model download failed"

网络问题或磁盘空间不足。

**解决方案**：
- 检查互联网连接
- 确保 `~/.aha/` 中有足够的磁盘空间
- 重试：如果中断，下载将恢复

### 性能问题

#### 推理速度慢

**解决方案**：
1. 启用 GPU 加速：`--features cuda` 或 `--features metal`
2. 启用 Flash Attention：`--features "cuda,flash-attn"`
3. 使用更小的模型
4. 检查是否正在使用 GPU（应在监控工具中看到 GPU 使用率）

#### CPU 使用率高

**解决方案**：
1. 启用 GPU 加速
2. 减少批处理大小
3. 使用低精度模型

## 系统要求

*注：模型不同需求不同的硬件和软件，供参考*
### 最低要求

- **CPU**：x86_64 或 ARM64
- **RAM**：8 GB（推荐 16 GB）
- **磁盘**：10 GB 用于模型（因模型而异）
- **OS**：Linux、macOS 或 Windows

### 推荐要求

- **CPU**：现代多核处理器
- **RAM**：24 GB 或更多
- **GPU**：NVIDIA GPU（带 CUDA）或 Apple Silicon
- **磁盘**：具有 50+ GB 可用空间的 SSD
- **OS**：Linux (Ubuntu 22.04+) 或 macOS (Monterey+)

## 模型大小

流行模型的**大致**下载大小：

| 模型 | 大小 | RAM 使用 |
|------|------|----------|
| qwen3-0.6b | ~1.2 GB | ~2 GB |
| qwen3vl-2b | ~4 GB | ~6 GB |
| qwen3vl-8b | ~16 GB | ~20 GB |
| qwen3vl-32b | ~64 GB | ~70 GB |

## 后续步骤

成功安装后：

1. 阅读[快速入门指南](./getting-started.zh-CN.md)
2. 下载您的第一个模型：`aha download -m qwen3-0.6b`
3. 启动服务：`aha cli -m qwen3-0.6b`
4. 探索 [API 参考](./api.zh-CN.md)

## 另见

- [快速入门](./getting-started.zh-CN.md) - 快速入门指南
- [CLI 参考](./cli.zh-CN.md) - 命令行使用
- [API 参考](./api.zh-CN.md) - REST API 文档
- [开发指南](./development.zh-CN.md) - 贡献指南
