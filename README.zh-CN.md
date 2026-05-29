<p align="center">
  <img src="assets/img/logo.png" alt="aha logo" width="120"/>
</p>

<p align="center">
  <a href="https://github.com/jhqxxx/aha/stargazers">
    <img src="https://img.shields.io/github/stars/jhqxxx/aha" alt="GitHub Stars">
  </a>
  <a href="https://github.com/jhqxxx/aha/issues">
    <img src="https://img.shields.io/github/issues/jhqxxx/aha" alt="GitHub Issues">
  </a>
  <a href="https://github.com/jhqxxx/aha/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/jhqxxx/aha" alt="GitHub License">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

# aha

**轻量 AI 推理引擎 —— 文本、视觉、语音与 OCR 一站式解决方案**

aha 是一款基于 Rust 和 Candle 框架构建的高性能跨平台 AI 推理引擎。将最先进的 AI 模型带到您的本地机器——无需 API 密钥，无需云依赖，纯粹、快速的 AI，直接在您的硬件上运行。

### 支持的模型

| 类别 | 模型 |
|------|------|
| **文本** | Qwen3, MiniCPM4, MiniCPM5, LFM2, LFM2.5 |
| **视觉** | Qwen2.5-VL, Qwen3-VL, Qwen3.5 <br> LFM2.5-VL, LFM2-VL |
| **OCR** | DeepSeek-OCR, DeepSeek-OCR-2, PaddleOCR-VL,  <br>PaddleOCR-VL1.5, Hunyuan-OCR, GLM-OCR |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano, Qwen3-ASR |
| **TTS** | VoxCPM, VoxCPM1.5, VoxCPM2, Moss-TTS-Nano |
| **图像** | RMBG-2.0 (背景移除) |
| **嵌入** | Qwen3-Embedding, all-MiniLM-L6-v2 |
| **重排序** | Qwen3-Reranker |

## 更新日志
### 2026-05-29
- generate代码重构完成

### 2026-05-28
- generate代码重构进度 1/3

### 2026-05-27
- 新增 MiniCPM5

### 2026-05-24
- 更新文档

### 2026-05-11
- 添加Moss-TTS-Nano，该模型比较小，计算误差影响较大，效果比python原版差

### 2026-05-09
- 合并 pr/eastgold15/46, 添加 aha-ui


**[查看完整更新日志](docs/changelog.zh-CN.md)** →


## 为什么选择 aha？
- **🚀 高性能推理** - 基于 Candle 框架，提供高效的张量计算和模型推理
- **🔧 统一接口** — 一个工具搞定文本、视觉、语音和 OCR
- **📦 本地优先** — 所有处理在本地运行，数据不离境
- **🎯 跨平台** — 支持 Linux、macOS 和 Windows
- **⚡ GPU 加速** — 可选 CUDA 支持以获得更快推理
- **🛡️ 内存安全** — Rust 构建，稳定可靠
- **🧠 注意力优化** - 可选 Flash Attention 支持，优化长序列处理

## 快速开始

### 安装

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
cargo build --release
```

**可选特性：**

```bash
# CUDA (NVIDIA GPU 加速)
cargo build --release --features cuda

# Metal (Apple GPU 加速，适用于 macOS)
cargo build --release --features metal

# Flash Attention (更快推理)
cargo build --release --features cuda,flash-attn

# FFmpeg (多媒体处理)
cargo build --release --features ffmpeg
```

### CLI 快速参考

```bash

# 列出所有支持的模型
aha list

# 仅下载模型
aha download -m Qwen/Qwen3-ASR-0.6B

# 下载模型并启动服务
aha cli -m Qwen/Qwen3-ASR-0.6B

# 直接运行推理（无需启动服务）
aha run -m Qwen/Qwen3-ASR-0.6B -i "audio.wav"

# 本地运行 all-MiniLM-L6-v2 向量模型（原生 safetensors）
aha run -m all-minilm-l6-v2 -i "Rust embedding test" --weight-path D:\model_download\all-MiniLM-L6-v2

# 仅启动服务（模型已下载）
aha serv -m Qwen/Qwen3-ASR-0.6B -p 10100

```

### 对话

```bash
aha serv -m Qwen/Qwen3-0.6B -p 10100
```

然后使用统一(兼容 OpenAI)的 API：

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'
```

### aha-ui
```bash
cd aha-ui
``` 

#### 使用 npm
##### 安装 npm
参考 https://nodejs.org/en/download
```bash
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"
# Download and install Node.js:
nvm install 24
# Verify the Node.js version:
node -v # Should print "v24.15.0".
# Verify npm version:
npm -v # Should print "11.12.1".
```

##### npm 运行 aha-ui
```bash
# 确定在 aha-ui目录
# 确定 aha 编译完成
npm install
npm run tauri dev
```

##### npm 编译 & 安装 & 运行
```bash
npm run tauri build
# target in
# -- aha-ui/src-tauri/target/release/bundle/deb/aha-ui_0.1.0_amd64.deb
# -- aha-ui/src-tauri/target/release/bundle/rpm/aha-ui-0.1.0-1.x86_64.rpm
# -- aha-ui/src-tauri/target/release/bundle/appimage/aha-ui_0.1.0_amd64.AppImage
```

#### 使用 pnpm
##### 安装 pnpm
```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

##### pnpm 运行 aha-ui
```bash
# 确定在 aha-ui目录
# 确定 aha 编译完成
pnpm run tauri dev
```

##### pnpm 编译 & 安装 & 运行
```bash
pnpm run tauri build
# target in
# -- aha-ui/src-tauri/target/release/bundle/deb/aha-ui_0.1.0_amd64.deb
# -- aha-ui/src-tauri/target/release/bundle/rpm/aha-ui-0.1.0-1.x86_64.rpm
# -- aha-ui/src-tauri/target/release/bundle/appimage/aha-ui_0.1.0_amd64.AppImage
```

## 文档

| 文档 | 描述 |
|------|------|
| [快速入门](docs/getting-started.zh-CN.md) | aha 入门指南 |
| [安装指南](docs/installation.zh-CN.md) | 详细安装说明 |
| [CLI 参考](docs/cli.zh-CN.md) | 命令行界面 |
| [API 文档](docs/api.zh-CN.md) | 库与 REST API |
| [支持的模型](docs/supported-models.zh-CN.md) | 可用的 AI 模型 |
| [核心概念](docs/concepts.zh-CN.md) | 架构与设计 |
| [开发指南](docs/development.zh-CN.md) | 贡献指南 |
| [更新日志](docs/changelog.zh-CN.md) | 版本历史 |

## 开发

### aha 作为库使用
> cargo add aha

```rust
// VoxCPM示例
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav_mono;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/OpenBMB/VoxCPM2/";

    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    let generate = voxcpm_generate.inference(
        "aha是一个基于Rust和Candle框架的本地AI推理引擎，支持多模态模型（文本、视觉、语音、OCR）。".to_string(),
        None,
        None,
        2,
        1000,
        10,
        2.0,
        6.0,
    )?;

    save_wav_mono(&generate, "voxcpm2.wav", voxcpm_generate.sample_rate() as u32)?;
    Ok(())
}
```


### 扩展新的模型 

- 在src/models/创建新模型文件
- 在src/models/mod.rs中导出
- 在src/exec/中添加支持cli运行模型推理
- 在tests/中添加测试和示例


## 特性

- 基于 Candle 框架的高性能推理
- 多模态模型支持（视觉、语言、语音）
- 简洁易用的 API 设计
- 最小化依赖，紧凑的二进制文件
- Flash Attention 支持长序列处理
- FFmpeg 支持多媒体处理

## 许可证

Apache-2.0 &mdash; 详见 [LICENSE](LICENSE)

## 致谢

- [Candle](https://github.com/huggingface/candle) - 优秀的 Rust 机器学习框架
- 所有模型作者和贡献者

## 微信 & 捐赠
<div align="center">

| 微信群 | 捐赠 |
|--------------|--------|
| ![Wechat Group](./assets/img/aha_weixinqun.png) | ![Donate](./assets/img/donate.png) |

</div>

---

<p align="center">
  <sub>由 aha 团队用 ❤️ 构建</sub>
</p>

<p align="center">
  <sub>我们持续扩展支持的模型列表，欢迎贡献！</sub>
</p>

<p align="center">
  <sub>如果这个项目对你有帮助，请给我们一个 ⭐ Star！</sub>
</p>
