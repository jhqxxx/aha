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
  <a href="https://github.com/jhqxxx/aha/actions">
    <img src="https://img.shields.io/badge/OpenAPI-Swagger%20UI-brightgreen" alt="OpenAPI">
  </a>
</p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a> | <strong>English</strong>
</p>

<p align="center">
  <strong>Official Website:</strong> <a href="https://s5dgj23f.pub.makeit.coderbox.cn/">https://s5dgj23f.pub.makeit.coderbox.cn/</a>
</p>

# aha

**Lightweight AI Inference Engine — All-in-one Solution for Text, Vision, Speech, and OCR**

aha is a high-performance, cross-platform AI inference engine built with Rust and the Candle framework. It brings state-of-the-art AI models to your local machine—no API keys, no cloud dependencies, just pure, fast AI running directly on your hardware.

> **✨ New: Multi-Model Parallel & OpenAPI Documentation** — Run multiple models simultaneously and explore the API via built-in Swagger UI!


### Supported Models

| Category | Models |
|----------|--------|
| **Text** | Qwen3, MiniCPM4, MiniCPM5, LFM2, LFM2.5 |
| **Vision** | Qwen2.5-VL, Qwen3-VL, Qwen3.5, <br> LFM2.5-VL, LFM2-VL |
| **OCR** | DeepSeek-OCR, DeepSeek-OCR-2 , PaddleOCR-VL <br> PaddleOCR-VL1.5, PaddleOCR-VL1.6, Hunyuan-OCR <br> GLM-OCR |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano, Qwen3-ASR |
| **TTS** | VoxCPM, VoxCPM1.5, VoxCPM2, Moss-TTS-Nano |
| **Image** | RMBG-2.0 (background removal) |
| **Embedding** | Qwen3-Embedding, all-MiniLM-L6-v2 |
| **Reranker** | Qwen3-Reranker |

## Changelog
### 2026-06-06
- add PaddleOCR-VL-1.6

### 2026-05-29
- generate code refactored

### 2026-05-28
- generate code refactoring progress 1/3

### 2026-05-27
- add MiniCPM5

### 2026-05-24
- update doc

### 2026-05-11
- add Moss-TTS-Nano，its performance is worse than the original Python version


**[View full changelog](docs/changelog.md)** →

## Why aha?
- **🚀 High-Performance Inference** — Powered by Candle framework for efficient tensor computation and model inference
- **🔧 Unified Interface** — One tool for text, vision, speech, and OCR
- **📦 Local-First** — All processing runs locally, no data leaves your machine
- **🎯 Cross-Platform** — Works on Linux, macOS, and Windows
- **⚡ GPU Accelerated** — Optional CUDA/Metal support for faster inference
- **🛡️ Memory Safe** — Built with Rust for reliability
- **✨ Multi-Model Parallel** — Load and run multiple models simultaneously with shared resources
- **📖 OpenAPI & Swagger UI** — Auto-generated API docs, explore and test endpoints interactively
- **🧠 Attention Optimization** — Optional Flash Attention support for optimized long sequence processing

## Changelog
### 2026-05-31
- OpenAPI 文档集成: 基于 utoipa + Swagger UI 自动生成 API 文档
- Multi-model parallel: 支持同时加载和运行多个模型，共享 Device、Tokenizer 缓存
- aha-ui 更新: 支持多模型选择启动、显示 OpenAPI URL 和已加载模型列表

### 2026-05-09
- merge pr/eastgold15/46, add aha-ui

### 2026-04-25
- VoxCPM update stream

### 2026-04-17
- Qwen3ASR add vad data recognition

### 2026-04-16
- fix FireRedVAD fsmn cache bug

### 2026-04-15
- add FireRedVAD

### 2026-04-10
- fix LiquidAI/LFM2.5-VL-450M chat_template load bug

### 2026-04-08
- add VoxCPM2

### 0.2.5 (2026-04-06)
- add qwen3-embedding/qwen3-reranker/all-minilm-l6-v2

**[View full changelog](docs/changelog.md)** →
## Quick Start

### Installation

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
cargo build --release
```

**Optional Features:**

```bash
# CUDA (NVIDIA GPU acceleration)
cargo build --release --features cuda

# Metal (Apple GPU acceleration for macOS)
cargo build --release --features metal

# Flash Attention (faster inference)
cargo build --release --features cuda,flash-attn

# FFmpeg (multimedia processing)
cargo build --release --features ffmpeg
```

### CLI Quick Reference

```bash

# List all supported models
aha list

# Download model only
aha download -m Qwen/Qwen3-ASR-0.6B

# Download model and start service
aha cli -m Qwen/Qwen3-ASR-0.6B

# Run inference directly (without starting service)
aha run -m Qwen/Qwen3-ASR-0.6B -i "audio.wav"

# Run local all-MiniLM-L6-v2 embedding (native safetensors)
aha run -m all-minilm-l6-v2 -i "Rust embedding test" --weight-path D:\model_download\all-MiniLM-L6-v2

# Start service only (model already downloaded)
aha serv -m Qwen/Qwen3-ASR-0.6B -p 10100

# ✨ Start multi-model service (load two models at once)
aha serv -m PaddleOCR-VL1.5 -m Qwen3-0.6B -p 10100
```

### ✨ Multi-Model Parallel

aha supports loading and running **multiple models simultaneously** — share the same GPU device and tokenizer cache across all models.

```bash
# Start ASR + OCR + LLM at the same time
aha serv -m GLM-ASR-Nano -m PaddleOCR-VL1.5 -m Qwen3-0.6B -p 10100
```

Each model is available via the same API endpoint, specify by the `model` field:

```bash
# Use different models with the same API
curl http://localhost:10100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "GLMASRNano2512",
  "messages": [{"role": "user", "content": "Transcribe this audio"}]
}'
```

### 📖 OpenAPI Documentation (Swagger UI)

When aha server is running, explore and test all API endpoints interactively:

- **Swagger UI**: [http://localhost:10100/swagger-ui/](http://localhost:10100/swagger-ui/)
- **OpenAPI JSON**: [http://localhost:10100/api-docs/openapi.json](http://localhost:10100/api-docs/openapi.json)

Swagger UI provides a complete interactive API explorer with request/response schemas, making integration effortless.

### Chat

```bash
aha serv -m Qwen/Qwen3-0.6B -p 10100
```

Then use the unified (OpenAI-compatible) API:

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }
'
```

### 🖥️ aha-ui (Tauri Desktop App)

aha ships with a modern Tauri desktop UI for visual model management:

```bash
cd aha-ui
```

**Using npm:**
```bash
# Install Node.js via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 24

# Run in dev mode
npm install
npm run tauri dev
```

**Using pnpm (recommended):**
```bash
# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Run in dev mode
pnpm run tauri dev

# Build for production
pnpm run tauri build
# Targets in:
# aha-ui/src-tauri/target/release/bundle/deb/aha-ui_0.1.0_amd64.deb
# aha-ui/src-tauri/target/release/bundle/rpm/aha-ui-0.1.0-1.x86_64.rpm
# aha-ui/src-tauri/target/release/bundle/appimage/aha-ui_0.1.0_amd64.AppImage
```

> **Note:** Make sure `aha` is compiled first, then run `aha-ui` from the `aha-ui` directory.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | First steps with aha |
| [Installation](docs/installation.md) | Detailed installation guide |
| [CLI Reference](docs/cli.md) | Command-line interface |
| [API Documentation](docs/api.md) | Library & REST API |
| [Multi-Model Guide](docs/multi-model-guide.md) | Multi-model parallel setup |
| [Supported Models](docs/supported-models.md) | Available AI models |
| [Concepts](docs/concepts.md) | Architecture & design |
| [Development](docs/development.md) | Contributing guide |
| [Changelog](docs/changelog.md) | Version history |

## API Endpoints

When the server is running, the following endpoints are available (all OpenAI-compatible where applicable):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/audio/transcriptions` | POST | ASR / Speech-to-Text |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/rerank` | POST | Document reranking |
| `/v1/models` | GET | List available models |
| `/audio/speech` | POST | TTS / Text-to-Speech |
| `/images/remove_background` | POST | Image background removal |
| `/admin/models/list` | GET | List loaded models (multi-model) |
| `/health` | GET | Health check |
| `/swagger-ui/` | GET | Interactive API documentation (Swagger UI) |
| `/api-docs/openapi.json` | GET | OpenAPI specification (JSON) |


## Development

### Using aha as a Library
> cargo add aha

```rust
// VoxCPM example
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

### Extending New Models

- Create new model file in src/models/
- Export in src/models/mod.rs
- Add support for CLI model inference in src/exec/
- Add tests and examples in tests/

## Features

- High-performance inference via Candle framework
- Multi-modal model support (vision, language, speech, OCR, ASR, TTS, embedding, reranking)
- **Multi-model parallel**: Load and run multiple models simultaneously with shared resources
- **Auto-generated OpenAPI docs**: Interactive Swagger UI for all endpoints
- Clean, easy-to-use API design (OpenAI-compatible)
- Minimal dependencies, compact binaries (~50MB)
- Flash Attention support for long sequences
- FFmpeg support for multimedia processing
- Tauri desktop UI for visual model management

## License

Apache-2.0 &mdash; See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — Excellent Rust ML framework
- [utoipa](https://github.com/juhaku/utoipa) — Auto-generated OpenAPI documentation
- All model authors and contributors

## Wechat & Donate
<div align="center">

| Wechat Group | Donate |
|--------------|--------|
| ![Wechat Group](./assets/img/aha_weixinqun.png) | ![Donate](./assets/img/donate.png) |

</div>

---

<p align="center">
  <sub>Built with ❤️ by the aha team</sub>
</p>

<p align="center">
  <sub>We're continuously expanding our model support. Contributions are welcome!</sub>
</p>
<p align="center">
  <sub>If this project helps you, please consider giving us a ⭐ Star!</sub>
</p>
