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
  <a href="README.zh-CN.md">简体中文</a> | <strong>English</strong>
</p>

# aha

**Lightweight AI Inference Engine — All-in-one Solution for Text, Vision, Speech, and OCR**

aha is a high-performance, cross-platform AI inference engine built with Rust and the Candle framework. It brings state-of-the-art AI models to your local machine—no API keys, no cloud dependencies, just pure, fast AI running directly on your hardware.

## Changelog
### 2026-03-01
- update interpolate.rs

### 2026-02-24
- update candle version 0.9.2

### v0.2.0 (2026-02-05)
- Added Qwen3-ASR speech recognition model

### v0.1.9 (2026-01-31)
- Added CLI `list` subcommand to show supported models
- Added CLI subcommand structure support (`cli`, `serv`, `download`, `run`)
- Fixed Qwen3VL thinking startswith bug
- Fixed `aha run` multiple inputs bug

### v0.1.8 (2026-01-17)
- Added Qwen3 text model support
- Added Fun-ASR-Nano-2512 speech recognition model
- Fixed ModelScope Fun-ASR-Nano model load error
- Updated audio resampling with rubato

### v0.1.7 (2026-01-07)
- Added GLM-ASR-Nano-2512 speech recognition model
- Merged Metal (GPU) support for Apple Silicon
- Added dynamic home directory and model download script

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
aha download -m qwen3asr-0.6b

# Download model and start service
aha -m qwen3asr-0.6b

# Run inference directly (without starting service)
aha run -m qwen3asr-0.6b -i "audio.wav"

# Start service only (model already downloaded)
aha serv -m qwen3asr-0.6b -p 10100

```

### Chat

```bash
aha serv -m qwen3-0.6b -p 10100
```

Then use the unified (OpenAI-compatible) API:

```bash
curl http://localhost:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }
'
```

### Supported Models

| Category | Models |
|----------|--------|
| **Text** | Qwen3, MiniCPM4 |
| **Vision** | Qwen2.5-VL, Qwen3-VL |
| **OCR** | DeepSeek-OCR, Hunyuan-OCR, PaddleOCR-VL |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano, Qwen3-ASR |
| **Audio** | VoxCPM, VoxCPM1.5 |
| **Image** | RMBG-2.0 (background removal) |

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | First steps with aha |
| [Installation](docs/installation.md) | Detailed installation guide |
| [CLI Reference](docs/cli.md) | Command-line interface |
| [API Documentation](docs/api.md) | Library & REST API |
| [Supported Models](docs/supported-models.md) | Available AI models |
| [Concepts](docs/concepts.md) | Architecture & design |
| [Development](docs/development.md) | Contributing guide |
| [Changelog](docs/changelog.md) | Version history |

## Why aha?
- **🚀 High-Performance Inference** - Powered by Candle framework for efficient tensor computation and model inference
- **🔧 Unified Interface** — One tool for text, vision, speech, and OCR
- **📦 Local-First** — All processing runs locally, no data leaves your machine
- **🎯 Cross-Platform** — Works on Linux, macOS, and Windows
- **⚡ GPU Accelerated** — Optional CUDA support for faster inference
- **🛡️ Memory Safe** — Built with Rust for reliability
- **🧠 Attention Optimization** - Optional Flash Attention support for optimized long sequence processing

## Development

### Using aha as a Library
> cargo add aha

```rust
# VoxCPM example
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/openbmb/VoxCPM-0.5B/";

    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;

    let generate = voxcpm_generate.generate(
        "The sun is shining bright, flowers smile at me, birds say early early early".to_string(),
        None,
        None,
        2,
        100,
        10,
        2.0,
        false,
        6.0,
    )?;

    let _ = save_wav(&generate, "voxcpm.wav")?;
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
- Multi-modal model support (vision, language, speech)
- Clean, easy-to-use API design
- Minimal dependencies, compact binaries
- Flash Attention support for long sequences
- FFmpeg support for multimedia processing

## License

Apache-2.0 &mdash; See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Excellent Rust ML framework
- All model authors and contributors

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
