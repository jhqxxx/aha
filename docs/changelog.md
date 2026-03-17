# Changelog

All notable changes to aha will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### 2026-03-17
- fix qwen3.5 position_ids create bug
- cli param add 
  - gguf_path: Local GGUF model weight path (required for loading models with GGUF)
  - mmproj_path: Local path to mmproj GGUF weights (required for multimodal model GGUF loading)
- WhichModel add qwen3.5-gguf

### 2026-03-16
- Added Qwen3.5 mmproj

### 2026-03-14
- update rust version 
- Added Qwen3.5 gguf support, but the 4B model still has issues; to be resolved.

## [0.2.2] (2026-03-07)
- Added GLM-OCR model

## [0.2.1] - (2026-03-05)
- Added Qwen3.5 model

### 2026-03-01
- update interpolate.rs

### 2026-02-24
- update candle version 0.9.2

## [0.2.0] - 2026-02-05

### Added
- Qwen3-ASR speech recognition model

## [0.1.9] - 2026-01-31

### Added
- CLI `list` subcommand to show supported models
- CLI subcommand structure support (`cli`, `serv`, `download`, `run`)
- Direct model inference via new `run` subcommand

### Fixed
- Qwen3VL thinking startswith bug
- `aha run` multiple inputs bug

## [0.1.8] - 2026-01-17

### Added
- Qwen3 text model support
- Fun-ASR-Nano-2512 speech recognition model

### Fixed
- ModelScope Fun-ASR-Nano model load error

### Changed
- Updated audio resampling with rubato

## [0.1.7] - 2026-01-07

### Added
- GLM-ASR-Nano-2512 speech recognition model
- Metal (GPU) support for Apple Silicon
- Dynamic home directory and model download script

## [0.1.6] - 2025-12-23

### Added
- RMBG-2.0 background removal model
- Image and audio API endpoints

### Changed
- Performance optimizations for RMBG2.0 image processing

## [0.1.5] - 2025-12-11

### Added
- VoxCPM1.5 voice generation model
- PaddleOCR-VL text recognition model

## [0.1.4] - 2025-12-09

### Added
- PaddleOCR-VL model support
- FFmpeg feature for multimedia processing

## [0.1.3] - 2025-12-03

### Added
- Hunyuan-OCR model support

## [0.1.2] - 2025-11-23

### Added
- DeepSeek-OCR model support

## [0.1.1] - 2025-11-12

### Added
- Qwen3-VL models (2B, 4B, 8B, 32B)

### Fixed
- Added serde default for tie_word_embeddings in Qwen3VL

## [0.1.0] - 2025-10-10

### Added
- Initial release
- Qwen2.5-VL model support
- MiniCPM4 model support
- VoxCPM voice generation model
- OpenAI-compatible REST API
- CLI interface for all model types
