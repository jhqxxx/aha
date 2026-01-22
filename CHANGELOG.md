# Changelog


## [Unreleased]

### Added

- **CLI `run` Subcommand**: Direct model inference from CLI without HTTP service overhead:
  - `aha run` - Run model inference directly
  - `-m, --model <MODEL>` - Specify which model to use
  - `-in, --input <INPUT>` - Input text or file path (model-specific interpretation)
  - `-out, --output <OUTPUT>` - Output file path (optional, auto-generated if not specified)
  - `--weight-path <WEIGHT_PATH>` - Local model weight path (required)


### Changed

- **CLI Structure**: Refactored CLI to use clap's Subcommand feature while maintaining backward compatibility
- **Backward Compatibility**: Commands without subcommand now default to `cli` subcommand:
  - `aha -m qwen3vl-2b` is equivalent to `aha cli -m qwen3vl-2b`
  - All existing parameter options and defaults remain unchanged

### Technical Details

**Subcommand Parameters:**

`aha cli`:
- `-a, --address <ADDRESS>` - Server address (default: 127.0.0.1)
- `-p, --port <PORT>` - Server port (default: 10100)
- `-m, --model <MODEL>` - Model to use (required)
- `--weight-path <WEIGHT_PATH>` - Local model weight path (optional)
- `--save-dir <SAVE_DIR>` - Directory to save downloaded model (optional)
- `--download-retries <DOWNLOAD_RETRIES>` - Download retry attempts (default: 3)

`aha serv`:
- `-a, --address <ADDRESS>` - Server address (default: 127.0.0.1)
- `-p, --port <PORT>` - Server port (default: 10100)
- `-m, --model <MODEL>` - Model to use (required)
- `--weight-path <WEIGHT_PATH>` - Local model weight path (required)

`aha download`:
- `-m, --model <MODEL>` - Model to download (required)
- `-s, --save-dir <SAVE_DIR>` - Directory to save downloaded model (optional)
- `--download-retries <DOWNLOAD_RETRIES>` - Download retry attempts (default: 3)

**Code Changes:**
- Modified `src/main.rs` only
- Extracted common functions: `get_model_id()`, `start_http_server()`
- Reused existing `download_model()` and `init()` functions
- No changes to other modules or dependencies

## [0.1.8] - 2025-01-20

### Added

- Support for Fun-ASR-Nano-2512 model
- Support for Qwen3-0.6B model

## [0.1.7] - 2024-XX-XX

### Added

- Support for GLM-ASR-Nano-2512 model

## [0.1.6] - 2024-XX-XX

### Added

- Support for RMBG-2.0 model (background removal)

## [0.1.5] - 2024-XX-XX

### Added

- Support for VoxCPM1.5 model

## [0.1.4] - 2024-XX-XX

### Added

- Support for PaddleOCR-VL model

## [0.1.3] - 2024-XX-XX

### Added

- Support for Hunyuan-OCR model

## [0.1.2] - 2024-XX-XX

### Added

- Support for DeepSeek-OCR model

## [0.1.1] - 2024-XX-XX

### Added

- Support for Qwen3VL model family (2B, 4B, 8B, 32B)

## [0.1.0] - 2024-XX-XX

### Added

- Initial release
- Support for Qwen2.5VL models (3B, 7B)
- Support for MiniCPM4-0.5B model
- Support for VoxCPM-0.5B model
