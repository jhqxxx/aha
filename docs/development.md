# Development Guide

This guide covers contributing to AHA, including development setup, adding new models, and submitting contributions.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Model](#adding-a-new-model)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Contributions](#submitting-contributions)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- Rust 1.85+ (edition 2024)
- Git
- Optional: CUDA Toolkit, Metal, FFmpeg (for feature development)

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/jhqxxx/aha.git
cd aha

# Build in debug mode (faster compilation)
cargo build

# Build in release mode (better performance)
cargo build --release

# Build with features
cargo build --features cuda,flash-attn
cargo build --features metal
cargo build --features ffmpeg
```

### Development Workflow

```bash
# Run the CLI
cargo run -- -m qwen3-0.6b

# Run tests
cargo test

# Run specific test
cargo test test_qwen3vl_generate

# Run with logging
RUST_LOG=debug cargo run -- -m qwen3-0.6b

# Check code without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy
```

## Project Structure

```
aha/
├── Cargo.toml              # Build configuration
├── src/
│   ├── main.rs             # CLI entry point
│   ├── lib.rs              # Library exports
│   ├── api.rs              # HTTP API handlers
│   ├── exec/               # CLI command implementations
│   │   ├── mod.rs
│   │   ├── cli.rs
│   │   ├── serv.rs
│   │   ├── download.rs
│   │   └── run.rs
│   ├── models/             # Model implementations
│   │   ├── mod.rs          # Model factory and exports
│   │   ├── common/         # Shared model utilities
│   │   │   ├── config.rs
│   │   │   └── mod.rs
│   │   ├── qwen3vl/        # Qwen3-VL model
│   │   │   ├── config.rs
│   │   │   ├── model.rs
│   │   │   ├── generate.rs
│   │   │   ├── processor.rs
│   │   │   └── mod.rs
│   │   ├── voxcpm/         # VoxCPM model
│   │   └── ...             # Other models
│   ├── tokenizer/          # Tokenization utilities
│   ├── chat_template/      # Chat template handling
│   ├── position_embed/     # Positional embeddings
│   └── utils/              # Utility functions
│       ├── audio_utils.rs
│       ├── image_utils.rs
│       ├── tensor_utils.rs
│       └── mod.rs
├── tests/                  # Integration tests
│   ├── test_qwen2_5vl.rs
│   ├── test_qwen3vl.rs
│   └── ...
└── docs/                   # Documentation
```

## Adding a New Model

This section provides a step-by-step guide for adding a new model to AHA.

### Step 1: Create Model Directory

Create a new directory under `src/models/`:

```bash
mkdir -p src/models/newmodel
```

### Step 2: Implement Model Files

Create the following files in `src/models/newmodel/`:

#### config.rs

Define model configuration:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NewModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    // ... other config fields
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
}

```

#### model.rs

Implement the model architecture:

```rust
use candle::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::config::NewModelConfig;

pub struct NewModel {
    // Define model components
    config: NewModelConfig,
}

impl NewModel {
    pub fn load(vb: VarBuilder, config: &NewModelConfig) -> Result<Self> {
        // Load model weights
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Implement forward pass
        Ok(input_ids.clone())
    }
}
```

#### generate.rs

Implement the GenerateModel trait:

```rust
use std::path::Path;
use anyhow::Result;
use candle::Device;

use super::config::{GenerationConfig, NewModelConfig};
use aha::models::common::GenerateModel;

pub struct NewModelGenerate {
    // Define generate state
    model: super::model::NewModel,
    config: GenerationConfig,
    device: Device,
}

impl GenerateModel for NewModelGenerate {
    type Config = NewModelConfig;
    type GenerateConfig = GenerationConfig;

    fn init(
        model_path: &str,
        config: Option<Self::Config>,
        generate_config: Option<Self::GenerateConfig>,
    ) -> Result<Self> {
        // Load model from disk
        let device = Device::Cpu;
        let config = config.unwrap_or_default();
        let generate_config = generate_config.unwrap_or_default();

        // Load weights
        let model_path = Path::new(model_path);
        // ... load model implementation

        Ok(Self {
            model: todo!(),
            config: generate_config,
            device,
        })
    }

    fn generate(&mut self, prompt: &str) -> Result<String> {
        // Tokenize prompt
        // Run inference
        // Decode output
        Ok(prompt.to_string())
    }
}
```

#### processor.rs (optional)

For complex models with multimodal input:

```rust
use anyhow::Result;

pub struct NewModelProcessor {
    // Processing state
}

impl NewModelProcessor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_text(&self, text: &str) -> Result<Vec<String>> {
        // Process text input
        Ok(vec![text.to_string()])
    }

    pub fn process_image(&self, image_path: &str) -> Result<Tensor> {
        // Process image input
        todo!()
    }
}
```

#### mod.rs

Export the model:

```rust
mod config;
mod model;
mod generate;
pub mod processor;

pub use generate::NewModelGenerate;
pub use config::{GenerationConfig, NewModelConfig};
```

### Step 3: Register Model

Update `src/models/mod.rs`:

```rust
// Add to imports
pub mod newmodel;

// Add to WhichModel enum
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum WhichModel {
    // ... existing models
    NewModel,
}

// Add to model loading
pub fn load_model(
    model_type: &WhichModel,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn common::GenerateModel>> {
    match model_type {
        // ... existing models
        WhichModel::NewModel => {
            Ok(Box::new(newmodel::NewModelGenerate::init(model_path, None, None)?))
        }
    }
}
```

### Step 4: Update CLI

Update `src/main.rs` to include the new model in help text and CLI options if needed.

### Step 5: Add Tests

Create `tests/test_newmodel.rs`:

```rust
use anyhow::Result;

#[test]
fn test_newmodel_generate() -> Result<()> {
    let model_path = "path/to/test/model";
    let mut model = aha::models::newmodel::NewModelGenerate::init(model_path, None, None)?;
    
    let result = model.generate("Test prompt")?;
    assert!(!result.is_empty());
    
    Ok(())
}
```

### Step 6: Update Documentation

Update the following files:
- `README.md` - Add model to supported models list
- `docs/cli.md` - Add model ID to model list
- `docs/api.md` - Add model to supported models section
- `CHANGELOG.md` - Add entry for new model

## Multi-Artifact Integration Rules

When adding or extending model loaders, follow the current artifact architecture:

1. Use `LoadSpec` (`src/models/core/artifact.rs`) as the only input contract for loading.
2. Keep model identity (`WhichModel`) separate from artifact format (`ArtifactKind`).
3. Register supported formats in `supported_artifacts(model)`; do not expose unsupported formats.
4. Route initialization through family loader registry (`src/models/core/registry.rs`) instead of extending a large `match` in `load_model`.
5. Keep GGUF/ONNX path handling local-only (`gguf_path` / `onnx_path` / `tokenizer_dir`), and avoid adding remote download logic for these formats.
6. Maintain API compatibility of `ModelInstance` capabilities (`generate`, `generate_stream`, `embedding`, `rerank`) when adding new backends.

## Testing

### Unit Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_newmodel_generate

# Run with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4
```

### Integration Tests

Integration tests are located in `tests/`:

```bash
# Run specific integration test
cargo test --test test_qwen3vl
```

### Manual Testing

Test the CLI and API manually:

```bash
# Test CLI
cargo run -- -m newmodel

# Test API
cargo run -- -m newmodel -p 8080
curl http://127.0.0.1:8080/chat/completions -d '{...}'
```

### Model-Specific Tests

Each model should have tests covering:
- Model loading
- Basic inference
- Edge cases
- Error handling

## Code Style

### Formatting

```bash
# Format all code
cargo fmt

# Check formatting without making changes
cargo fmt --check
```

### Linting

```bash
# Run clippy
cargo clippy

# Fix clippy warnings
cargo clippy --fix
```

### Conventions

- Use 4 spaces for indentation
- Prefer `Result<T>` over `Option<T>` for errors
- Use `anyhow::Result` for application errors
- Document public APIs with rustdoc comments
- Keep functions focused and small
- Use meaningful variable names

### Documentation

```rust
/// Generates text using the NewModel.
///
/// # Arguments
///
/// * `prompt` - The input prompt text
/// * `max_tokens` - Maximum tokens to generate
///
/// # Returns
///
/// Generated text as a String
///
/// # Examples
///
/// ```no_run
/// let result = model.generate("Hello")?;
/// ```
pub fn generate(&mut self, prompt: &str) -> Result<String> {
    // Implementation
}
```

## Submitting Contributions

### Pull Request Process

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/new-model
   ```
3. Make your changes
4. Add tests
5. Update documentation
6. Ensure all tests pass:
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```
7. Commit and push:
   ```bash
   git commit -m "Add NewModel support"
   git push origin feature/new-model
   ```
8. Create a pull request on GitHub

### Pull Request Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code formatted (`cargo fmt`)
- [ ] Clippy warnings fixed (`cargo clippy`)
- [ ] All tests pass (`cargo test`)
- [ ] Commit messages follow conventions

### Commit Message Conventions

Follow conventional commits:

```
feat: add NewModel support
fix: correct tensor dimensions in Qwen3VL
docs: update installation guide
test: add integration test for VoxCPM
refactor: simplify model loading logic
perf: improve inference speed by 20%
```

## Release Process

### Version Bump

Update `Cargo.toml`:

```toml
[package]
version = "0.1.9"  # Bump version
```

### Update Changelog

Add entry to `CHANGELOG.md`:

```markdown
## [0.1.9] - 2025-02-06

### Added
- NewModel support

### Fixed
- Memory leak in image processing

### Changed
- Improved error messages
```

### Create Release

```bash
# Tag the release
git tag -a v0.1.9 -m "Release v0.1.9"
git push origin v0.1.9

# Build release artifacts
cargo build --release

# Publish to crates.io (if applicable)
cargo publish
```

## Debugging

### Logging

Enable debug logging:

```bash
RUST_LOG=debug cargo run -- -m qwen3-0.6b
```

Set specific module logging:

```bash
RUST_LOG=aha::models::qwen3vl=debug cargo run -- -m qwen3-0.6b
```

### Debugging Tests

```bash
# Print test output
cargo test -- --nocapture

# Show backtrace
RUST_BACKTRACE=1 cargo test
```

### Common Issues

#### Build Errors

- **Linking errors**: Install required system dependencies
- **CUDA errors**: Ensure CUDA toolkit is installed
- **Metal errors**: Check you're on Apple Silicon

#### Runtime Errors

- **Model not found**: Check model path and download
- **Out of memory**: Use smaller model or enable GPU
- **Slow inference**: Enable GPU acceleration

## Resources

- [Candle Documentation](https://github.com/huggingface/candle)
- [Rust Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## See Also

- [Architecture & Design](./concepts.md) - How AHA works
- [Installation](./installation.md) - Setup guide
- [API Reference](./api.md) - API documentation
