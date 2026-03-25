# 开发指南

本指南涵盖为 AHA 做贡献，包括开发设置、添加新模型和提交贡献。

## 目录

- [开发设置](#开发设置)
- [项目结构](#项目结构)
- [添加新模型](#添加新模型)
- [测试](#测试)
- [代码风格](#代码风格)
- [提交贡献](#提交贡献)
- [发布流程](#发布流程)

## 开发设置

### 前置要求

- Rust 1.85+ (edition 2024)
- Git
- 可选：CUDA Toolkit、Metal、FFmpeg（用于功能开发）

### 克隆和构建

```bash
# 克隆仓库
git clone https://github.com/jhqxxx/aha.git
cd aha

# 调试模式构建（编译更快）
cargo build

# 发布模式构建（性能更好）
cargo build --release

# 使用功能构建
cargo build --features cuda,flash-attn
cargo build --features metal
cargo build --features ffmpeg
```

### 开发工作流程

```bash
# 运行 CLI
cargo run -- -m qwen3-0.6b

# 运行测试
cargo test

# 运行特定测试
cargo test test_qwen3vl_generate

# 启用日志运行
RUST_LOG=debug cargo run -- -m qwen3-0.6b

# 检查代码而不构建
cargo check

# 格式化代码
cargo fmt

# 运行 linter
cargo clippy
```

## 项目结构

```
aha/
├── Cargo.toml              # 构建配置
├── src/
│   ├── main.rs             # CLI 入口点
│   ├── lib.rs              # 库导出
│   ├── api.rs              # HTTP API 处理程序
│   ├── exec/               # CLI 命令实现
│   │   ├── mod.rs
│   │   ├── cli.rs
│   │   ├── serv.rs
│   │   ├── download.rs
│   │   └── run.rs
│   ├── models/             # 模型实现
│   │   ├── mod.rs          # 模型工厂和导出
│   │   ├── common/         # 共享模型工具
│   │   │   ├── config.rs
│   │   │   └── mod.rs
│   │   ├── qwen3vl/        # Qwen3-VL 模型
│   │   │   ├── config.rs
│   │   │   ├── model.rs
│   │   │   ├── generate.rs
│   │   │   ├── processor.rs
│   │   │   └── mod.rs
│   │   ├── voxcpm/         # VoxCPM 模型
│   │   └── ...             # 其他模型
│   ├── tokenizer/          # 分词工具
│   ├── chat_template/      # 聊天模板处理
│   ├── position_embed/     # 位置编码
│   └── utils/              # 工具函数
│       ├── audio_utils.rs
│       ├── image_utils.rs
│       ├── tensor_utils.rs
│       └── mod.rs
├── tests/                  # 集成测试
│   ├── test_qwen2_5vl.rs
│   ├── test_qwen3vl.rs
│   └── ...
└── docs/                   # 文档
```

## 添加新模型

本节提供了向 AHA 添加新模型的分步指南。

### 步骤 1：创建模型目录

在 `src/models/` 下创建新目录：

```bash
mkdir -p src/models/newmodel
```

### 步骤 2：实现模型文件

在 `src/models/newmodel/` 中创建以下文件：

#### config.rs

定义模型配置：

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NewModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    // ... 其他配置字段
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
}

```

#### model.rs

实现模型架构：

```rust
use candle::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::config::NewModelConfig;

pub struct NewModel {
    // 定义模型组件
    config: NewModelConfig,
}

impl NewModel {
    pub fn load(vb: VarBuilder, config: &NewModelConfig) -> Result<Self> {
        // 加载模型权重
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // 实现前向传播
        Ok(input_ids.clone())
    }
}
```

#### generate.rs

实现 GenerateModel trait：

```rust
use std::path::Path;
use anyhow::Result;
use candle::Device;

use super::config::{GenerationConfig, NewModelConfig};
use aha::models::common::GenerateModel;

pub struct NewModelGenerate {
    // 定义生成状态
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
        // 从磁盘加载模型
        let device = Device::Cpu;
        let config = config.unwrap_or_default();
        let generate_config = generate_config.unwrap_or_default();

        // 加载权重
        let model_path = Path::new(model_path);
        // ... 加载模型实现

        Ok(Self {
            model: todo!(),
            config: generate_config,
            device,
        })
    }

    fn generate(&mut self, prompt: &str) -> Result<String> {
        // 分词提示
        // 运行推理
        // 解码输出
        Ok(prompt.to_string())
    }
}
```

#### processor.rs（可选）

对于具有多模态输入的复杂模型：

```rust
use anyhow::Result;

pub struct NewModelProcessor {
    // 处理状态
}

impl NewModelProcessor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_text(&self, text: &str) -> Result<Vec<String>> {
        // 处理文本输入
        Ok(vec![text.to_string()])
    }

    pub fn process_image(&self, image_path: &str) -> Result<Tensor> {
        // 处理图像输入
        todo!()
    }
}
```

#### mod.rs

导出模型：

```rust
mod config;
mod model;
mod generate;
pub mod processor;

pub use generate::NewModelGenerate;
pub use config::{GenerationConfig, NewModelConfig};
```

### 步骤 3：注册模型

更新 `src/models/mod.rs`：

```rust
// 添加到导入
pub mod newmodel;

// 添加到 WhichModel 枚举
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum WhichModel {
    // ... 现有模型
    NewModel,
}

// 添加到模型加载
pub fn load_model(
    model_type: &WhichModel,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn common::GenerateModel>> {
    match model_type {
        // ... 现有模型
        WhichModel::NewModel => {
            Ok(Box::new(newmodel::NewModelGenerate::init(model_path, None, None)?))
        }
    }
}
```

### 步骤 4：更新 CLI

更新 `src/main.rs` 以在帮助文本和 CLI 选项中包含新模型（如果需要）。

### 步骤 5：添加测试

创建 `tests/test_newmodel.rs`：

```rust
use anyhow::Result;

#[test]
fn test_newmodel_generate() -> Result<()> {
    let model_path = "path/to/test/model";
    let mut model = aha::models::newmodel::NewModelGenerate::init(model_path, None, None)?;
    
    let result = model.generate("测试提示")?;
    assert!(!result.is_empty());
    
    Ok(())
}
```

### 步骤 6：更新文档

更新以下文件：
- `README.md` - 将模型添加到支持的模型列表
- `docs/cli.md` - 将模型 ID 添加到模型列表
- `docs/api.md` - 将模型添加到支持的模型部分
- `CHANGELOG.md` - 为新模型添加条目

## 多制品接入规则

为模型新增或扩展加载后端时，请遵循当前制品架构：

1. 统一使用 `LoadSpec`（`src/models/core/artifact.rs`）作为加载输入契约。
2. 保持模型身份（`WhichModel`）与制品格式（`ArtifactKind`）解耦。
3. 在 `supported_artifacts(model)` 中声明支持矩阵，不要提前暴露未实现格式。
4. 通过 family loader registry（`src/models/core/registry.rs`）分发初始化，不再扩展 `load_model` 的大 `match`。
5. GGUF/ONNX 保持本地路径输入（`gguf_path` / `onnx_path` / `tokenizer_dir`），不引入远程下载管理逻辑。
6. 新后端接入时保持 `ModelInstance` 能力接口语义兼容（`generate`、`generate_stream`、`embedding`、`rerank`）。

## 测试

### 单元测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_newmodel_generate

# 启用输出运行
cargo test -- --nocapture

# 并行运行测试
cargo test -- --test-threads=4
```

### 集成测试

集成测试位于 `tests/` 中：

```bash
# 运行特定集成测试
cargo test --test test_qwen3vl
```

### 手动测试

手动测试 CLI 和 API：

```bash
# 测试 CLI
cargo run -- -m newmodel

# 测试 API
cargo run -- -m newmodel -p 8080
curl http://127.0.0.1:8080/chat/completions -d '{...}'
```

### 模型特定测试

每个模型都应该有涵盖以下内容的测试：
- 模型加载
- 基本推理
- 边缘情况
- 错误处理

## 代码风格

### 格式化

```bash
# 格式化所有代码
cargo fmt

# 检查格式化而不进行更改
cargo fmt --check
```

### Linting

```bash
# 运行 clippy
cargo clippy

# 修复 clippy 警告
cargo clippy --fix
```

### 约定

- 使用 4 个空格缩进
- 对于错误，优先使用 `Result<T>` 而不是 `Option<T>`
- 对于应用程序错误使用 `anyhow::Result`
- 使用 rustdoc 注释记录公共 API
- 保持函数专注和小型
- 使用有意义的变量名

### 文档

```rust
/// 使用 NewModel 生成文本。
///
/// # 参数
///
/// * `prompt` - 输入提示文本
/// * `max_tokens` - 要生成的最大令牌数
///
/// # 返回
///
/// 生成的文本作为字符串
///
/// # 示例
///
/// ```no_run
/// let result = model.generate("你好")?;
/// ```
pub fn generate(&mut self, prompt: &str) -> Result<String> {
    // 实现
}
```

## 提交贡献

### Pull Request 流程

1. Fork 仓库
2. 创建功能分支：
   ```bash
   git checkout -b feature/new-model
   ```
3. 进行更改
4. 添加测试
5. 更新文档
6. 确保所有测试通过：
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```
7. 提交并推送：
   ```bash
   git commit -m "添加 NewModel 支持"
   git push origin feature/new-model
   ```
8. 在 GitHub 上创建 pull request

### Pull Request 清单

- [ ] 已添加/更新测试
- [ ] 已更新文档
- [ ] 已更新 CHANGELOG.md
- [ ] 代码已格式化（`cargo fmt`）
- [ ] 已修复 Clippy 警告（`cargo clippy`）
- [ ] 所有测试通过（`cargo test`）
- [ ] 提交消息遵循约定

### 提交消息约定

遵循约定式提交：

```
feat: 添加 NewModel 支持
fix: 修正 Qwen3VL 中的张量维度
docs: 更新安装指南
test: 添加 VoxCPM 集成测试
refactor: 简化模型加载逻辑
perf: 将推理速度提高 20%
```

## 发布流程

### 版本升级

更新 `Cargo.toml`：

```toml
[package]
version = "0.1.9"  # 升级版本
```

### 更新变更日志

将条目添加到 `CHANGELOG.md`：

```markdown
## [0.1.9] - 2025-02-06

### 新增
- NewModel 支持

### 修复
- 图像处理中的内存泄漏

### 更改
- 改进错误消息
```

### 创建发布

```bash
# 标记发布
git tag -a v0.1.9 -m "Release v0.1.9"
git push origin v0.1.9

# 构建发布产物
cargo build --release

# 发布到 crates.io（如果适用）
cargo publish
```

## 调试

### 日志

启用调试日志：

```bash
RUST_LOG=debug cargo run -- -m qwen3-0.6b
```

设置特定模块日志：

```bash
RUST_LOG=aha::models::qwen3vl=debug cargo run -- -m qwen3-0.6b
```

### 调试测试

```bash
# 打印测试输出
cargo test -- --nocapture

# 显示回溯
RUST_BACKTRACE=1 cargo test
```

### 常见问题

#### 构建错误

- **链接错误**：安装所需的系统依赖
- **CUDA 错误**：确保已安装 CUDA toolkit
- **Metal 错误**：检查您是否在 Apple Silicon 上

#### 运行时错误

- **未找到模型**：检查模型路径和下载
- **内存不足**：使用更小的模型或启用 GPU
- **推理速度慢**：启用 GPU 加速

## 资源

- [Candle 文档](https://github.com/huggingface/candle)
- [Rust 指南](https://rust-lang.github.io/api-guidelines/)
- [约定式提交](https://www.conventionalcommits.org/)

## 另见

- [架构与设计](./concepts.zh-CN.md) - AHA 的工作原理
- [安装](./installation.zh-CN.md) - 设置指南
- [API 参考](./api.zh-CN.md) - API 文档
