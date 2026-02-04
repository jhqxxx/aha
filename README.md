# aha 
一个基于 Candle 框架的 Rust 模型推理库，提供高效、易用的多模态模型推理能力。

## 特性
* 🚀 高性能推理 - 基于 Candle 框架，提供高效的张量计算和模型推理
* 🎯 多模型支持 - 集成视觉、语言和语音多模态模型
* 🔧 易于使用 - 简洁的 API 设计，快速上手
* 🛡️ 内存安全 - 得益于 Rust 的所有权系统，确保内存安全
* 📦 轻量级 - 最小化依赖，编译产物小巧
* ⚡ GPU 加速 - 可选 CUDA 支持
* 🧠 注意力优化 - 可选 Flash Attention 支持，优化长序列处理

## 支持的模型
### 当前已实现
* [Qwen2.5VL](https://huggingface.co/collections/Qwen/qwen25-vl) - 阿里通义千问 2.5 多模态大语言模型
    - 模型：[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 开源协议：[Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
    - 模型：[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [MiniCPM4](https://huggingface.co/collections/openbmb/minicpm4) - 面壁智能 MiniCPM 系列语言模型
    - 模型：
    	- huggingface: [MiniCPM4-0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B)  开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
        - AtomGit: [MiniCPM4-0.5B](https://ai.atomgit.com/OpenBMB/MiniCPM4-0.5B)
* [VoxCPM](https://huggingface.co/collections/openbmb/voxcpm) - 面壁智能语音生成模型
    - 模型：
    	- huggingface: [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
        - AtomGit: [VoxCPM-0.5B](https://ai.atomgit.com/OpenBMB/VoxCPM-0.5B)
* [Qwen3VL](https://huggingface.co/collections/Qwen/qwen3-vl) - 阿里通义千问 3 多模态大语言模型
    - 模型：[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
    - 模型：[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
    - 模型：[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
    - 模型：[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* DeepSeek-OCR - 深度求索光学文字识别模型
    - 模型：[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) 开源协议：[MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
* Hunyuan-OCR - 腾讯混元光学文字识别模型
    - 模型：
    	- huggingface: [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) 开源协议：[TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE)
        - AtomGit: [HunyuanOCR](https://ai.atomgit.com/tencent_hunyuan/HunyuanOCR)
* [PaddleOCR-VL](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl) - 百度飞桨光学文字识别模型
    - 模型：
    	- huggingface: [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
        - AtomGit: [PaddleOCR-VL](https://ai.atomgit.com/paddlepaddle/PaddleOCR-VL)
* [VoxCPM1.5](https://huggingface.co/collections/openbmb/voxcpm) - 面壁智能语音生成模型1.5版本
    - 模型：
    	- huggingface: [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
        - AtomGit: [VoxCPM1.5](https://ai.atomgit.com/OpenBMB/VoxCPM1.5)
* [RMBG2.0](https://huggingface.co/collections/briaai/rmbg) - RMBGv2.0由BRIA AI开发，供非商业用途使用。
    - 模型：[RMBG2.0](https://huggingface.co/briaai/RMBG-2.0) 开源协议：[Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
* GLM-ASR-Nano-2512 - 智谱AI语音识别模型
    - 模型：
    	- huggingface: [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) 开源协议：[MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
        - AtomGit: [GLM-ASR-Nano-2512](https://ai.atomgit.com/zai-org/GLM-ASR-Nano-2512)
* Fun-ASR-Nano-2512 - 通义百聆语音识别模型
    - 模型:[Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) 开源协议未标明
* [Qwen3](https://huggingface.co/collections/Qwen/qwen3) - 通义千问 Qwen3系列语言模型
    - 模型:[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 开源协议: [Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
* [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) - 通义千问 Qwen3语音识别模型
    - 模型:[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)    
    - 模型:[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 开源协议：[Apache license 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)

## 计划支持
我们持续扩展支持的模型列表，欢迎贡献！

⭐ 如果这个项目对你有帮助，请给我们一个 Star！

## 环境依赖
* 启用ffmpeg的feature时：
    * ubuntu/WSL
    ```bash
    sudo apt-get update
    sudo apt-get install -y clang pkg-config ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libswresample-dev libswscale-dev
    ```
    * windows参考： https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building

## 功能特性
项目提供了几个可选的功能特性，您可以根据需要启用它们：
* flash-attn: 启用 Flash Attention 支持以提升模型推理性能：
```bash
cargo build -r --features flash-attn
```

* cuda: 为 candle 核心组件启用 CUDA 支持，实现 Nvidia GPU 加速计算：
```bash
cargo build -r --features cuda
```

* cuda: 为 candle 核心组件启用 Metal 支持，利用 Apple GPU 加速计算（适用于 macOS/iOS 平台）：
```bash
cargo build -r --features metal
```

* ffmpeg: 启用 FFmpeg 支持，提供多媒体处理功能：
```bash
cargo build -r --features ffmpeg
```
* 组合使用功能特性

```bash
# 同时启用 CUDA 和 Flash Attention 以获得最佳性能
cargo build -r --features "cuda,flash-attn"
```
## 命令行工具 (CLI)
项目提供命令行工具，支持模型下载、服务启动和直接推理等多种操作。

### 快速开始
```bash
# 下载模型并启动服务
aha -m qwen3asr-0.6b

# 直接运行推理（无需启动服务）
aha run -m qwen3asr-0.6b -i "audio.wav" --weight-path /path/to/model

# 仅下载模型
aha download -m qwen3asr-0.6b
```
### 主要功能
* 模型管理 - 自动下载、本地模型加载
* 服务模式 - 启动 HTTP API 服务
* 直接推理 - 命令行直接执行推理任务
* 多模型支持 - 支持aha已实现的模型类型

### 详细使用说明
请参阅 [CLI_USAGE.md](./docs/CLI_USAGE.md) 获取完整的命令行工具使用说明、参数详解和支持的模型列表。

## 安装及使用

### 从源码构建部署
```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
```

#### cargo run 运行参数说明
##### 基本用法
```bash
cargo run -F cuda -r -- [参数]
```
##### 参数详解
1. 端口设置
-----
    -p, --port <PORT>
* 设置HTTP服务监听的端口号
* 默认值：10100
* 示例：--port 8080 或 -p 8080

2. 模型选择（必选）
-----
    -m, --model <MODEL>
* 指定要加载的模型类型
* 可选值：
    * minicpm4-0.5b：OpenBMB/MiniCPM4-0.5B 模型
    * qwen2.5vl-3b：Qwen/Qwen2.5-VL-3B-Instruct 模型
    * qwen2.5vl-7b：Qwen/Qwen2.5-VL-7B-Instruct 模型
    * qwen3-0.6b: Qwen/Qwen3-0.6B 模型
    * qwen3asr-0.6b: Qwen/Qwen3-ASR-0.6B 模型
    * qwen3asr-1.7b: Qwen/Qwen3-ASR-1.7B 模型
    * qwen3vl-2b：Qwen/Qwen3-VL-2B-Instruct 模型
    * qwen3vl-4b：Qwen/Qwen3-VL-4B-Instruct 模型
    * qwen3vl-8b：Qwen/Qwen3-VL-8B-Instruct 模型
    * qwen3vl-32b：Qwen/Qwen3-VL-32B-Instruct 模型
    * deepseek-ocr: deepseek-ai/DeepSeek-OCR 模型
    * hunyuan-ocr: Tencent-Hunyuan/HunyuanOCR 模型
    * paddleocr-vl: PaddlePaddle/PaddleOCR-VL 模型
    * rmbg2.0: AI-ModelScope/RMBG-2.0 模型
    * voxcpm: OpenBMB/VoxCPM-0.5B 模型
    * voxcpm1.5: OpenBMB/VoxCPM1.5 模型
    * glm-asr-nano-2512: ZhipuAI/GLM-ASR-Nano-2512 模型
    * fun-asr-nano-2512: FunAudioLLM/Fun-ASR-Nano-2512 模型
* 示例：--model deepseek-ocr 或 -m qwen3vl-2b

3. 权重路径
-----
    --weight-path <WEIGHT_PATH>
* 指定本地模型权重文件路径
* 如果指定此参数，则跳过模型下载步骤
* 示例：--weight-path /path/to/model/dir

4. 保存路径
-----
    --save-dir <SAVE_DIR>
* 指定模型下载保存的目录
* 默认保存在用户主目录下的 .aha 文件夹中
* 示例：--save-dir /custom/model/path

5. 下载重试次数
----- 
    --download-retries <DOWNLOAD_RETRIES>
* 设置模型下载失败时的最大重试次数
* 默认值：3次
* 示例：--download-retries 5

##### 注意事项
* 参数前需要使用双横线 -- 分隔 cargo 命令和应用程序参数
* 模型参数 (--model 或 -m) 是必需的
* 如果未指定 --weight-path，程序会自动下载指定模型
* 下载的模型默认保存在 ~/.aha/ 目录下（除非指定了 --save-dir）

### API接口介绍
项目提供基于 OpenAI API 兼容的 RESTful 接口，支持多种模型推理任务。

#### 接口列表
1. 对话接口
- **端点**: `POST /chat/completions`
- **功能**: 多模态对话和文本生成
- **支持模型**: Qwen2.5VL, Qwen3, Qwen3VL, DeepSeekOCR, GLM-ASR-Nano-2512, Fun-ASR-Nano-2512 等
- **请求格式**: OpenAI Chat Completion 格式
- **响应格式**: OpenAI Chat Completion 格式
- **流式支持**: 支持

2. 图像处理接口
- **端点**: `POST /images/remove_background`
- **功能**: 图像背景移除
- **支持模型**: RMBG-2.0
- **请求格式**: OpenAI Chat Completion 格式
- **响应格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

3. 语音生成接口
- **端点**: `POST /audio/speech`
- **功能**: 语音合成和生成
- **支持模型**: VoxCPM,VoxCPM1.5
- **请求格式**: OpenAI Chat Completion 格式
- **响应格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

### 作为库使用
* cargo add aha
* 或者在Cargo.toml中添加
```toml
[dependencies]
aha = { git = "https://github.com/jhqxxx/aha.git" }

# 启用 CUDA 支持（可选）
aha = { git = "https://github.com/jhqxxx/aha.git", features = ["cuda"] }

# 启用Flash Attention 支持（可选）
aha = { git = "https://github.com/jhqxxx/aha.git", features = ["cuda", "flash-attn"] }
```
#### VoxCPM示例
```rust
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/openbmb/VoxCPM-0.5B/";
    
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    
    let generate = voxcpm_generate.generate(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
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

### 从源码构建运行测试
```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
# 修改测试用例中模型路径
# 运行 PaddleOCR-Vl 示例
cargo test -F cuda paddleocr_vl_generate -r -- --nocapture

# 运行 Hunyuan-OCR 示例
cargo test -F cuda hunyuan_ocr_generate -r -- --nocapture

# 运行 DeepSeek-OCR 示例
cargo test -F cuda deepseek_ocr_generate -r -- --nocapture

# 运行 Qwen3VL 示例
cargo test -F cuda qwen3vl_generate -r -- --nocapture

# 运行 MiniCPM4 示例  
cargo test -F cuda minicpm_generate -r -- --nocapture

# 运行 VoxCPM 示例
cargo test -F cuda voxcpm_generate -r -- --nocapture
```

## 开发
### 项目结构
```text
.
├── Cargo.toml
├── README.md
├── src
│   ├── chat_template
│   ├── models
│   │   ├── common
│   │   ├── deepseek_ocr
│   │   ├── hunyuan_ocr
│   │   ├── minicpm4
│   │   ├── paddleocr_vl
│   │   ├── qwen2_5vl
│   │   ├── qwen3vl
│   │   ├── voxcpm
│   │   └── mod.rs
│   ├── position_embed
│   ├── tokenizer
│   ├── utils
│   ├── api.rs
│   └── lib.rs
└── tests
    ├── test_hunyuan_ocr.rs
    ├── test_deepseek_ocr.rs
    ├── test_minicpm4.rs
    ├── test_paddleocr_vl.rs
    ├── test_qwen2_5vl.rs
    └── test_voxcpm.rs
```

### 添加新模型
* 在src/models/创建新模型文件
* 在src/models/mod.rs中导出
* 在tests/中添加测试和示例

## 许可证
本项目采用 Apache License, Version 2.0 许可证 - 查看 [LICENSE](./LICENSE) 文件了解详情。

## 致谢
* [Candle](https://github.com/huggingface/candle) - 优秀的 Rust 机器学习框架
* 所有模型的原作者和贡献者

## 支持
#### 如果你遇到问题：
1. 查看 Issues 是否已有解决方案
2. 提交新的 Issue，包含详细描述和复现步骤

## 更新日志
### [Unreleased] - 2025-02-04
* 支持Qwen3-ASR 模型
### v0.1.8
* 支持Fun-ASR-Nano-2512, Qwen3 模型

### v0.1.7
* 支持GLM-ASR-Nano-2512 模型

### v0.1.6
* 支持RMGB2.0 模型

### v0.1.5
* 支持VoxCPM1.5 模型

### v0.1.4
* 添加PaddleOCR-VL 模型

### v0.1.3
* 添加 Hunyuan-OCR 模型

### v0.1.2
* 添加 DeepSeek-OCR 模型

### v0.1.1
* 添加 Qwen3VL 模型

### v0.1.0
* 初始版本发布
* 支持 Qwen2.5VL, MiniCPM4, VoxCPM 模型


