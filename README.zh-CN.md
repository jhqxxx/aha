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
| **文本** | Qwen3, MiniCPM4, LFM2, LFM2.5 |
| **视觉** | Qwen2.5-VL, Qwen3-VL, Qwen3.5 <br> LFM2.5-VL, LFM2-VL |
| **OCR** | DeepSeek-OCR, DeepSeek-OCR-2, PaddleOCR-VL,  <br>PaddleOCR-VL1.5, Hunyuan-OCR, GLM-OCR |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano, Qwen3-ASR |
| **TTS** | VoxCPM, VoxCPM1.5 |
| **图像** | RMBG-2.0 (背景移除) |

## 为什么选择 aha？
- **🚀 高性能推理** - 基于 Candle 框架，提供高效的张量计算和模型推理
- **🔧 统一接口** — 一个工具搞定文本、视觉、语音和 OCR
- **📦 本地优先** — 所有处理在本地运行，数据不离境
- **🎯 跨平台** — 支持 Linux、macOS 和 Windows
- **⚡ GPU 加速** — 可选 CUDA 支持以获得更快推理
- **🛡️ 内存安全** — Rust 构建，稳定可靠
- **🧠 注意力优化** - 可选 Flash Attention 支持，优化长序列处理

## 更新日志
## Changelog
### 2026-04-03
- CLI 更新: 必须指定子命令
- ChatCompletionParameters 新增 repeat_penalty 和 repeat_last_n 参数
- 生成添加重复惩罚代码

### 2026-04-02
- 重构生成代码 
- \<think\>...\</think\> 思维链内容使用reasoning_content字段返回。
- 对话返回添加耗时信息

### 2026-04-01
- 重构 deepseek_ocr/fun_asr_nano 生成代码 

### 2026-03-31
- 新增 server 和 cli 模块
- aha模型名称使用 modelscope id 替换
- 更新 WhichModel 枚举
- Usage 增加时间信息
- 删除 aha_openai_dive, chrono 依赖

### v0.2.5 (2026-03-30)
- 新增 LFM2.5VL-1.6B
- 新增 LFM2VL-1.6B

### v0.2.4 (2026-03-23)
- 新增 LFM2.5-1.2B-Instruct
- 新增 LFM2-1.2B

**[查看完整更新日志](docs/changelog.zh-CN.md)** →


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
# VoxCPM示例
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

## Wechat
![260405 expired](./assets/img/aha_weixinqun.png)

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
