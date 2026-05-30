# 快速入门

欢迎使用 AHA！本指南将帮助您快速上手。

## 快速开始（5 分钟）

### 1. 查看可用模型

```bash
aha list
```

### 2. 下载第一个模型

```bash
# 从下载一个小型文本模型开始
aha download -m Qwen/Qwen3-0.6B
```

### 3. 启动服务

```bash
# 启动 HTTP API 服务器
aha cli -m Qwen/Qwen3-0.6B
```

服务将在 `http://127.0.0.1:10100` 上启动

### 4. 发起第一个 API 调用

在新终端中：

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "你好，AHA！"}
    ],
    "stream": false
  }'
```

## 基本概念

### 什么是 AHA？

AHA 是一个本地 AI 推理引擎，具有以下特点：
- 在您的机器上运行模型（无需云 API）
- 支持多种模型类型（文本、视觉、音频、OCR、ASR）
- 提供 OpenAI 兼容的 API
- 模型下载后可离线工作

### 模型类别

| 类别 | 描述 | 示例模型 |
|------|------|----------|
| **文本** | 文本生成和对话 | Qwen3、MiniCPM4 |
| **视觉** | 图像理解 | Qwen2.5VL、Qwen3VL |
| **OCR** | 从图像中提取文本 | DeepSeek-OCR、Hunyuan-OCR |
| **ASR** | 语音转文本 | GLM-ASR、Fun-ASR、Qwen3-ASR |
| **音频** | 文本转语音 | VoxCPM、VoxCPM1.5 |
| **图像** | 图像处理 | RMBG2.0（背景移除） |

### CLI 命令

| 命令 | 用途 |
|------|------|
| `aha cli` | 下载模型并启动服务 |
| `aha serv` | 使用现有模型启动服务 |
| `aha download` | 仅下载模型 |
| `aha run` | 直接推理，无需服务器 |
| `aha list` | 列出可用模型 |

## 常见工作流程

### 文本生成

```bash
# 启动服务
aha cli -m Qwen/Qwen3-0.6B

# 在另一个终端中，发起请求
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手。"},
      {"role": "user", "content": "用简单的术语解释量子计算。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 视觉理解

```bash
# 启动视觉模型
aha cli -m Qwen/Qwen3-VL-2B-Instruct

# 分析图像
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "详细描述这张图片。"},
          {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
      }
    ],
    "stream": false
  }'
```

### OCR（文本提取）

```bash
# 启动 OCR 模型
aha cli -m deepseek-ai/DeepSeek-OCR

# 从图像中提取文本
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "提取这张图片中的所有文本。"},
          {"type": "image", "image_url": {"url": "file:///path/to/document.jpg"}}
        ]
      }
    ]
  }'
```

### 语音识别（ASR）

```bash
# 启动 ASR 模型
aha cli -m ZhipuAI/GLM-ASR-Nano-2512

# 转写音频
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ZhipuAI/GLM-ASR-Nano-2512",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "转写这段音频。"},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ]
  }'
```

### 文本转语音

```bash
# 启动 TTS 模型
aha cli -m OpenBMB/VoxCPM1.5

# 生成语音
curl http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenBMB/VoxCPM1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你好，这是 AHA 在说话。"}
        ]
      }
    ]
  }'
```

### 背景移除

```bash
# 启动 RMBG2.0 模型
aha cli -m AI-ModelScope/RMBG-2.0

# 移除图像背景
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AI-ModelScope/RMBG-2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image_url": {"url": "file:///path/to/document.jpg"}}
        ]
      }
    ]
  }'
```

### 直接推理（无需服务器）

```bash
# 直接运行推理，无需启动 HTTP 服务器
aha run -m Qwen/Qwen3-0.6B \
  -i "写一首关于AI的俳句" \
  --weight-path ~/.aha/Qwen/Qwen3-0.6B
```

## 配置选项

### 更改端口

```bash
# 使用端口 8080 而不是默认的 10100
aha cli -m Qwen/Qwen3-0.6B -p 8080
```

### 绑定到所有接口

```bash
# 允许外部访问（请谨慎使用）
aha cli -m Qwen/Qwen3-0.6B -a 0.0.0.0 -p 8080
```

### 使用本地模型

```bash
# 跳过下载，使用现有模型
aha serv -m Qwen/Qwen3-0.6B \
  --weight-path /path/to/model \
  -p 8080
```

### 自定义保存目录

```bash
# 将模型下载到特定目录
aha download -m Qwen/Qwen3-VL-2B-Instruct -s /data/models
```

## 流式响应

对于chat/completions，无"stream"字段或"stream": true时使用流式传输，"stream": false为非流式：

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "给我讲个故事"}
    ],
    "stream": false
  }'
```

## 模型选择指南

### 文本生成
- **Qwen/Qwen3-0.6B**：快速、轻量级（~1.2 GB）
- **OpenBMB/MiniCPM4-0.5B**：小型、高效（~1 GB）

### 视觉任务
- **Qwen/Qwen3-VL-2B-Instruct**：平衡性能（~4 GB）
- **Qwen/Qwen3-VL-8B-Instruct**：更高质量（~16 GB）

### OCR
- **deepseek-ai/DeepSeek-OCR**：通用
- **Tencent-Hunyuan/HunyuanOCR**：适合中文文本
- **PaddlePaddle/PaddleOCR-VL**：轻量级选项

### 语音识别
- **ZhipuAI/GLM-ASR-Nano-2512**：快速、准确
- **FunAudioLLM/Fun-ASR-Nano-2512**：适合中文
- **Qwen/Qwen3-ASR-0.6B**：轻量级

### 文本转语音
- **OpenBMB/VoxCPM1.5**：高质量中文

### 背景移除
- **AI-ModelScope/RMBG-2.0**：最先进的结果

## 提示与最佳实践

### 1. 从小开始

从小型模型开始了解工作流程：
```bash
aha download -m Qwen/Qwen3-0.6B
```

### 2. 使用 GPU 加速

使用 GPU 支持构建以获得更好的性能：
```bash
# NVIDIA GPU
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### 3. 预先下载模型

在网络良好时下载模型：
```bash
aha download -m Qwen/Qwen3-VL-2B-Instruct
```

稍后在没有网络的情况下使用：
```bash
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path ~/.aha/Qwen/Qwen3-VL-2B-Instruct
```

### 4. 管理磁盘空间

模型默认存储在 `~/.aha/` 中。如需要，清理：
```bash
# 检查磁盘使用情况
du -sh ~/.aha/*

# 删除旧模型
rm -rf ~/.aha/old-model-name
```

### 5. 监控资源

对于大型模型，监控您的资源：
```bash
# Linux
htop
nvidia-smi  # 对于 NVIDIA GPU

# macOS
活动监视器
```

## 故障排除

### 端口已被占用

```bash
# 使用不同的端口
aha cli -m Qwen/Qwen3-0.6B -p 8080
```

### 模型下载失败

```bash
# 重试更多次数
aha download -m Qwen/Qwen3-VL-2B-Instruct --download-retries 5
```

### 内存不足

```bash
# 使用更小的模型
aha cli -m Qwen/Qwen3-0.6B
```

## 后续步骤

1. 探索 [API 参考](./api.zh-CN.md) 了解详细的端点文档
2. 阅读 [CLI 参考](./cli.zh-CN.md) 了解所有命令选项
3. 查看 [架构与设计](./concepts.zh-CN.md) 了解 AHA 的工作原理
4. 如果您想贡献，请参阅 [开发指南](./development.zh-CN.md)

## 示例仓库

更多示例，请查看仓库中的 [tests](../tests/) 目录。

## 另见

- [API 参考](./api.zh-CN.md) - 完整的 API 文档
- [CLI 参考](./cli.zh-CN.md) - 命令行参考
- [安装指南](./installation.zh-CN.md) - 安装说明
- [开发指南](./development.zh-CN.md) - 贡献指南
