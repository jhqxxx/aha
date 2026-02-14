# CLI 参考

aha 的完整命令行界面参考。

AHA 是一个基于 Candle 框架的高性能模型推理库，支持多种多模态模型，包括视觉、语言和语音模型。

```bash
aha [COMMAND] [OPTIONS]
```

## 全局选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径 | - |
| `--save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |
| `-h, --help` | 显示帮助信息 | - |
| `-V, --version` | 显示版本号 | - |

## 子命令

### cli - 下载模型并启动服务（默认）

下载指定的模型并启动 HTTP 服务。当不指定子命令时，默认使用此命令。

**语法：**
```bash
aha cli [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径（如指定则跳过下载） | - |
| `--save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |

**示例：**

```bash
# 下载模型并启动服务（默认端口 10100）
aha cli -m qwen3vl-2b

# 指定端口和保存目录
aha cli -m qwen3vl-2b -p 8080 --save-dir /data/models

# 使用本地模型（不下载）
aha cli -m qwen3vl-2b --weight-path /path/to/model

# 向后兼容方式（等同于 cli 子命令）
aha -m qwen3vl-2b
```

### run - 直接模型推理

直接运行模型推理，无需启动 HTTP 服务。适用于一次性推理任务或批处理。

**语法：**
```bash
aha run [OPTIONS] --model <MODEL> --input <INPUT> [--input <INPUT2>] --weight-path <WEIGHT_PATH>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `-i, --input <INPUT>` | 输入文本或文件路径（模型特定解释，支持1-2个参数, input1： 提示文本, input2: 文件地址） | - |
| `-o, --output <OUTPUT>` | 输出文件路径（可选，未指定则自动生成） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径（必选） | - |

**示例：**

```bash
# VoxCPM1.5 文字转语音（单个输入）
aha run -m voxcpm1.5 -i "太阳当空照" -o output.wav --weight-path /path/to/model

# VoxCPM1.5 从文件读取输入（单个输入）
aha run -m voxcpm1.5 -i "file://./input.txt" --weight-path /path/to/model

# MiniCPM4 文本生成（单个输入）
aha run -m minicpm4-0.5b -i "你好" --weight-path /path/to/model

# DeepSeek OCR 图片识别（单个输入）
aha run -m deepseek-ocr -i "image.jpg" --weight-path /path/to/model

# RMBG2.0 背景移除（单个输入）
aha run -m RMBG2.0 -i "photo.png" -o "no_bg.png" --weight-path /path/to/model

# GLM-ASR 语音识别（两个输入：提示文本 + 音频文件）
aha run -m glm-asr-nano-2512 -i "请转写这段音频" -i "audio.wav" --weight-path /path/to/model

# Fun-ASR 语音识别（两个输入：提示文本 + 音频文件）
aha run -m fun-asr-nano-2512 -i "语音转写：" -i "audio.wav" --weight-path /path/to/model

# qwen3 文本生成（单个输入）
aha run -m qwen3-0.6b -i "你好" --weight-path /path/to/model

# qwen2.5vl 图像理解（两个输入：提示文本 + 图片文件）
aha run -m qwen2.5vl-3b -i "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本" -i "image.jpg" --weight-path /path/to/model

# Qwen3-ASR 语音识别（单个输入：音频文件）
aha run -m qwen3asr-0.6b -i "audio.wav" --weight-path /path/to/model
```

### serv - 启动服务

使用指定模型启动 HTTP 服务。`--weight-path` 是可选的 - 如果不指定，默认使用 `~/.aha/{model_id}`。

**语法：**
```bash
aha serv [OPTIONS] --model <MODEL> [--weight-path <WEIGHT_PATH>]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径（可选） | ~/.aha/{model_id} |
| `--allow-remote-shutdown` | 允许远程关机请求（不推荐） | false |

**示例：**

```bash
# 使用默认模型路径启动服务 (~/.aha/{model_id})
aha serv -m qwen3vl-2b

# 使用本地模型启动服务
aha serv -m qwen3vl-2b --weight-path /path/to/model

# 指定端口启动
aha serv -m qwen3vl-2b -p 8080

# 指定监听地址
aha serv -m qwen3vl-2b -a 0.0.0.0

# 启用远程关机（不推荐用于生产环境）
aha serv -m qwen3vl-2b --allow-remote-shutdown
```

### ps - 列出运行中的服务

列出所有当前正在运行的 AHA 服务，显示进程 ID、端口和状态。

**语法：**
```bash
aha ps [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-c, --compact` | 紧凑输出格式（仅显示服务 ID） | false |

**示例：**

```bash
# 列出所有运行中的服务（表格格式）
aha ps

# 紧凑输出（仅服务 ID）
aha ps -c
```

**输出格式：**

```
Service ID           PID        Model                Port       Address         Status
-------------------------------------------------------------------------------------
56860@10100          56860      N/A                  10100      127.0.0.1       Running
```

**字段说明：**
- `Service ID`: 服务唯一标识符，格式为 `pid@port`
- `PID`: 进程 ID
- `Model`: 模型名称（如果未检测到则显示 N/A）
- `Port`: 服务端口号
- `Address`: 服务监听地址
- `Status`: 服务状态（Running、Stopping、Unknown）

### download - 下载模型

仅下载指定模型，不启动服务。

**语法：**
```bash
aha download [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `-s, --save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |

**示例：**

```bash
# 下载模型到默认目录
aha download -m qwen3vl-2b

# 指定保存目录
aha download -m qwen3vl-2b -s /data/models

# 指定下载重试次数
aha download -m qwen3vl-2b --download-retries 5

# 下载 MiniCPM4-0.5B 模型
aha download -m minicpm4-0.5b -s models
```

### delete - 删除已下载的模型

删除默认位置（`~/.aha/{model_id}`）的已下载模型。

**语法：**
```bash
aha delete [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |

**示例：**

```bash
# 删除 RMBG2.0 模型
aha delete -m rmbg2.0

# 删除 Qwen3-VL-2B 模型
aha delete --model qwen3vl-2b
```

**行为说明：**
- 删除前会显示模型信息（ID、位置、大小）
- 需要用户确认（y/N）才会执行删除
- 如果模型目录不存在，显示"模型未找到"消息
- 删除完成后显示"删除成功"消息

### list - 列出所有支持的模型

列出所有支持的模型及其 ModelScope ID。

**语法：**
```bash
aha list [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-j, --json` | 以 JSON 格式输出（包含 name、model_id 和 type 字段） | false |

**示例：**

```bash
# 以表格格式列出模型（默认）
aha list

# 以 JSON 格式列出模型
aha list --json

# 简写形式
aha list -j
```

**JSON 输出格式：**

使用 `--json` 时，输出包含：
- `name`：与 `-m` 参数一起使用的模型标识符
- `model_id`：完整的 ModelScope 模型 ID
- `type`：模型类别（`llm`、`ocr`、`asr` 或 `image`）

示例：
```json
[
  {
    "name": "qwen3vl-2b",
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "type": "llm"
  },
  {
    "name": "deepseek-ocr",
    "model_id": "deepseek-ai/DeepSeek-OCR",
    "type": "ocr"
  }
]
```

**模型类型：**
- `llm`：语言模型（文本生成、对话等）
- `ocr`：光学字符识别模型
- `asr`：自动语音识别模型
- `image`：图像处理模型

## 支持的模型

| 模型标识 | 模型名称 | 说明 |
|---------|---------|------|
| `minicpm4-0.5b` | OpenBMB/MiniCPM4-0.5B | 面壁智能 MiniCPM4 0.5B 模型 |
| `qwen2.5vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | 通义千问 2.5 VL 3B 模型 |
| `qwen2.5vl-7b` | Qwen/Qwen2.5-VL-7B-Instruct | 通义千问 2.5 VL 7B 模型 |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | 通义千问 3 0.6B 模型 |
| `qwen3vl-2b` | Qwen/Qwen3-VL-2B-Instruct | 通义千问 3 VL 2B 模型 |
| `qwen3vl-4b` | Qwen/Qwen3-VL-4B-Instruct | 通义千问 3 VL 4B 模型 |
| `qwen3vl-8b` | Qwen/Qwen3-VL-8B-Instruct | 通义千问 3 VL 8B 模型 |
| `qwen3vl-32b` | Qwen/Qwen3-VL-32B-Instruct | 通义千问 3 VL 32B 模型 |
| `deepseek-ocr` | deepseek-ai/DeepSeek-OCR | DeepSeek OCR 模型 |
| `hunyuan-ocr` | Tencent-Hunyuan/HunyuanOCR | 腾讯混元 OCR 模型 |
| `paddleocr-vl` | PaddlePaddle/PaddleOCR-VL | 百度飞桨 OCR VL 模型 |
| `RMBG2.0` | AI-ModelScope/RMBG-2.0 | RMBG 2.0 背景移除模型 |
| `voxcpm` | OpenBMB/VoxCPM-0.5B | 面壁智能 VoxCPM 0.5B 语音生成模型 |
| `voxcpm1.5` | OpenBMB/VoxCPM1.5 | 面壁智能 VoxCPM 1.5 语音生成模型 |
| `glm-asr-nano-2512` | ZhipuAI/GLM-ASR-Nano-2512 | 智谱 AI ASR Nano 2512 语音识别模型 |
| `fun-asr-nano-2512` | FunAudioLLM/Fun-ASR-Nano-2512 | 通义百聆 ASR Nano 2512 语音识别模型 |

## 常见使用场景

### 场景 1：快速启动推理服务

```bash
# 一条命令下载并启动服务
aha -m qwen3vl-2b
```

### 场景 2：使用已有模型启动服务

```bash
# 假设模型已下载到 /data/models/Qwen/Qwen3-VL-2B-Instruct
aha serv -m qwen3vl-2b --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### 场景 3：预先下载模型

```bash
# 下载模型到指定目录，稍后使用
aha download -m qwen3vl-2b -s /data/models

# 后续启动时直接使用
aha serv -m qwen3vl-2b --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### 场景 4：自定义服务端口和地址

```bash
# 在 0.0.0.0:8080 启动服务，允许外部访问
aha -m qwen3vl-2b -a 0.0.0.0 -p 8080
```

## API 接口

服务启动后，提供以下 API 接口：

### 对话接口
- **端点**: `POST /chat/completions`
- **功能**: 多模态对话和文本生成
- **支持模型**: Qwen2.5VL, Qwen3, Qwen3VL, DeepSeekOCR, GLM-ASR-Nano-2512, Fun-ASR-Nano-2512 等
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 支持

### 图像处理接口
- **端点**: `POST /images/remove_background`
- **功能**: 图像背景移除
- **支持模型**: RMBG-2.0
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

### 语音生成接口
- **端点**: `POST /audio/speech`
- **功能**: 语音合成和生成
- **支持模型**: VoxCPM, VoxCPM1.5
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

### 关机接口
- **端点**: `POST /shutdown`
- **功能**: 优雅地关闭服务器
- **安全性**: 默认仅允许本地访问，使用 `--allow-remote-shutdown` 标志启用远程访问（不推荐）
- **格式**: JSON 响应

## 向后兼容性

为了保持与旧版本的兼容性，以下两种使用方式是等效的：

```bash
# 新方式（推荐）
aha cli -m qwen3vl-2b

# 旧方式（向后兼容）
aha -m qwen3vl-2b
```

## 注意事项

1. **serv 子命令必须指定 `--weight-path`**：由于 `serv` 子命令不下载模型，必须通过 `--weight-path` 指定已下载的模型路径。

2. **下载重试机制**：默认重试 3 次，每次失败后等待 2 秒再重试。可通过 `--download-retries` 调整重试次数。

3. **默认保存目录**：模型默认保存到 `~/.aha/` 目录下，可通过 `--save-dir` 或 `-s` 参数自定义。

4. **端口占用**：启动服务前确保指定的端口未被占用，默认端口为 10100。

5. **权限问题**：如果保存到系统目录（如 `/data/models`），确保有相应的写入权限。

## 获取帮助

```bash
# 查看主帮助
aha --help

# 查看子命令帮助
aha cli --help
aha serv --help
aha download --help

# 查看版本信息
aha --version
```

## 另见

- [快速入门](./getting-started.zh-CN.md) - 快速入门指南
- [API 文档](./api.zh-CN.md) - REST API 参考
- [支持的模型](./supported-tools.zh-CN.md) - 可用模型
