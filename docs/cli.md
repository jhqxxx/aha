# CLI Reference

Complete command-line interface reference for aha.

AHA is a high-performance model inference library based on the Candle framework, supporting various multimodal models including vision, language, and audio models.

```bash
aha [COMMAND] [OPTIONS]
```

## Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path | - |
| `--save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |
| `-h, --help` | Display help information | - |
| `-V, --version` | Display version number | - |

## Commands

### cli - Download model and start service (default)

Download the specified model and start an HTTP service. This command is used by default when no subcommand is specified.

**Syntax:**
```bash
aha cli [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (skip download if specified) | - |
| `--save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |

**Examples:**

```bash
# Download model and start service (default port 10100)
aha cli -m qwen3vl-2b

# Specify port and save directory
aha cli -m qwen3vl-2b -p 8080 --save-dir /data/models

# Use local model (skip download)
aha cli -m qwen3vl-2b --weight-path /path/to/model

# Backward compatible way (equivalent to cli subcommand)
aha -m qwen3vl-2b
```

### run - Direct model inference

Run model inference directly without starting an HTTP service. Suitable for one-time inference tasks or batch processing.

**Syntax:**
```bash
aha run [OPTIONS] --model <MODEL> --input <INPUT> [--input <INPUT2>] --weight-path <WEIGHT_PATH>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |
| `-i, --input <INPUT>` | Input text or file path (model-specific interpretation, supports 1-2 parameters: input1: prompt text, input2: file path) | - |
| `-o, --output <OUTPUT>` | Output file path (optional, auto-generated if not specified) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (required) | - |

**Examples:**

```bash
# VoxCPM1.5 text-to-speech (single input)
aha run -m voxcpm1.5 -i "太阳当空照" -o output.wav --weight-path /path/to/model

# VoxCPM1.5 read input from file (single input)
aha run -m voxcpm1.5 -i "file://./input.txt" --weight-path /path/to/model

# MiniCPM4 text generation (single input)
aha run -m minicpm4-0.5b -i "你好" --weight-path /path/to/model

# DeepSeek OCR image recognition (single input)
aha run -m deepseek-ocr -i "image.jpg" --weight-path /path/to/model

# RMBG2.0 background removal (single input)
aha run -m RMBG2.0 -i "photo.png" -o "no_bg.png" --weight-path /path/to/model

# GLM-ASR speech recognition (two inputs: prompt text + audio file)
aha run -m glm-asr-nano-2512 -i "请转写这段音频" -i "audio.wav" --weight-path /path/to/model

# Fun-ASR speech recognition (two inputs: prompt text + audio file)
aha run -m fun-asr-nano-2512 -i "语音转写：" -i "audio.wav" --weight-path /path/to/model

# qwen3 text generation (single input)
aha run -m qwen3-0.6b -i "你好" --weight-path /path/to/model

# qwen2.5vl image understanding (two inputs: prompt text + image file)
aha run -m qwen2.5vl-3b -i "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本" -i "image.jpg" --weight-path /path/to/model

# Qwen3-ASR speech recognition (single input: audio file)
aha run -m qwen3asr-0.6b -i "audio.wav" --weight-path /path/to/model
```

### serv - Start service

Start HTTP service with a model. The `--weight-path` is optional - if not specified, it defaults to `~/.aha/{model_id}`.

**Syntax:**
```bash
aha serv [OPTIONS] --model <MODEL> [--weight-path <WEIGHT_PATH>]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (optional) | ~/.aha/{model_id} |
| `--allow-remote-shutdown` | Allow remote shutdown requests (not recommended) | false |

**Examples:**

```bash
# Start service with default model path (~/.aha/{model_id})
aha serv -m qwen3vl-2b

# Start service with local model
aha serv -m qwen3vl-2b --weight-path /path/to/model

# Start with specified port
aha serv -m qwen3vl-2b -p 8080

# Specify listen address
aha serv -m qwen3vl-2b -a 0.0.0.0

# Enable remote shutdown (not recommended for production)
aha serv -m qwen3vl-2b --allow-remote-shutdown
```

### ps - List running services

List all currently running AHA services with their process IDs, ports, and status.

**Syntax:**
```bash
aha ps [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --compact` | Compact output format (show service IDs only) | false |

**Examples:**

```bash
# List all running services (table format)
aha ps

# Compact output (service IDs only)
aha ps -c
```

**Output Format:**

```
Service ID           PID        Model                Port       Address         Status
-------------------------------------------------------------------------------------
56860@10100          56860      N/A                  10100      127.0.0.1       Running
```

**Fields:**
- `Service ID`: Unique identifier in format `pid@port`
- `PID`: Process ID
- `Model`: Model name (N/A if not detected)
- `Port`: Service port number
- `Address`: Service listen address
- `Status`: Service status (Running, Stopping, Unknown)

### download - Download model

Download the specified model only, without starting the service.

**Syntax:**
```bash
aha download [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |
| `-s, --save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |

**Examples:**

```bash
# Download model to default directory
aha download -m qwen3vl-2b

# Specify save directory
aha download -m qwen3vl-2b -s /data/models

# Specify download retry count
aha download -m qwen3vl-2b --download-retries 5

# Download MiniCPM4-0.5B model
aha download -m minicpm4-0.5b -s models
```

### delete - Delete downloaded model

Delete a downloaded model from the default location (`~/.aha/{model_id}`).

**Syntax:**
```bash
aha delete [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |

**Examples:**

```bash
# Delete RMBG2.0 model from default location
aha delete -m rmbg2.0

# Delete Qwen3-VL-2B model
aha delete --model qwen3vl-2b
```

**Behavior:**
- Displays model information (ID, location, size) before deletion
- Requires confirmation (y/N) before proceeding
- Shows "Model not found" message if the model directory doesn't exist
- Shows "Model deleted successfully" message after completion

### list - List all supported models

List all supported models with their ModelScope IDs.

**Syntax:**
```bash
aha list [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-j, --json` | Output in JSON format (includes name, model_id, and type fields) | false |

**Examples:**

```bash
# List models in table format (default)
aha list

# List models in JSON format
aha list --json

# Short form
aha list -j
```

**JSON Output Format:**

When using `--json`, the output includes:
- `name`: Model identifier used with `-m` flag
- `model_id`: Full ModelScope model ID
- `type`: Model category (`llm`, `ocr`, `asr`, or `image`)

Example:
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

**Model Types:**
- `llm`: Language models (text generation, chat, etc.)
- `ocr`: Optical Character Recognition models
- `asr`: Automatic Speech Recognition models
- `image`: Image processing models

## Supported Models

| Model ID | Model Name | Description |
|----------|------------|-------------|
| `minicpm4-0.5b` | OpenBMB/MiniCPM4-0.5B | OpenBMB MiniCPM4 0.5B model |
| `qwen2.5vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | Qwen 2.5 VL 3B model |
| `qwen2.5vl-7b` | Qwen/Qwen2.5-VL-7B-Instruct | Qwen 2.5 VL 7B model |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | Qwen 3 0.6B model |
| `qwen3vl-2b` | Qwen/Qwen3-VL-2B-Instruct | Qwen 3 VL 2B model |
| `qwen3vl-4b` | Qwen/Qwen3-VL-4B-Instruct | Qwen 3 VL 4B model |
| `qwen3vl-8b` | Qwen/Qwen3-VL-8B-Instruct | Qwen 3 VL 8B model |
| `qwen3vl-32b` | Qwen/Qwen3-VL-32B-Instruct | Qwen 3 VL 32B model |
| `deepseek-ocr` | deepseek-ai/DeepSeek-OCR | DeepSeek OCR model |
| `hunyuan-ocr` | Tencent-Hunyuan/HunyuanOCR | Tencent Hunyuan OCR model |
| `paddleocr-vl` | PaddlePaddle/PaddleOCR-VL | Baidu PaddleOCR VL model |
| `RMBG2.0` | AI-ModelScope/RMBG-2.0 | RMBG 2.0 background removal model |
| `voxcpm` | OpenBMB/VoxCPM-0.5B | OpenBMB VoxCPM 0.5B speech synthesis model |
| `voxcpm1.5` | OpenBMB/VoxCPM1.5 | OpenBMB VoxCPM 1.5 speech synthesis model |
| `glm-asr-nano-2512` | ZhipuAI/GLM-ASR-Nano-2512 | Zhipu AI ASR Nano 2512 speech recognition model |
| `fun-asr-nano-2512` | FunAudioLLM/Fun-ASR-Nano-2512 | FunAudioLLM ASR Nano 2512 speech recognition model |

## Common Use Cases

### Scenario 1: Quick start inference service

```bash
# One command to download and start service
aha -m qwen3vl-2b
```

### Scenario 2: Start service with existing model

```bash
# Assuming model is downloaded to /data/models/Qwen/Qwen3-VL-2B-Instruct
aha serv -m qwen3vl-2b --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### Scenario 3: Pre-download model

```bash
# Download model to specified directory for later use
aha download -m qwen3vl-2b -s /data/models

# Later start with local model
aha serv -m qwen3vl-2b --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### Scenario 4: Custom service port and address

```bash
# Start service on 0.0.0.0:8080, allow external access
aha -m qwen3vl-2b -a 0.0.0.0 -p 8080
```

## API Endpoints

After the service starts, the following API endpoints are available:

### Chat Completion Endpoint
- **Endpoint**: `POST /chat/completions`
- **Function**: Multimodal chat and text generation
- **Supported Models**: Qwen2.5VL, Qwen3, Qwen3VL, DeepSeekOCR, GLM-ASR-Nano-2512, Fun-ASR-Nano-2512, etc.
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: Yes

### Image Processing Endpoint
- **Endpoint**: `POST /images/remove_background`
- **Function**: Image background removal
- **Supported Models**: RMBG-2.0
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: No

### Audio Generation Endpoint
- **Endpoint**: `POST /audio/speech`
- **Function**: Speech synthesis and generation
- **Supported Models**: VoxCPM, VoxCPM1.5
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: No

### Shutdown Endpoint
- **Endpoint**: `POST /shutdown`
- **Function**: Gracefully shut down the server
- **Security**: Localhost only by default, use `--allow-remote-shutdown` flag to enable remote access (not recommended)
- **Format**: JSON response


## Backward Compatibility

To maintain compatibility with older versions, the following two usage methods are equivalent:

```bash
# New way (recommended)
aha cli -m qwen3vl-2b

# Old way (backward compatible)
aha -m qwen3vl-2b
```

## Notes

1. **serv subcommand requires `--weight-path`**: Since the `serv` subcommand does not download models, you must specify the path to an already downloaded model via `--weight-path`.

2. **Download retry mechanism**: By default, retries 3 times, waiting 2 seconds after each failure before retrying. You can adjust the retry count with `--download-retries`.

3. **Default save directory**: Models are saved to `~/.aha/` directory by default, which can be customized via `--save-dir` or `-s` parameter.

4. **Port occupation**: Ensure the specified port is not occupied before starting the service. The default port is 10100.

5. **Permission issues**: If saving to a system directory (such as `/data/models`), ensure you have the corresponding write permissions.

## Getting Help

```bash
# View main help
aha --help

# View subcommand help
aha cli --help
aha serv --help
aha download --help

# View version information
aha --version
```

## See Also

- [Getting Started](./getting-started.md) - Quick start guide
- [API Documentation](./api.md) - REST API reference
- [Supported Models](./supported-tools.md) - Available models
