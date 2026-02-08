# API Reference

Complete reference for the AHA REST API.

## Overview

AHA provides an OpenAI-compatible REST API for running AI model inference. All endpoints follow standard HTTP conventions and return JSON responses.

### Base URL

By default, the API server runs on:
```
http://127.0.0.1:10100
```

You can customize this when starting the service:
```bash
aha cli -m qwen3-0.6b -a 0.0.0.0 -p 8080
```

### Authentication

Currently, AHA does not require authentication. All endpoints are publicly accessible on the configured address/port.

**Security Note**: If you expose the API to external networks, consider implementing authentication through a reverse proxy (e.g., nginx, traefik).

### Content Types

All requests should use:
```
Content-Type: application/json
```

### Response Format

Success responses follow this structure:
```json
{
  "data": { ... },
  "model": "model-name",
  "usage": {
    "total_tokens": 30
  }
}
```

Error responses:
```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "error_code"
  }
}
```

## Endpoints

### Health Check

Check the service health status. This endpoint is useful for container orchestration (Kubernetes), load balancers, and monitoring systems.

#### Endpoint
```
GET /health
```

#### Response

**Healthy (HTTP 200):**

```json
{
  "status": "ok"
}
```

**Unhealthy (HTTP 503):**

```json
{
  "status": "unhealthy",
  "error": "model not initialized"
}
```

#### Example

```bash
curl http://127.0.0.1:10100/health
```

### Models

Get information about the currently loaded model (OpenAI API compatible format).

#### Endpoint
```
GET /models
```

#### Response

**Success (HTTP 200):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-0.6b",
      "object": "model",
      "created": null,
      "owned_by": "Qwen"
    }
  ]
}
```

**Not Initialized (HTTP 503):**

```json
{
  "error": "model not initialized"
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | Fixed value: "list" |
| `data` | array | Array of model objects (currently contains one loaded model) |
| `id` | string | Model identifier in kebab-case (e.g., "qwen3-0.6b") |
| `object` | string | Fixed value: "model" |
| `created` | integer\|null | Unix timestamp (currently null) |
| `owned_by` | string | Model owner/organization name |

#### Example

```bash
curl http://127.0.0.1:10100/models
```

### Chat Completions

Generate chat completions or text responses.

#### Endpoint
```
POST /chat/completions
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "qwen3-0.6b") |
| `messages` | array | Yes | Array of message objects |
| `temperature` | number | No | Sampling temperature (0-2, default: 1) |
| `top_p` | number | No | Nucleus sampling (0-1, default: 1) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `stream` | boolean | No | Enable streaming (default: false) |

#### Message Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | "system", "user", or "assistant" |
| `content` | string/array | Yes | Message content (string or multimodal array) |

#### Multimodal Content

For vision/audio models, content can be an array:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
  ]
}
```

Supported content types:
- `text` - Text content
- `image_url` - Image file (file://, base64://, https:// or http://)
- `audio_url` - Audio file (file://, base64://, https:// or http://)

#### Examples

**Simple Chat:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**With System Message:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain Rust in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Vision Understanding:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
      }
    ]
  }'
```

**OCR (Text Extraction):**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract all text"},
          {"type": "image", "image_url": {"url": "file:///path/to/document.png"}}
        ]
      }
    ]
  }'
```

**ASR (Speech Recognition):**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-asr-nano-2512",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Transcribe this audio"},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ]
  }'
```

**Streaming Response:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

Streaming responses are sent as Server-Sent Events (SSE):
```
data: {"id": "1", "choices": [{"delta": {"content": "Once"}}]}

data: {"id": "1", "choices": [{"delta": {"content": " upon"}}]}

data: [DONE]
```

#### Response

**Non-streaming:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "total_tokens": 19
  }
}
```

#### Supported Models

- Text: `qwen3-0.6b`, `minicpm4-0.5b`
- Vision: `qwen2.5vl-3b`, `qwen2.5vl-7b`, `qwen3vl-2b`, `qwen3vl-4b`, `qwen3vl-8b`, `qwen3vl-32b`
- OCR: `deepseek-ocr`, `hunyuan-ocr`, `paddleocr-vl`
- ASR: `glm-asr-nano-2512`, `fun-asr-nano-2512`, `qwen3asr-0.6b`, `qwen3asr-1.7b`

### Audio Speech

Generate speech from text (Text-to-Speech).

#### Endpoint
```
POST /audio/speech
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "voxcpm1.5") |
| `messages` | array | Yes | Array of message objects |

#### Example

```bash
curl http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Hello, this is AHA speaking."},
          {"type": "audio", "audio_url": {"url": "https://package-release.coderbox.cn/aiway/test/other/%E5%93%AA%E5%90%92.wav"}}
        ]
      }
    ]
  }'
```

#### Response

Returns audio data in base64 WAV format.

#### Supported Models

- `voxcpm`, `voxcpm1.5`

### Images Remove Background

Remove background from images.

#### Endpoint
```
POST /images/remove_background
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "rmbg2.0") |
| `messages` | array | Yes | Array of message objects |

#### Example

**From File:**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmbg2.0",
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

**From Base64:**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmbg2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image_url": {"url": "base64://$(base64 -w 0 photo.png)"}}
        ]
      }
    ]
  }'
```

#### Response

Returns the processed image in base64 PNG format.

#### Supported Models

- `rmbg2.0`

### Graceful Shutdown

Gracefully shut down the AHA server. This endpoint initiates a graceful shutdown process that:
1. Stops accepting new connections
2. Waits for existing requests to complete (up to 1 second)
3. Cleans up PID files
4. Exits the process

#### Endpoint
```
POST /shutdown
```

#### Request Body

None (empty request)

#### Response

**Success (HTTP 200):**

```json
{
  "message": "Shutting down..."
}
```

**Forbidden (HTTP 403):**

When remote shutdown is not allowed:

```json
{
  "error": "Remote shutdown not allowed. Use --allow-remote-shutdown flag to enable (not recommended)."
}
```

#### Security

By default, the shutdown endpoint only allows requests from localhost (127.0.0.1). To enable remote shutdown, start the server with the `--allow-remote-shutdown` flag:

```bash
aha serv -m qwen3-0.6b --allow-remote-shutdown
```

**Warning:** Enabling remote shutdown is not recommended for production use unless properly secured.

#### Example

```bash
curl -X POST http://127.0.0.1:10100/shutdown
```

#### Logging

All shutdown requests are logged to stderr with the format:

```
[SHUTDOWN] Shutdown requested (remote_allowed: false)
```

## Error Handling

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Model or endpoint not found |
| 500 | Internal Server Error - Model inference error |
| 503 | Service Unavailable - Model not loaded |

### Error Response Format

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## Rate Limiting

Currently, AHA does not implement rate limiting. The server can handle concurrent requests limited only by system resources.

## File Upload Limits

- String data: 5 MB
- File uploads: 100 MB

## OpenAI Compatibility

AHA's text generation API is designed to be compatible with OpenAI's API format. Multimodal APIs are derived from the text generation API with minimal changes:

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10100",
    api_key="dummy"  # Not used but required by library
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript Example

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:10100',
  apiKey: 'dummy'  // Not used but required
});

const response = await client.chat.completions.create({
  model: 'qwen3-0.6b',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

## Best Practices

### 1. Use Streaming for Long Responses

For long text generation, use streaming to get responses incrementally:

```bash
curl ... -d '{"stream": true, ...}'
```

### 2. Set Appropriate Token Limits

Prevent excessively long responses:

```json
{
  "max_tokens": 500
}
```

### 3. Adjust Temperature

Control response creativity:
- `0.0-0.3`: Deterministic, focused
- `0.4-0.7`: Balanced (default: 1.0)
- `0.8-2.0`: Creative, varied

### 4. Use System Messages

Set behavior with system messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a technical writer."},
    {"role": "user", "content": "..."}
  ]
}
```

## See Also

- [Getting Started](./getting-started.md) - Quick start guide
- [CLI Reference](./cli.md) - Command-line usage
- [Installation](./installation.md) - Installation guide
- [Development](./development.md) - Contributing guide
