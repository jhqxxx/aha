# 多模型并行运行 - 快速参考

## 🚀 快速开始

### 启动多模型服务

```bash
# 基本用法：同时加载多个模型
aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100

# 包含 OCR 的全能组合
aha serv \
  -m Qwen/Qwen3-0.6B \
  -m GLM-ASR-Nano \
  -m PaddleOCR-VL \
  -p 10100

# 指定本地路径
aha serv \
  -m Qwen/Qwen3-0.6B \
  -m DeepSeek-OCR \
  --weight-path /models/qwen3 \
  -p 10100
```

### 查看已加载模型

```bash
curl http://localhost:10100/admin/models/list | jq
```

输出：
```json
{
  "models": ["Qwen3", "GlmAsrNano", "PaddleOCRVL"],
  "count": 3
}
```

## 📡 API 调用示例

### 文本生成（LLM）

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "messages": [
      {"role": "user", "content": "解释量子计算"}
    ],
    "stream": false,
    "max_tokens": 200
  }'
```

### 语音识别（ASR）

```bash
curl http://localhost:10100/v1/audio/transcriptions \
  -F "model=GlmAsrNano" \
  -F "file=@recording.wav" \
  -F "language=zh"
```

### OCR 文字识别

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PaddleOCRVL",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
          {"type": "text", "text": "识别图片中的文字"}
        ]
      }
    ]
  }'
```

### 流式输出

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

## 💡 推荐模型组合

### 办公自动化助手
```bash
aha serv -m Qwen/Qwen3-0.6B -m PaddleOCR-VL -p 10100
```
- 内存占用：~4GB
- 功能：文档处理 + OCR 识别

### 多媒体处理工作站
```bash
aha serv \
  -m GLM-ASR-Nano \
  -m VoxCPM \
  -m RMBG-2.0 \
  -p 10100
```
- 内存占用：~3GB
- 功能：语音识别 + 语音合成 + 背景移除

### 全能 AI 助手
```bash
aha serv \
  -m Qwen/Qwen3-0.6B \
  -m GLM-ASR-Nano \
  -m DeepSeek-OCR \
  -m Qwen3-Embedding \
  -p 10100
```
- 内存占用：~6GB
- 功能：对话 + 语音 + OCR + 向量搜索

## ⚙️ 性能调优

### GPU 加速（NVIDIA）
```bash
cargo build --release --features cuda
./target/release/aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100
```
- 自动使用 F16 数据类型
- 显存占用减半
- 推理速度提升 2-3x

### GPU 加速（Apple Silicon）
```bash
cargo build --release --features metal
./target/release/aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100
```

### Flash Attention（长文本优化）
```bash
cargo build --release --features cuda,flash-attn
```
- 适合处理 >4K tokens 的长文档
- 内存占用从 O(n²) 降到 O(n)

## 🔍 故障排查

### 问题：模型未找到
```bash
# 检查可用模型列表
curl http://localhost:10100/admin/models/list

# 确保 model 参数正确
curl ... -d '{"model": "Qwen3", ...}'  # ✅ 正确
curl ... -d '{"model": "qwen3", ...}'  # ❌ 错误（大小写敏感）
```

### 问题：内存不足
```bash
# 方案1：减少模型数量
aha serv -m Qwen/Qwen3-0.6B -p 10100  # 只加载一个模型

# 方案2：使用量化版本
aha serv -m Qwen/Qwen3-0.6B-GGUF \
  --gguf-path /path/to/qwen3-q4_k_m.gguf \
  -p 10100

# 方案3：增加 swap 空间（Linux）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 问题：GPU 显存不足
```bash
# 检查显存使用
nvidia-smi  # NVIDIA
powermetrics --samplers gpu_power  # macOS

# 解决方案：
# 1. 卸载不需要的模型（未来版本支持）
# 2. 使用 F16 模式（默认启用）
# 3. 减少并发请求数
```

### 问题：响应缓慢
```bash
# 检查系统负载
htop  # Linux/macOS
taskmgr  # Windows

# 优化建议：
# 1. 启用 GPU 加速
# 2. 减少同时运行的模型数量
# 3. 使用更小的模型变体
# 4. 增加 batch size（如果支持）
```

## 📊 监控命令

### 实时性能监控
```bash
# Linux
watch -n 1 'nvidia-smi && echo "---" && free -h'

# macOS
powermetrics --samplers cpu_power,gpu_power,memory_pressure -i 1000

# Windows (PowerShell)
Get-Counter '\Processor(_Total)\% Processor Time' -Continuous
```

### API 性能测试
```bash
# 单请求延迟
time curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3", "messages": [{"role": "user", "content": "Hi"}]}'

# 并发测试（需要安装 ab）
ab -n 100 -c 10 http://localhost:10100/v1/chat/completions

# 压力测试（需要安装 wrk）
wrk -t4 -c20 -d30s http://localhost:10100/v1/chat/completions \
  -s post.lua -- "model=Qwen3&message=Hello"
```

## 🛠️ 开发调试

### 作为库使用
```rust
use aha::server::api::{init_shared_resources, load_and_register_model};
use aha::models::common::model_mapping::WhichModel;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化共享资源
    init_shared_resources();
    
    // 加载模型
    load_and_register_model(
        WhichModel::Qwen3_0_6B,
        "/path/to/qwen3".to_string(),
        None,
        None,
    )?;
    
    load_and_register_model(
        WhichModel::GlmAsrNano,
        "/path/to/glm-asr".to_string(),
        None,
        None,
    )?;
    
    // 启动你的服务...
    Ok(())
}
```

### 日志级别
```bash
# 设置日志级别
RUST_LOG=debug ./target/release/aha serv -m Qwen/Qwen3-0.6B -p 10100

# 只显示错误
RUST_LOG=error ./target/release/aha serv -m Qwen/Qwen3-0.6B -p 10100
```

## 📚 相关文档

- [完整使用指南](multi-model-guide.md)
- [架构设计文档](multi-model-architecture.md)
- [API 文档](../docs/api.md)
- [支持的模型列表](../docs/supported-models.md)

## 💬 获取帮助

- GitHub Issues: https://github.com/jhqxxx/aha/issues
- 微信群：扫描 README 中的二维码
- 文档：https://github.com/jhqxxx/aha/docs

---

**提示**: 更多详细信息请查看完整文档。
