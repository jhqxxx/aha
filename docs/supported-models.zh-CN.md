# 支持的模型

aha 支持多个领域的最先进 AI 模型集合。

```shell
Available models:

Model ID                                 Owner                type       Download  
--------------------------------------------------------------------------------
LiquidAI/LFM2-1.2B                       LiquidAI             llm          ✔       
LiquidAI/LFM2.5-1.2B-Instruct            LiquidAI             llm          ✔       
LiquidAI/LFM2.5-VL-1.6B                  LiquidAI             vlm          ✔       
LiquidAI/LFM2-VL-1.6B                    LiquidAI             vlm          ✔       
OpenBMB/MiniCPM4-0.5B                    OpenBMB              llm          ✔       
Qwen/Qwen2.5-VL-3B-Instruct              Qwen                 vlm          ✔       
Qwen/Qwen2.5-VL-7B-Instruct              Qwen                 vlm                  
Qwen/Qwen3-0.6B                          Qwen                 llm          ✔
Qwen/Qwen3-1.7B                          Qwen                 llm                 
Qwen/Qwen3-4B                            Qwen                 llm          
Qwen/Qwen3.5-0.8B                        Qwen                 vlm          ✔       
Qwen/Qwen3.5-2B                          Qwen                 vlm                  
Qwen/Qwen3.5-4B                          Qwen                 vlm                  
Qwen/Qwen3.5-9B                          Qwen                 vlm                  
qwen3.5-gguf                             none                 vlm                  
Qwen/Qwen3-ASR-0.6B                      Qwen                 asr          ✔       
Qwen/Qwen3-ASR-1.7B                      Qwen                 asr                  
Qwen/Qwen3-VL-2B-Instruct                Qwen                 vlm          ✔       
Qwen/Qwen3-VL-4B-Instruct                Qwen                 vlm                  
Qwen/Qwen3-VL-8B-Instruct                Qwen                 vlm                  
Qwen/Qwen3-VL-32B-Instruct               Qwen                 vlm                  
deepseek-ai/DeepSeek-OCR                 deepseek-ai          ocr          ✔       
deepseek-ai/DeepSeek-OCR-2               deepseek-ai          ocr                  
Tencent-Hunyuan/HunyuanOCR               Tencent-Hunyuan      ocr          ✔       
PaddlePaddle/PaddleOCR-VL                PaddlePaddle         ocr          ✔       
PaddlePaddle/PaddleOCR-VL-1.5            PaddlePaddle         ocr                  
AI-ModelScope/RMBG-2.0                   AI-ModelScope        image        ✔       
OpenBMB/VoxCPM-0.5B                      OpenBMB              tts          ✔       
OpenBMB/VoxCPM1.5                        OpenBMB              tts          ✔       
ZhipuAI/GLM-ASR-Nano-2512                ZhipuAI              asr          ✔       
FunAudioLLM/Fun-ASR-Nano-2512            FunAudioLLM          asr          ✔       
ZhipuAI/GLM-OCR                          ZhipuAI              ocr          ✔
```

## 文本生成

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen3-0.6B** | 0.6B | Qwen/Qwen3-0.6B <br> Qwen/Qwen3-1.7B <br> Qwen/Qwen3-4B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM4-0.5B** | 0.5B | OpenBMB/MiniCPM4-0.5B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2-1.2B** | 1.2B | LiquidAI/LFM2-1.2B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/LICENSE) |
| **LFM2.5-1.2B-Instruct** | 1.2B | LiquidAI/LFM2.5-1.2B-Instruct | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |

## Embedding

| 模型 | 参数量 | 描述 | 开源协议 |
|------|--------|------|---------|
| **Qwen3-Embedding-0.6B** | 0.6B | 文本向量（safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-Embedding-4B** | 4B | 文本向量（safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-Embedding-8B** | 8B | 文本向量（safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **all-MiniLM-L6-v2** | 22M | sentence-transformers 文本向量（safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/LICENSE) |

## Reranker

| 模型 | 参数量 | 描述 | 开源协议 |
|------|--------|------|---------|
| **Qwen3-Reranker-0.6B** | 0.6B | 文本重排（embedding-similarity 基线，safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-Reranker-4B** | 4B | 文本重排（embedding-similarity 基线，safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-Reranker-8B** | 8B | 文本重排（embedding-similarity 基线，safetensors / gguf / onnx） | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 视觉与多模态

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen2.5-VL** | 3B <br> 7B | Qwen/Qwen2.5-VL-3B-Instruct <br> Qwen/Qwen2.5-VL-7B-Instruct | [Qwen 研究许可协议](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) <br> [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL** | 2B <br> 4B <br> 8B <br> 32B | Qwen/Qwen3-VL-2B-Instruct <br> Qwen/Qwen3-VL-4B-Instruct <br> Qwen/Qwen3-VL-8B-Instruct <br> Qwen/Qwen3-VL-32B-Instruct | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5** | 0.8B <br> 2B <br> 4B <br> 9B | Qwen/Qwen3.5-0.8B <br> Qwen/Qwen3.5-2B <br> Qwen/Qwen3.5-4B <br> Qwen/Qwen3.5-9B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2.5-VL-1.6B** | 1.6B | LiquidAI/LFM2.5-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |
| **LFM2-VL-1.6B** | 1.6B | LiquidAI/LFM2-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |

## OCR

| 模型 | 语言 | 模型id | 开源协议 |
|------|------|------|---------|
| **PaddleOCR-VL** | 多语言 | PaddlePaddle/PaddleOCR-VL <br> PaddlePaddle/PaddleOCR-VL-1.5 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Hunyuan-OCR** | 中文 | Tencent-Hunyuan/HunyuanOCR | [腾讯混元社区许可协议](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |
| **DeepSeek-OCR** | 多语言 | deepseek-ai/DeepSeek-OCR <br> deepseek-ai/DeepSeek-OCR-2 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **GLM-OCR** | 8 | ZhipuAI/GLM-OCR | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |

GLM-OCR 本地制品格式：`safetensors`、`gguf`、`onnx`

## 语音识别 (ASR)

| 模型 | 参数量 | 语言 | 模型id | 开源协议 |
|------|--------|------|-----|---------|
| **Fun-ASR-Nano-2512** | - | 中/英 | FunAudioLLM/Fun-ASR-Nano-2512 | 未标明 |
| **GLM-ASR-Nano-2512** | - | 中/英 | ZhipuAI/GLM-ASR-Nano-2512 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR** | 0.6B <br> 1.7B | 中/英 | Qwen/Qwen3-ASR-0.6B <br> Qwen/Qwen3-ASR-1.7B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |


## 语音生成

| 模型 | 版本 | 模型id | 开源协议 |
|------|--------|------|------|
| **VoxCPM** | 1<br>1.5 | OpenBMB/VoxCPM-0.5B <br> OpenBMB/VoxCPM1.5 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 图像处理

| 模型 | 类型 | 模型id | 开源协议 |
|------|------|-----|---------|
| **RMBG-2.0** | 背景移除 | AI-ModelScope/RMBG-2.0 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans) |

## 模型来源

模型来源：

- [Hugging Face](https://huggingface.co) - 主模型中心
- [ModelScope](https://modelscope.cn) - 中文模型中心

## 已收录仓库（当前运行时暂未直接接入）

以下仓库已纳入项目模型目录，但当前 `aha` 运行时尚不能直接推理：

### MLX / 特定格式变体
- Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit
- Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit
- Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit
- Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-8bit
- Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit
- Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit
- Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-8bit

### Embedding 模型
- google/embeddinggemma-300m
- ggml-org/embeddinggemma-300M-GGUF
- onnx-community/embeddinggemma-300m-ONNX
- unsloth/embeddinggemma-300m-GGUF
- onnx-community/Qwen3-Embedding-0.6B-ONNX
- Qwen/Qwen3-Embedding-0.6B-GGUF
- onnx-community/Qwen3-Embedding-4B-ONNX
- Qwen/Qwen3-Embedding-4B-GGUF
- Qwen/Qwen3-Embedding-8B-GGUF
- onnx-community/Qwen3-Embedding-8B-ONNX
- perplexity-ai/pplx-embed-v1-0.6b
- nomic-ai/nomic-embed-text-v2-moe
- nomic-ai/nomic-embed-text-v2-moe-GGUF
- jinaai/jina-embeddings-v5-text-small
- jinaai/jina-embeddings-v5-text-nano
- jinaai/jina-embeddings-v5-text-small-text-matching
- jinaai/jina-embeddings-v5-text-small-text-matching-GGUF
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### Reranker 模型
- BAAI/bge-reranker-v2-m3
- ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF

### ONNX 仓库
- onnx-community/GLM-OCR-ONNX
- onnx-community/Qwen3-Reranker-0.6B-ONNX
- onnx-community/Qwen3.5-2B-ONNX
- onnx-community/Qwen3.5-4B-ONNX
- onnx-community/Qwen3.5-0.8B-ONNX
- onnx-community/Qwen3-VL-2B-Instruct-ONNX
- onnx-community/ONNX_Qwen3-Embedding-0.6B
- onnx-community/Nanbeige4.1-3B-ONNX
- onnx-community/Qwen3-Embedding-8B-ONNX
- onnx-community/Qwen3-Embedding-4B-ONNX
- onnx-community/bge-reranker-v2-m3-ONNX
- onnx-community/all-MiniLM-L6-v2-ONNX

## 添加新模型

参见 [开发指南](./development.zh-CN.md) 了解添加新模型集成的说明。

## 许可证合规

**重要提示**：每个模型都有自己的许可证。在生产环境使用前请查看模型许可证。一些关键注意事项：

- **Apache 2.0**: 宽松许可，支持商业使用
- **MIT**: 宽松许可，支持商业使用
- **Qwen 研究许可协议**: 研究用途，可能有使用限制
- **腾讯混元社区许可协议**: 自定义许可，请查看条款
- **CC BY-NC 4.0**: 仅限非商业用途

在生产环境部署前，请务必验证许可证条款。

## 模型更新

模型不定期更新。

## 性能基准

CPU (M1 Pro) 上的近似推理速度：

| 模型 | 任务 | Tokens/秒 |
|------|------|-----------|
| Qwen3-0.6B | 文本 | 40-50 |
| Qwen2.5-VL-3B | 视觉 | 20-30 |
| Qwen3-ASR-0.6B | ASR | 200-500x |
| VoxCPM-0.5B | TTS | 实时 |

*基准测试因硬件和输入大小而异。*
