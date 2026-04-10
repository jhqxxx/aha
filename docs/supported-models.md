# Supported Models

aha supports a growing collection of state-of-the-art AI models across multiple domains.

```shell
Available models:

Model ID                                 Owner                type       Download  
--------------------------------------------------------------------------------
sentence-transformers/all-MiniLM-L6-v2   sentence-transformers embedding    ✔       
LiquidAI/LFM2-1.2B                       LiquidAI             llm          ✔       
LiquidAI/LFM2.5-1.2B-Instruct            LiquidAI             llm          ✔       
LiquidAI/LFM2.5-VL-1.6B                  LiquidAI             vlm          ✔       
LiquidAI/LFM2-VL-1.6B                    LiquidAI             vlm          ✔       
OpenBMB/MiniCPM4-0.5B                    OpenBMB              llm          ✔       
Qwen/Qwen2.5-VL-3B-Instruct              Qwen                 vlm          ✔       
Qwen/Qwen2.5-VL-7B-Instruct              Qwen                 vlm                  
Qwen/Qwen3-0.6B                          Qwen                 llm          ✔       
Qwen/Qwen3-1.7B                          Qwen                 llm          ✔       
Qwen/Qwen3-4B                            Qwen                 llm          ✔       
Qwen/Qwen3.5-0.8B                        Qwen                 vlm          ✔       
Qwen/Qwen3.5-2B                          Qwen                 vlm                  
Qwen/Qwen3.5-4B                          Qwen                 vlm                  
Qwen/Qwen3.5-9B                          Qwen                 vlm                  
qwen3.5-gguf                             none                 vlm                  
Qwen/Qwen3-ASR-0.6B                      Qwen                 asr          ✔       
Qwen/Qwen3-ASR-1.7B                      Qwen                 asr                  
Qwen/Qwen3-Embedding-0.6B                Qwen                 embedding    ✔       
Qwen/Qwen3-Embedding-4B                  Qwen                 embedding            
Qwen/Qwen3-Embedding-8B                  Qwen                 embedding            
Qwen/Qwen3-Reranker-0.6B                 Qwen                 reranker     ✔       
Qwen/Qwen3-Reranker-4B                   Qwen                 reranker             
Qwen/Qwen3-Reranker-8B                   Qwen                 reranker             
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

## Language Model

| Model | Parameters | Model Id | License |
|-------|-----------|--------|---------|
| **Qwen3-0.6B** | 0.6B | Qwen/Qwen3-0.6B <br> Qwen/Qwen3-1.7B <br> Qwen/Qwen3-4B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM4-0.5B** | 0.5B | OpenBMB/MiniCPM4-0.5B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2-1.2B** | 1.2B | LiquidAI/LFM2-1.2B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/LICENSE) |
| **LFM2.5-1.2B-Instruct** | 1.2B | LiquidAI/LFM2.5-1.2B-Instruct | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |


## Embedding

| Model | Parameters | Model Id | License |
|-------|-----------|-------------|---------|
| **Qwen3-Embedding** | 0.6B <br> 4B <br> 8B| Qwen/Qwen3-Embedding-0.6B <br> Qwen/Qwen3-Embedding-4B <br> Qwen/Qwen3-Embedding-8B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **all-MiniLM-L6-v2** | 91M | sentence-transformers/all-MiniLM-L6-v2 | [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/LICENSE) |

## Reranker

| Model | Parameters | Model Id | License |
|-------|-----------|-------------|---------|
| **Qwen3-Reranker** | 0.6B <br> 4B <br> 8B| Qwen/Qwen3-Reranker-0.6B <br> Qwen/Qwen3-Reranker-4B <br> Qwen/Qwen3-Reranker-8B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Vision & Multimodal

| Model | Parameters | Model Id | License |
|-------|-----------|----------|---------|
| **Qwen2.5-VL** | 3B <br> 7B | Qwen/Qwen2.5-VL-3B-Instruct <br> Qwen/Qwen2.5-VL-7B-Instruct | [Qwen 研究许可协议](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) <br> [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)  |
| **Qwen3-VL** | 2B <br> 4B <br> 8B <br> 32B | Qwen/Qwen3-VL-2B-Instruct <br> Qwen/Qwen3-VL-4B-Instruct <br> Qwen/Qwen3-VL-8B-Instruct <br> Qwen/Qwen3-VL-32B-Instruct | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5** | 0.8B <br> 2B <br> 4B <br> 9B | Qwen/Qwen3.5-0.8B <br> Qwen/Qwen3.5-2B <br> Qwen/Qwen3.5-4B <br> Qwen/Qwen3.5-9B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2.5-VL** | 450M <br> 1.6B | LiquidAI/LFM2.5-VL-450M <br> LiquidAI/LFM2.5-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |
| **LFM2-VL-1.6B** | 1.6B | LiquidAI/LFM2-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |


## OCR

| Model | Languages | Model Id | License |
|-------|-----------|--------|---------|
| **PaddleOCR-VL** | Multi | PaddlePaddle/PaddleOCR-VL <br> PaddlePaddle/PaddleOCR-VL-1.5 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Hunyuan-OCR** | Chinese | Tencent-Hunyuan/HunyuanOCR | [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |
| **DeepSeek-OCR** | Multi | deepseek-ai/DeepSeek-OCR <br> deepseek-ai/DeepSeek-OCR-2 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **GLM-OCR** | 8 | ZhipuAI/GLM-OCR | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |


## Speech Recognition (ASR)

| Model | Parameters | Language | Model Id | License |
|-------|-----------|----------|----------|---------|
| **Fun-ASR-Nano-2512** | 2G | Chinese/English | FunAudioLLM/Fun-ASR-Nano-2512 | Not Specified |
| **GLM-ASR-Nano-2512** | 4.5G | Chinese/English | ZhipuAI/GLM-ASR-Nano-2512 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR** | 0.6B <br> 1.7B | Chinese/English | Qwen/Qwen3-ASR-0.6B <br> Qwen/Qwen3-ASR-1.7B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Audio Generation

| Model | version | Model Id | License |
|-------|-----------|---------|---------|
| **VoxCPM** | 1<br>1.5<br>2 | OpenBMB/VoxCPM-0.5B <br> OpenBMB/VoxCPM1.5 <br> OpenBMB/VoxCPM2 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Image Processing

| Model | Type | Model Id | License |
|-------|------|-------------|---------|
| **RMBG-2.0** | Background Removal | AI-ModelScope/RMBG-2.0 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) |

## Model Sources

Models are sourced from:

- [Hugging Face](https://huggingface.co) - Primary model hub
- [ModelScope](https://modelscope.cn) - Chinese model hub

## Adding New Models

See [Development Guide](./development.md) for instructions on adding new model integrations.

## License Compliance

**Important**: Each model has its own license. Please review the model's license before use in production. Some key considerations:

- **Apache 2.0**: Permissive, commercial-friendly
- **MIT**: Permissive, commercial-friendly
- **Qwen Research License**: Research use, may have restrictions
- **Tencent Hunyuan Community License**: Custom license, review terms
- **CC BY-NC 4.0**: Non-commercial only

Always verify license terms before deployment in production environments.

## Model Updates

Models updated from time to time. 

## Performance Benchmarks

Approximate inference speeds on CPU (M1 Pro):

| Model | Task | Tokens/sec |
|-------|------|------------|
| Qwen3-0.6B | Text | 40-50 |
| Qwen2.5-VL-3B | Vision | 20-30 |
| Qwen3-ASR-0.6B | ASR | 200-500x |
| VoxCPM-0.5B | TTS | Real-time |

*Benchmarks vary by hardware and input size.*
