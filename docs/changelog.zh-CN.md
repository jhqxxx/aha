# 更新日志

所有 aha 的重大更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

### 2026-04-10
- 修复 LiquidAI/LFM2.5-VL-450M chat_template 加载bug

### 2026-04-08
- 添加 VoxCPM2

### 0.2.5 (2026-04-06)
- 添加 qwen3-embedding/qwen3-reranker/all-minilm-l6-v2

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

### 2026-03-30
- 新增 LFM2.5VL-1.6B
- 新增 LFM2VL-1.6B

### v0.2.4 (2026-03-23)
- 新增 LFM2.5-1.2B-Instruct
- 新增 LFM2-1.2B

### v0.2.3 (2026-03-18)
- 新增 DeepSeek-OCR-2

### 2026-03-17
- 新增 PaddleOCR-VL1.5 模型
- 修复 qwen3.5 position_ids 创建错误
- cli 参数增加 
  - gguf_path: 本地 GGUF 模型权重路径（加载 GGUF 模型时需要）
  - mmproj_path: 本地 mmproj GGUF 权重路径（加载多模态 GGUF 时需要）
- WhichModel 增加 qwen3.5-gguf

### 2026-03-16
- 增加 Qwen3.5 mmproj

### 2026-03-14
- 更新rust版本 
- 增加了对 Qwen3.5 gguf 的支持，但 4B 模型仍然存在问题；待解决。

## [0.2.2] (2026-03-07)
- 新增 GLM-OCR 模型

## [0.2.1] - (2026-03-05)
- 新增Qwen3.5 模型

### 2026-03-01
- 更新 interpolate.rs

### 2026-02-24
- 更新 candle 版本 0.9.2

## [0.2.0] - 2026-02-05

### 新增
- Qwen3-ASR 语音识别模型

## [0.1.9] - 2026-01-31

### 新增
- CLI `list` 子命令，显示支持的模型
- CLI 子命令结构支持（`cli`、`serv`、`download`、`run`）
- 通过新的 `run` 子命令直接进行模型推理

### 修复
- Qwen3VL thinking startswith bug
- `aha run` 多输入 bug

## [0.1.8] - 2026-01-17

### 新增
- Qwen3 文本模型支持
- Fun-ASR-Nano-2512 语音识别模型

### 修复
- ModelScope Fun-ASR-Nano 模型加载错误

### 变更
- 使用 rubato 更新音频重采样

## [0.1.7] - 2026-01-07

### 新增
- GLM-ASR-Nano-2512 语音识别模型
- Metal (GPU) 支持，适用于 Apple Silicon
- 动态主目录和模型下载脚本

## [0.1.6] - 2025-12-23

### 新增
- RMBG-2.0 背景移除模型
- 图像和音频 API 端点

### 变更
- RMBG2.0 图像处理性能优化

## [0.1.5] - 2025-12-11

### 新增
- VoxCPM1.5 语音生成模型
- PaddleOCR-VL 文字识别模型

## [0.1.4] - 2025-12-09

### 新增
- PaddleOCR-VL 模型支持
- FFmpeg 多媒体处理功能

## [0.1.3] - 2025-12-03

### 新增
- Hunyuan-OCR 模型支持

## [0.1.2] - 2025-11-23

### 新增
- DeepSeek-OCR 模型支持

## [0.1.1] - 2025-11-12

### 新增
- Qwen3-VL 系列模型 (2B, 4B, 8B, 32B)

### 修复
- 为 Qwen3VL 的 tie_word_embeddings 添加 serde 默认值

## [0.1.0] - 2025-10-10

### 新增
- 初始版本发布
- Qwen2.5-VL 模型支持
- MiniCPM4 模型支持
- VoxCPM 语音生成模型
- 兼容 OpenAI 的 REST API
- 所有模型类型的 CLI 界面
