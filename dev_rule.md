# AHA 开发规范

本文档不是通用 Rust 模板，而是基于当前 `aha` 仓库真实代码结构整理出的项目级规范。目标有两个：

1. 统一现有模型目录的职责分层。
2. 为未来新增原生模型、GGUF 模型、ONNX 模型提供可执行的接入标准。

---

## 1. 当前项目真实结构

### 1.1 顶层目录职责

```text
aha/
├── src/
│   ├── main.rs            # CLI 与服务启动入口
│   ├── lib.rs             # 对外库导出
│   ├── api/               # HTTP/OpenAI 兼容接口
│   ├── exec/              # `aha run` 的模型直调入口
│   ├── models/            # 模型实现与模型工厂
│   ├── tokenizer/         # tokenizer 封装
│   ├── chat_template/     # chat template 封装
│   ├── position_embed/    # 位置编码
│   ├── process.rs         # 服务进程管理
│   └── utils/             # 下载、设备、dtype、文件查找等通用工具
├── tests/                 # 集成测试与格式测试
├── docs/                  # 面向用户的文档
└── dev_rule.md            # 本规范
```

### 1.2 模型接入链路

新增一个模型，必须理解下面这条链路，而不是只加 `src/models/<name>/`：

1. `src/models/<name>/`
   实现模型本体、配置、推理入口。
2. `src/models/mod.rs`
   注册模块、枚举 `WhichModel`、模型元信息、模型工厂 `load_model`、`ModelInstance` 能力分派。
3. `src/exec/<name>.rs`
   接入 `aha run -m <model>` 的直调能力。
4. `src/exec/mod.rs`
   导出 exec 模块。
5. `src/main.rs`
   将 `WhichModel` 路由到具体 exec 实现；必要时处理 GGUF / mmproj / ONNX 的路径参数。
6. `src/api/mod.rs`
   若模型属于 `chat` / `embedding` / `rerank` / `asr` / `ocr` 等服务能力，需要保证 `ModelInstance` 对应方法可用。
7. `tests/`
   至少补齐加载测试和最小推理测试。
8. `docs/`、`README*`、`changelog`
   更新用户可见说明。

---

## 2. 模型目录统一规范

### 2.1 最低标准

从现在开始，`src/models/<model_name>/` 的最低结构如下：

```text
src/models/<model_name>/
├── mod.rs
├── config.rs
├── model.rs
└── generate.rs
```

说明：

- `mod.rs` 必须只做模块导出，不写业务逻辑。
- `config.rs` 负责配置结构体、默认值、配置文件加载辅助。
- `model.rs` 负责底层网络结构、后端封装、权重加载、核心推理逻辑。
- `generate.rs` 负责对外可调用入口。

### 2.2 可选文件

满足以下条件时再新增文件，不要随意拆目录：

- `processor.rs`
  仅用于多模态输入预处理或复杂后处理，例如图像、音频、视频、OCR patch 处理。
- `tokenizer.rs`
  仅当模型目录需要自定义 tokenizer 适配，且通用 `TokenizerModel` 不够用。
- `audio_vae.rs`、`minicpm4.rs` 等附属文件
  仅当模型实现包含稳定、独立的子模块时允许拆出。

### 2.3 各文件职责边界

#### `mod.rs`

只允许：

- `pub mod ...`
- 必要时 `pub use ...`

不允许：

- 业务实现
- 大段常量
- 初始化逻辑

#### `config.rs`

负责：

- `config.json`
- `generation_config.json`
- `preprocessor_config.json`
- 与格式相关但不属于运行态的配置映射

不负责：

- 真正加载权重
- 调用 tokenizer
- 调用 Candle forward

#### `model.rs`

负责：

- Candle 模型结构体
- 权重加载
- 格式后端封装
- 核心 `forward` / `embed` / `rerank` / `transcribe` / `process_image` 等内部能力

不负责：

- CLI 参数解析
- HTTP 请求结构
- `aha_openai_dive` 响应组装

#### `generate.rs`

负责：

- 作为当前模型目录的对外入口类型
- 将 tokenizer、processor、model backend 串起来
- 实现 `GenerateModel` trait，或提供该模型暴露给 `ModelInstance` 的统一入口

约束：

- 对外入口类型名必须稳定，供 `src/models/mod.rs`、`src/exec/*`、`tests/*` 使用。
- 目录重构时优先保持公开方法不变，例如 `init`、`generate`、`generate_stream`、`embed`、`rerank`。

---

## 3. 按能力划分的目录模板

### 3.1 文本生成模型

适用：

- Qwen3
- Qwen3.5
- MiniCPM4

模板：

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
└── generate.rs
```

规则：

- `generate.rs` 负责 `GenerateModel` 实现。
- `model.rs` 负责 `forward`、KV cache、权重加载。

### 3.2 多模态模型

适用：

- Qwen3VL
- Qwen2.5VL
- Qwen3ASR
- DeepSeek-OCR
- Hunyuan-OCR
- PaddleOCR-VL

模板：

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
├── generate.rs
└── processor.rs
```

规则：

- `processor.rs` 负责把请求输入变成模型可消费的 tensor 或替换文本。
- `generate.rs` 只做流程编排，不堆复杂前处理。

### 3.3 Embedding / Reranker 模型

适用：

- Qwen3-Embedding
- Qwen3-Reranker
- 后续任意检索相关模型

模板：

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
└── generate.rs
```

规则：

- `model.rs` 负责 embedding backend 或 rerank backend。
- `generate.rs` 提供稳定对外入口，供 `ModelInstance::embedding` / `ModelInstance::rerank` 调用。
- 公共检索算法优先放在 `src/models/common/retrieval.rs`，不要在各模型目录重复实现。

### 3.4 纯图像或工具型模型

适用：

- RMBG2.0

模板：

```text
src/models/<model>/
├── mod.rs
├── model.rs
└── generate.rs
```

补充要求：

- 如果模型已经存在稳定配置文件，仍然建议补 `config.rs`。
- 新增此类模型时，不要因为现有个别旧目录较薄，就继续复制旧写法。

---

## 4. 多格式模型规范

### 4.1 格式定义

当前项目已经在 `src/models/mod.rs` 中抽象出：

- `Safetensors`
- `Gguf`
- `Onnx`

新增模型时，必须先明确自己属于哪一种格式，不能等实现完成后再补枚举。

### 4.2 格式接入原则

#### 原生 / Safetensors 模型

规则：

- 默认走项目当前主路径。
- 权重查找统一使用 `find_type_files(path, "safetensors")`。
- 设备与 dtype 统一通过 `get_device`、`get_dtype` 获取。
- 能被自动下载的模型，必须在 `WhichModel::is_download_managed()` 语义下成立。

#### GGUF 模型

规则：

- 必须在 `WhichModel::artifact_format()` 返回 `ModelArtifactFormat::Gguf`。
- `main.rs` 中必须要求 `--gguf-path`，多模态 GGUF 额外支持 `--mmproj-path`。
- tokenizer 若从 GGUF metadata 构建，必须封装在 `model.rs` 或专门 backend 中，不允许散落在 exec 或 main。
- GGUF 的 mmproj 仅属于后端实现细节，不应污染上层 API 类型。

#### ONNX 模型

规则：

- 必须在 `WhichModel::artifact_format()` 返回 `ModelArtifactFormat::Onnx`。
- 若运行时尚未集成 ONNX 推理，仍然要保证：
  - 模型枚举和格式分类准确；
  - 测试能够验证 ONNX 文件可发现、可读取、可创建 session；
  - 错误提示明确说明“格式已识别，但 runtime 未集成”。
- ONNX Runtime 的 session 创建、输入输出张量映射必须收敛到模型目录内部，不允许堆到测试文件里作为“事实实现”。

### 4.3 多格式目录组织建议

对未来一个同时支持 `safetensors + gguf + onnx` 的模型，推荐结构：

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
├── generate.rs
├── backend_safetensors.rs   # 当实现明显变大时再拆
├── backend_gguf.rs
└── backend_onnx.rs
```

约束：

- 如果格式实现还很轻，先放在 `model.rs` 的私有 backend 结构体里。
- 只有当 `model.rs` 已明显过大，才允许拆 `backend_*` 文件。
- 不允许一开始就为了“看起来完整”拆很多空文件。

### 4.4 多格式统一入口要求

对于同一模型家族，不管底层格式是什么，对外入口必须统一：

- 同一个 `WhichModel` 语义只表达“模型变体”，不表达内部临时实现。
- `generate.rs` 的公开类型应稳定。
- `src/models/mod.rs::load_model()` 负责根据 `WhichModel` 选择正确初始化路径。

---

## 5. 命名与导出规范

### 5.1 目录命名

- 目录名统一使用小写蛇形命名。
- 与上游模型 ID 不一致时，以项目内部统一命名为准，例如 `qwen3_asr`、`qwen3_embedding`。

### 5.2 类型命名

- 配置类型：`<ModelName>Config`
- 生成配置类型：`<ModelName>GenerationConfig`
- 底层模型：`<ModelName>Model` 或 `<ModelName>Backend`
- 对外入口：`<ModelName>GenerateModel`、`<ModelName>EmbeddingModel`、`<ModelName>RerankerModel`
- 处理器：`<ModelName>Processor`

### 5.3 `mod.rs` 命名要求

必须导出到以下位置：

- `src/models/mod.rs`
- `src/exec/mod.rs`

禁止：

- 同一个模型目录里同时存在多个对外主入口但没有清晰职责说明。

---

## 6. 新增模型的强制清单

### 6.1 模型目录

必须完成：

1. 创建 `src/models/<model>/`
2. 按模板补齐最少文件
3. 保持职责边界清晰

### 6.2 模型注册

必须同时修改 `src/models/mod.rs`：

1. `pub mod <model>;`
2. `WhichModel` 新增枚举项
3. `LISTED_MODELS` 增加公开模型
4. `artifact_format()`
5. `openai_model_id()`
6. `owner()`
7. `model_id()`
8. `model_type()`
9. `ModelInstance` 增加实例变体
10. `load_model()` 增加加载分支

### 6.3 CLI 接入

必须同时修改：

1. `src/exec/<model>.rs`
2. `src/exec/mod.rs`
3. `src/main.rs`

要求：

- `aha run` 必须能走到该模型。
- GGUF / ONNX 如有特殊路径参数，必须在 `main.rs` 路由时处理。

### 6.4 API 接入

当模型支持服务能力时，必须确保：

- `chat` 类型实现 `GenerateModel`
- `embedding` 类型实现 `ModelInstance::embedding`
- `rerank` 类型实现 `ModelInstance::rerank`
- `asr` / `ocr` / `image` 类型能复用现有 API 行为

### 6.5 文档接入

至少更新：

- `README.md`
- `README.zh-CN.md`
- `docs/supported-models.md`
- `docs/supported-models.zh-CN.md`
- `docs/changelog.md`
- `docs/changelog.zh-CN.md`

如果 CLI 或 API 行为变更，还要更新：

- `docs/cli*.md`
- `docs/api*.md`

---

## 7. 测试规范

### 7.1 每个新模型至少要有的测试

1. 加载测试
   验证模型文件能被发现并初始化。
2. 最小推理测试
   验证 `generate` / `embed` / `rerank` / `transcribe` / `remove_background` 至少能跑通一次。
3. 错误输入测试
   验证空输入、非法输入、缺失文件路径能返回明确错误。

### 7.2 多格式模型测试矩阵

对于 `safetensors + gguf + onnx` 多格式模型，建议至少覆盖：

1. safetensors 加载测试
2. gguf 文件发现与 tokenizer/session/backend 加载测试
3. onnx 文件发现测试
4. onnxruntime session 创建测试
5. 至少一种真实输入的结果合理性测试

### 7.3 测试文件命名

- 单模型：`tests/test_<model>.rs`
- 多格式专项：`tests/test_<model>_multi_format.rs`

### 7.4 校验命令

最少校验：

```bash
cargo fmt --check
cargo check
```

建议校验：

```bash
cargo test --test <target_test>
```

注意：

- 大模型测试通常依赖本地权重，不要求在所有环境全量跑通。
- 但编译检查必须通过。

---

## 8. 代码风格约束

### 8.1 不允许的做法

- 在 `mod.rs` 堆业务逻辑
- 在 `exec` 中写模型核心推理代码
- 在 `main.rs` 中写模型权重加载细节
- 在测试文件里藏正式实现逻辑
- 为了“结构完整”创建一堆空模块
- 复制已有模型代码后只改字符串，不抽象公共逻辑

### 8.2 推荐做法

- 公共检索逻辑放 `src/models/common/retrieval.rs`
- 公共 GGUF 工具放 `src/models/common/gguf.rs`
- 设备、dtype、权重文件发现统一走 `src/utils/mod.rs`
- 公开方法保持小而稳定，内部再分层

### 8.3 兼容性原则

当重构模型目录时：

- 优先保持对外类型名不变
- 优先保持 `init(...)` 签名不变
- 优先保持 `src/models/mod.rs`、`src/exec/*`、测试里的调用方式不变

如果必须改公开接口，必须同步修改：

- `src/models/mod.rs`
- `src/exec/*`
- `src/main.rs`
- `src/api/mod.rs`
- `tests/*`
- 相关文档

---

## 9. 当前仓库的落地结论

根据当前仓库实际情况，后续新增模型必须遵循以下结论：

1. `qwen3_embedding`、`qwen3_reranker` 现在也必须按 `config.rs + model.rs + generate.rs + mod.rs` 维护。
2. 未来新增 Embedding/Reranker 模型，不允许再只保留 `mod.rs + generate.rs`。
3. `docs/development.md` 当前更接近通用模板，不应作为新增模型时的唯一依据；以本文件和实际代码结构为准。
4. ONNX 目前在仓库里已经有测试与格式识别语义，但运行时集成尚未完整落地；新增 ONNX 模型时，必须同时补 runtime 设计，而不是只补测试。
5. GGUF 模型接入必须从一开始就考虑 `main.rs` 路径参数、`WhichModel::artifact_format()`、`load_model()` 的完整链路。

---

## 10. 新增模型模板清单

### 10.1 原生文本模型

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
└── generate.rs
```

### 10.2 原生多模态模型

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
├── generate.rs
└── processor.rs
```

### 10.3 多格式检索模型

```text
src/models/<model>/
├── mod.rs
├── config.rs
├── model.rs
├── generate.rs
├── backend_safetensors.rs   # 可选
├── backend_gguf.rs          # 可选
└── backend_onnx.rs          # 可选
```

最终原则只有一句：

新增模型不是“把代码放进一个目录”，而是“把模型完整接入到 models factory、CLI、API、tests、docs 五条链路中”，并且目录职责必须与现有标准结构保持一致。
