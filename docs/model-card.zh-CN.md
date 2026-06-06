# OCR
## PaddleOCR-VL
#### 可选提示词: 
"OCR:" | "Table Recognition:" | "Formula Recognition:" | "Chart Recognition:"

## PaddleOCR-VL1.5/1.6
#### 可选提示词: 
"OCR:" | "Table Recognition:" | "Formula Recognition:" | "Chart Recognition:" | "Spotting:" | "Seal Recognition:"

## DeepSeekOCR
#### 主要提示词
1. 文档: <image>\n<|grounding|>Convert the document to markdown.
2. 无布局信息: <image>\nFree OCR.

#### Metadata
* base_size: 512 | 640 | 1024 | 1280
* image_size: 512 | 640 | 1024 | 1280
* crop_mode: false | true

##### 示例:
```json
{"base_size": "640", "image_size": "640", "crop_mode": "false"}
```
#### 处理模式
| Type | base_size | image_size | crop_mode |
|------|-----------|------------|-----------|
| Tiny | 512 | 512 | false | 
| Small | 640 | 640 | false | 
| Base | 1024 | 1024 | false | 
| Large | 1280 | 1280 | false | 
| Gundam | 1024 | 640 | true | 

## DeepSeekOCR2
#### 主要提示词
1. 文档: <image>\n<|grounding|>Convert the document to markdown.
2. 无布局信息: <image>\nFree OCR.

#### Metadata
* crop_mode: false | true

## GLM-OCR
#### 限定提示词
##### 文档解析： 
"Text Recognition:" | "Formula Recognition:" | "Table Recognition:"
###### 信息提取: 
extract structured information from documents. Prompts must follow a strict JSON schema, For example, to extract personal ID information:
```json
请以以下 JSON 格式输出图像中的信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}
```

## HunyuanOCR
#### 场景化提示词

| 任务 | 英文 | 中文 |
|------|---------|---------|
| 检测 | Detect and recognize text in the image, and output the text coordinates in a formatted manner. |	检测并识别图片中的文字，将文本坐标格式化输出。|
| 解析 |	1. Identify the formula in the image and represent it using LaTeX format. <br> 2.Parse the table in the image into HTML. <br> 3. Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts. <br> 4.Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | 1. 识别图片中的公式，用 LaTeX 格式表示。 <br>  2. 把图中的表格解析为 HTML。 <br> 3. 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。 <br> 4. 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| 信息提取 | 1. Output the value of Key. <br> 2. Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format. <br> 3. Extract the subtitles from the image. | 1. 输出 Key 的值。 <br> 2. 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。 <br> 3. 提取图片中的字幕。 |
| 翻译 | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. |	先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |


# TTS
## VoxCPM (0.5B / 1.5)

### 模式说明
*   **Zero-shot TTS (默认模式)**: 无需参考音频，直接生成语音。
*   **Voice Cloning (克隆模式)**: 需要提供参考音频 (`audio_url`) 和对应的参考文本 (`prompt_text`)。
    *   **注意**: 对于 0.5B/1.5 版本，如果提供了 `audio_url`，则**必须**在 `metadata` 中提供 `prompt_text`；反之，如果没有 `audio_url`，则不应提供 `prompt_text`。

### Metadata 参数

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `prompt_text` | String | 条件必填 | **克隆模式必填**。参考音频对应的转录文本。用于提取说话人特征。 |
| `min_len` | Integer | 否 | 最小生成长度 (tokens/steps)，默认 `2`。 |
| `max_len` | Integer | 否 | 最大生成长度 (tokens/steps)，默认 `4096`。 |
| `inference_timesteps` | Integer | 否 | 推理步数，影响生成质量和速度，默认 `10`。值越大质量越高，速度越慢。 |
| `cfg_value` | Float | 否 | Classifier-Free Guidance 值，控制对提示词的遵循程度，默认 `2.0`。 |
| `retry_badcase_ratio_threshold` | Float | 否 | 坏案例重试阈值，默认 `6.0`。 |

#### 示例 1: Zero-shot TTS (无参考)
```json
{
    "model": "OpenBMB/VoxCPM-0.5B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "你好，这是一个测试语音。"
                }
            ]
        }
    ]
}
```

#### 示例 2: Voice Cloning (有参考)
```json
{
    "model": "OpenBMB/VoxCPM-0.5B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": {
                        "url": "https://example.com/reference.wav"
                    }
                },              
                {
                    "type": "text", 
                    "text": "这是我要生成的目标文本。"
                }
            ]
        }
    ],
    "metadata": {
        "prompt_text": "这是参考音频对应的文字内容。"
    }
}
```

#### 返回结果示例
模型将返回一个包含 Base64 编码音频数据的 JSON 对象。音频格式为 WAV。

```json
{
    "id": "chatcmpl-uuid...",
    "object": "chat.completion",
    "created": 1710000000,
    "model": "OpenBMB/VoxCPM-0.5B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": {
                            "url": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA..."
                        }
                    }
                ]
            },
            "finish_reason": "stop"
        }
    ],
    "usage": null
}
```


## VoxCPM2

VoxCPM2 支持三种高级语音生成模式。通过组合 `messages` 中的内容类型和 `metadata` 参数来切换模式。

### 1. Voice Design (声音设计)
仅通过自然语言描述生成全新声音（性别、年龄、语气、情感、语速等），**不需要**参考音频。

*   **输入**: 纯文本消息。
*   **Metadata**: 可选 `control_instruction` 用于更精细的控制（如情绪、语速）。

#### 示例:
```json
{
    "model": "OpenBMB/VoxCPM2",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "欢迎来到未来世界。"
                }
            ]
        }
    ],
    "metadata": {
        "control_instruction": "年轻女性，活力" 
    }
}
```

### 2. Controllable Cloning (可控克隆)
从短音频片段中克隆声音，并可选地通过风格指导来控制情感、语速和表达，同时保留音色。

*   **输入**: 参考音频 (`audio_url`) + 目标文本。
*   **Metadata**: 可选 `control_instruction` 用于调整风格（情绪、语速等）。**不需要** `prompt_text`。

#### 示例:
```json
{
    "model": "OpenBMB/VoxCPM2",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": {
                        "url": "https://example.com/short_clip.wav"
                    }
                },
                {
                    "type": "text", 
                    "text": "这段新闻播报需要更加严肃和沉稳。"
                }
            ]
        }
    ],
    "metadata": {
        "control_instruction": "serious, slow pace"
    }
}
```

### 3. Ultimate Cloning (极致克隆)
提供参考音频及其逐字稿，进行音频续写或高保真克隆。每个声音细节都被忠实还原。

*   **输入**: 参考音频 (`audio_url`) + 目标文本。
*   **Metadata**: **必须**提供 `prompt_text` (参考音频的准确转录)。

#### 示例:
```json
{
    "model": "OpenBMB/VoxCPM2",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": {
                        "url": "https://example.com/reference.wav"
                    }
                },
                {
                    "type": "text", 
                    "text": "这是接下来要说的新内容。"
                }
            ]
        }
    ],
    "metadata": {
        "prompt_text": "这是参考音频中实际说的文字。"
    }
}
```

#### 通用 Metadata 参数 (适用于所有 VoxCPM2 模式)

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `prompt_text` | String | 仅 Ultimate Cloning | 参考音频的逐字稿。仅在极致克隆模式下需要。 |
| `control_instruction` | String | 否 | 自然语言指令，用于控制 Voice Design 或 Controllable Cloning 的风格（如 "angry", "whispering", "fast"）。 |
| `min_len` | Integer | 否 | 最小生成长度，默认 `2`。 |
| `max_len` | Integer | 否 | 最大生成长度，默认 `4096`。 |
| `inference_timesteps` | Integer | 否 | 推理步数，默认 `10`。 |
| `cfg_value` | Float | 否 | CFG 值，默认 `2.0`。 |
| `retry_badcase_ratio_threshold` | Float | 否 | 坏案例重试阈值，默认 `6.0`。 |

#### 返回结果示例
与 VoxCPM 0.5B/1.5 相同，返回包含 Base64 编码 WAV 音频的 JSON 对象。

```json
{
    "id": "chatcmpl-uuid...",
    "object": "chat.completion",
    "created": 1710000000,
    "model": "OpenBMB/VoxCPM2",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": {
                            "url": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA..."
                        }
                    }
                ]
            },
            "finish_reason": "stop"
        }
    ],
    "usage": null
}
```