# OCR
## PaddleOCR-VL
#### Options prompt: 
"OCR:" | "Table Recognition:" | "Formula Recognition:" | "Chart Recognition:"

## PaddleOCR-VL1.5/1.6
#### Options prompt: 
"OCR:" | "Table Recognition:" | "Formula Recognition:" | "Chart Recognition:" | "Spotting:" | "Seal Recognition:"

## DeepSeekOCR
#### Main Prompts
1. document: <image>\n<|grounding|>Convert the document to markdown.
2. without layouts: <image>\nFree OCR.

#### Metadata
* base_size: 512 | 640 | 1024 | 1280
* image_size: 512 | 640 | 1024 | 1280
* crop_mode: false | true

##### Example:
```json
{"base_size": "640", "image_size": "640", "crop_mode": "false"}
```
#### Processing Types
| Type | base_size | image_size | crop_mode |
|------|-----------|------------|-----------|
| Tiny | 512 | 512 | false | 
| Small | 640 | 640 | false | 
| Base | 1024 | 1024 | false | 
| Large | 1280 | 1280 | false | 
| Gundam | 1024 | 640 | true | 

## DeepSeekOCR2
#### Main Prompts
1. document: <image>\n<|grounding|>Convert the document to markdown.
2. without layouts: <image>\nFree OCR.

#### Metadata
* crop_mode: false | true


## GLM-OCR
#### Prompt Limited
##### Document Parsing prompt： 
"Text Recognition:" | "Formula Recognition:" | "Table Recognition:"
###### Information Extraction: 
extract structured information from documents. Prompts must follow a strict JSON schema, For example, to extract personal ID information:
```json
Please output the information in the image in the following JSON format:
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
#### Application-oriented Prompts

| Task | English | Chinese |
|------|---------|---------|
| Spotting | Detect and recognize text in the image, and output the text coordinates in a formatted manner. |	检测并识别图片中的文字，将文本坐标格式化输出。|
| Parsing |	1. Identify the formula in the image and represent it using LaTeX format. <br> 2.Parse the table in the image into HTML. <br> 3. Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts. <br> 4.Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | 1. 识别图片中的公式，用 LaTeX 格式表示。 <br>  2. 把图中的表格解析为 HTML。 <br> 3. 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。 <br> 4. 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| Information Extraction | 1. Output the value of Key. <br> 2. Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format. <br> 3. Extract the subtitles from the image. | 1. 输出 Key 的值。 <br> 2. 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。 <br> 3. 提取图片中的字幕。 |
| Translation | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. |	先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |


Here is the continuation for the English [model-card.md](file:///home/jhq/rust_code/aha/docs/model-card.md) file, translating and adapting the content from the Chinese version regarding VoxCPM models.

# TTS
## VoxCPM (0.5B / 1.5)

#### Mode Description
*   **Zero-shot TTS (Default)**: Generates speech directly without reference audio.
*   **Voice Cloning**: Requires reference audio (`audio_url`) and its corresponding transcript (`prompt_text`).
    *   **Note**: For versions 0.5B/1.5, if `audio_url` is provided, `prompt_text` **must** be provided in `metadata`. Conversely, if there is no `audio_url`, `prompt_text` should not be provided.

#### Metadata Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt_text` | String | Conditional | **Required for Cloning**. The transcript of the reference audio. Used to extract speaker characteristics. |
| `min_len` | Integer | No | Minimum generation length (tokens/steps), default `2`. |
| `max_len` | Integer | No | Maximum generation length (tokens/steps), default `4096`. |
| `inference_timesteps` | Integer | No | Number of inference steps. Affects quality and speed. Default `10`. Higher values yield better quality but slower speed. |
| `cfg_value` | Float | No | Classifier-Free Guidance value. Controls adherence to the prompt. Default `2.0`. |
| `retry_badcase_ratio_threshold` | Float | No | Threshold for retrying bad cases. Default `6.0`. |

#### Example 1: Zero-shot TTS (No Reference)
```json
{
    "model": "OpenBMB/VoxCPM-0.5B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Hello, this is a test speech."
                }
            ]
        }
    ]
}
```

#### Example 2: Voice Cloning (With Reference)
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
                    "text": "This is the target text I want to generate."
                }
            ]
        }
    ],
    "metadata": {
        "prompt_text": "This is the text content corresponding to the reference audio."
    }
}
```

#### Response Example
The model returns a JSON object containing Base64-encoded audio data. The audio format is WAV.

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

VoxCPM2 supports three advanced voice generation modes. Switch between modes by combining content types in `messages` and parameters in `metadata`.

### 1. Voice Design
Generate a completely new voice from natural language descriptions alone (gender, age, tone, emotion, pace, etc.). **No reference audio required.**

*   **Input**: Text message only.
*   **Metadata**: Optional `control_instruction` for finer control (e.g., emotion, pace).

#### Example:
```json
{
    "model": "OpenBMB/VoxCPM2",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Welcome to the future world."
                }
            ]
        }
    ],
    "metadata": {
        "control_instruction": "Young female, energetic" 
    }
}
```

### 2. Controllable Cloning
Clone a voice from a short audio clip, with optional style guidance to steer emotion, pace, and expression while preserving timbre.

*   **Input**: Reference audio (`audio_url`) + Target text.
*   **Metadata**: Optional `control_instruction` to adjust style (emotion, pace, etc.). **`prompt_text` is NOT required.**

#### Example:
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
                    "text": "This news broadcast needs to be more serious and calm."
                }
            ]
        }
    ],
    "metadata": {
        "control_instruction": "serious, slow pace"
    }
}
```

### 3. Ultimate Cloning
Provide reference audio and its verbatim transcript for audio-continuation cloning or high-fidelity cloning. Every vocal nuance is faithfully reproduced.

*   **Input**: Reference audio (`audio_url`) + Target text.
*   **Metadata**: **Must** provide `prompt_text` (accurate transcript of the reference audio).

#### Example:
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
                    "text": "This is the new content to be spoken next."
                }
            ]
        }
    ],
    "metadata": {
        "prompt_text": "This is the actual text spoken in the reference audio."
    }
}
```

#### General Metadata Parameters (Applicable to all VoxCPM2 modes)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt_text` | String | Ultimate Cloning Only | Verbatim transcript of the reference audio. Required only for Ultimate Cloning mode. |
| `control_instruction` | String | No | Natural language instruction to control style for Voice Design or Controllable Cloning (e.g., "angry", "whispering", "fast"). |
| `min_len` | Integer | No | Minimum generation length, default `2`. |
| `max_len` | Integer | No | Maximum generation length, default `4096`. |
| `inference_timesteps` | Integer | No | Inference steps, default `10`. |
| `cfg_value` | Float | No | CFG value, default `2.0`. |
| `retry_badcase_ratio_threshold` | Float | No | Bad case retry threshold, default `6.0`. |


#### Response Example
Same as VoxCPM 0.5B/1.5, returns a JSON object containing Base64-encoded WAV audio.

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