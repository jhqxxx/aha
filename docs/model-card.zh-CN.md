# OCR
## PaddleOCR-VL
#### 可选提示词: 
"OCR:" | "Table Recognition:" | "Formula Recognition:" | "Chart Recognition:"

## PaddleOCR-VL1.5
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
## VoxCPM