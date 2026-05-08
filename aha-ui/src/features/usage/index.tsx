import { useState } from "react"
import { BookOpen, Copy, Check, Code, Terminal, FileType } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Header } from "@/components/layout/header"
import { Main } from "@/components/layout/main"
import { ProfileDropdown } from "@/components/profile-dropdown"
import { ThemeSwitch } from "@/components/theme-switch"

interface CodeBlockProps {
  code: string
  language?: string
}

function CodeBlock({ code, language = "bash" }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const langIcons: Record<string, React.ReactNode> = {
    bash: <Terminal className="w-3.5 h-3.5" />,
    python: <FileType className="w-3.5 h-3.5" />,
    typescript: <Code className="w-3.5 h-3.5" />,
  }

  return (
    <div className="relative group">
      <div className="flex items-center justify-between px-4 py-1.5 bg-muted/50 border-b border-border/50 rounded-t-lg">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {langIcons[language] || null}
          {language}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-green-500" />
              已复制
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              复制
            </>
          )}
        </button>
      </div>
      <pre className="bg-[#1e1e2e] text-gray-300 p-4 rounded-b-lg overflow-x-auto text-sm leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function ApiSection({
  title,
  method,
  path,
  description,
  children,
}: {
  title: string
  method: string
  path: string
  description: string
  children: React.ReactNode
}) {
  const methodColors: Record<string, string> = {
    GET: "bg-green-500/10 text-green-600 border-green-200 dark:border-green-800 dark:text-green-400",
    POST: "bg-blue-500/10 text-blue-600 border-blue-200 dark:border-blue-800 dark:text-blue-400",
  }

  return (
    <div className="space-y-3">
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
      <div className="flex items-center gap-2 font-mono text-sm">
        <span className={`px-2 py-0.5 rounded border text-xs font-medium ${methodColors[method] || ""}`}>
          {method}
        </span>
        <span className="text-foreground">{path}</span>
      </div>
      {children}
    </div>
  )
}

export function UsagePage() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:10100")

  return (
    <>
      <Header>
        <div className="flex items-center gap-2 ms-auto">
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BookOpen className="w-6 h-6" />
            API 使用指南
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            如何通过 HTTP 接口调用本地模型推理服务
          </p>
        </div>

        {/* 服务地址配置 */}
        <Card className="p-4 mb-6">
          <div className="flex items-end gap-4">
            <div className="flex-1 space-y-2">
              <Label htmlFor="base-url">服务地址</Label>
              <Input
                id="base-url"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://127.0.0.1:10100"
              />
            </div>
            <p className="text-xs text-muted-foreground pb-1">
              请确保服务已启动，地址与「启动服务」页面配置一致
            </p>
          </div>
        </Card>

        <div className="space-y-8">
          {/* 1. 查看可用模型 */}
          <Card className="p-5 space-y-4">
            <ApiSection
              title="查看可用模型"
              method="GET"
              path="/v1/models"
              description="获取当前服务中可用的模型列表"
            >
              <div className="space-y-3">
                <CodeBlock
                  language="bash"
                  code={`curl ${baseUrl}/v1/models`}
                />
                <CodeBlock
                  language="python"
                  code={`import requests

response = requests.get("${baseUrl}/v1/models")
models = response.json()
print(models)`}
                />
                <CodeBlock
                  language="typescript"
                  code={`const response = await fetch("${baseUrl}/v1/models")
const models = await response.json()
console.log(models)`}
                />
              </div>
            </ApiSection>
          </Card>

          {/* 2. Chat Completions */}
          <Card className="p-5 space-y-4">
            <ApiSection
              title="对话补全"
              method="POST"
              path="/v1/chat/completions"
              description="向模型发送对话消息，获取推理回复。兼容 OpenAI API 格式。"
            >
              <div className="space-y-3">
                <CodeBlock
                  language="bash"
                  code={`curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "your-model-id",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手"},
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": false
  }'`}
                />
                <CodeBlock
                  language="python"
                  code={`import requests

response = requests.post(
    "${baseUrl}/v1/chat/completions",
    json={
        "model": "your-model-id",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False,
    },
)
result = response.json()
print(result["choices"][0]["message"]["content"])`}
                />
                <CodeBlock
                  language="typescript"
                  code={`const response = await fetch("${baseUrl}/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "your-model-id",
    messages: [
      { role: "system", content: "你是一个有用的助手" },
      { role: "user", content: "你好，请介绍一下你自己" }
    ],
    temperature: 0.7,
    max_tokens: 2048,
    stream: false,
  }),
})
const result = await response.json()
console.log(result.choices[0].message.content)`}
                />
              </div>

              <div className="mt-4 p-3 bg-muted/50 rounded-md text-sm space-y-2">
                <p className="font-medium">请求参数说明</p>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-1 pr-2">参数</th>
                      <th className="text-left py-1 pr-2">类型</th>
                      <th className="text-left py-1">说明</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-muted">
                      <td className="py-1 pr-2 font-mono">model</td>
                      <td className="py-1 pr-2">string</td>
                      <td className="py-1">模型 ID</td>
                    </tr>
                    <tr className="border-b border-muted">
                      <td className="py-1 pr-2 font-mono">messages</td>
                      <td className="py-1 pr-2">array</td>
                      <td className="py-1">对话消息列表</td>
                    </tr>
                    <tr className="border-b border-muted">
                      <td className="py-1 pr-2 font-mono">temperature</td>
                      <td className="py-1 pr-2">number</td>
                      <td className="py-1">采样温度 (0~2)，默认 1.0</td>
                    </tr>
                    <tr className="border-b border-muted">
                      <td className="py-1 pr-2 font-mono">max_tokens</td>
                      <td className="py-1 pr-2">number</td>
                      <td className="py-1">最大生成 token 数</td>
                    </tr>
                    <tr>
                      <td className="py-1 pr-2 font-mono">stream</td>
                      <td className="py-1 pr-2">boolean</td>
                      <td className="py-1">是否流式输出，默认 false</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </ApiSection>
          </Card>

          {/* 3. 流式输出 */}
          <Card className="p-5 space-y-4">
            <ApiSection
              title="流式对话"
              method="POST"
              path="/v1/chat/completions (stream)"
              description="使用 SSE (Server-Sent Events) 实现流式输出，逐 token 返回推理结果。"
            >
              <div className="space-y-3">
                <CodeBlock
                  language="bash"
                  code={`curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "your-model-id",
    "messages": [
      {"role": "user", "content": "用 Python 写一个递归遍历目录的例子"}
    ],
    "stream": true
  }'`}
                />
                <CodeBlock
                  language="python"
                  code={`import requests

response = requests.post(
    "${baseUrl}/v1/chat/completions",
    json={
        "model": "your-model-id",
        "messages": [
            {"role": "user", "content": "用 Python 写一个递归遍历目录的例子"}
        ],
        "stream": True,
    },
    stream=True,
)
for line in response.iter_lines():
    if line:
        text = line.decode("utf-8").removeprefix("data: ")
        if text != "[DONE]":
            import json
            chunk = json.loads(text)
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            print(delta, end="", flush=True)`}
                />
                <CodeBlock
                  language="typescript"
                  code={`const response = await fetch("${baseUrl}/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "your-model-id",
    messages: [
      { role: "user", content: "用 Python 写一个递归遍历目录的例子" }
    ],
    stream: true,
  }),
})

const reader = response.body!.getReader()
const decoder = new TextDecoder()
let buffer = ""

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  buffer += decoder.decode(value, { stream: true })
  const lines = buffer.split("\\n")
  buffer = lines.pop() || ""

  for (const line of lines) {
    const text = line.replace(/^data: /, "").trim()
    if (!text || text === "[DONE]") continue

    const chunk = JSON.parse(text)
    const content = chunk.choices?.[0]?.delta?.content || ""
    process.stdout.write(content)
  }
}`}
                />
              </div>
            </ApiSection>
          </Card>

          {/* 4. 使用 OpenAI SDK */}
          <Card className="p-5 space-y-4">
            <h3 className="text-base font-semibold">使用 OpenAI SDK 调用</h3>
            <p className="text-sm text-muted-foreground">
              由于服务兼容 OpenAI API 格式，你可以直接使用 OpenAI 官方 SDK，只需修改 base URL 即可。
            </p>
            <div className="space-y-3">
              <CodeBlock
                language="python"
                code={`from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}/v1",
    api_key="not-needed",  # 本地服务不需要 API Key
)

response = client.chat.completions.create(
    model="your-model-id",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`}
              />
              <CodeBlock
                language="typescript"
                code={`import OpenAI from "openai"

const client = new OpenAI({
  baseURL: "${baseUrl}/v1",
  apiKey: "not-needed",
})

const stream = await client.chat.completions.create({
  model: "your-model-id",
  messages: [{ role: "user", content: "你好" }],
  stream: true,
})

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content || ""
  process.stdout.write(content)
}`}
              />
            </div>
          </Card>

          {/* 5. 健康检查 */}
          <Card className="p-5 space-y-4">
            <ApiSection
              title="健康检查"
              method="GET"
              path="/health"
              description="检查服务运行状态"
            >
              <CodeBlock
                language="bash"
                code={`curl ${baseUrl}/health`}
              />
            </ApiSection>
          </Card>
        </div>
      </Main>
    </>
  )
}
