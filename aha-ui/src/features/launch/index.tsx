import { useState, useEffect, useRef, useCallback } from "react"
import { invoke } from "@tauri-apps/api/core"
import { listen } from "@tauri-apps/api/event"
import {
  Play,
  Square,
  Terminal,
  RotateCw,
  Server,
  ExternalLink,
  Check,
  BookOpen,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Header } from "@/components/layout/header"
import { Main } from "@/components/layout/main"
import { ProfileDropdown } from "@/components/profile-dropdown"
import { ThemeSwitch } from "@/components/theme-switch"

interface ModelInfo {
  model_id: string
  owner: string
  model_type: string
  downloaded: boolean
}

interface LaunchConfig {
  models: string[]
  address: string
  port: number
  weight_path?: string | null
  save_dir?: string | null
  gguf_path?: string | null
  mmproj_path?: string | null
}

interface ServerStatus {
  running: boolean
  pid: number | null
  logs: string[]
}

const MODEL_TYPE_ORDER: Record<string, number> = {
  llm: 0,
  vlm: 1,
  ocr: 2,
  asr: 3,
  tts: 4,
  image: 5,
  embedding: 6,
  reranker: 7,
}

const TYPE_LABELS: Record<string, string> = {
  llm: "LLM",
  vlm: "VLM",
  ocr: "OCR",
  asr: "ASR",
  tts: "TTS",
  image: "图像",
  embedding: "嵌入",
  reranker: "重排序",
}

export function LaunchPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [address, setAddress] = useState("127.0.0.1")
  const [port, setPort] = useState("10100")
  const [weightPath, setWeightPath] = useState("")
  const [status, setStatus] = useState<ServerStatus>({
    running: false,
    pid: null,
    logs: [],
  })
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    invoke<ModelInfo[]>("list_models").then(setModels).catch(() => {})
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const s = await invoke<ServerStatus>("get_server_status")
      setStatus(s)
    } catch {}
  }, [])

  useEffect(() => {
    const unlistens: (() => void)[] = []

    listen<string>("server-log", (event) => {
      setStatus((prev) => ({
        ...prev,
        logs: [...prev.logs.slice(-1999), event.payload],
      }))
    }).then((fn) => unlistens.push(fn))

    listen<number>("server-started", () => {
      refreshStatus()
    }).then((fn) => unlistens.push(fn))

    listen("server-stopped", () => {
      refreshStatus()
    }).then((fn) => unlistens.push(fn))

    return () => unlistens.forEach((fn) => fn())
  }, [refreshStatus])

  useEffect(() => {
    refreshStatus()
  }, [refreshStatus])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [status.logs])

  const handleStart = async () => {
    if (selectedModels.length === 0) return

    // 如果服务正在运行，先停止
    if (status.running) {
      try {
        await invoke("stop_server")
        // 等一会在启动，确保端口释放
        await new Promise((r) => setTimeout(r, 500))
      } catch (e) {
        setStatus((prev) => ({
          ...prev,
          logs: [...prev.logs, `[错误] 停止旧服务失败: ${e}`],
        }))
        return
      }
    }

    try {
      await invoke("start_server", {
        config: {
          models: selectedModels,
          address,
          port: parseInt(port) || 10100,
          weight_path: weightPath || null,
          save_dir: null,
          gguf_path: null,
          mmproj_path: null,
        } satisfies LaunchConfig,
      })
    } catch (e) {
      setStatus((prev) => ({
        ...prev,
        logs: [...prev.logs, `[错误] ${e}`],
      }))
    }
  }

  const handleStop = async () => {
    try {
      await invoke("stop_server")
    } catch (e) {
      setStatus((prev) => ({
        ...prev,
        logs: [...prev.logs, `[错误] ${e}`],
      }))
    }
  }

  const handleClearLogs = async () => {
    try {
      await invoke("clear_logs")
      setStatus((prev) => ({ ...prev, logs: [] }))
    } catch {}
  }

  const downloadedModels = models
    .filter((m) => m.downloaded)
    .sort(
      (a, b) =>
        (MODEL_TYPE_ORDER[a.model_type] ?? 99) -
        (MODEL_TYPE_ORDER[b.model_type] ?? 99),
    )

  // 按类型分组已下载模型
  const groupedModels: Record<string, ModelInfo[]> = {}
  for (const m of downloadedModels) {
    const t = m.model_type || "other"
    if (!groupedModels[t]) groupedModels[t] = []
    groupedModels[t].push(m)
  }

  // 服务启动后的信息
  const serverUrl = `http://${address}:${port}`
  const swaggerUrl = `${serverUrl}/swagger-ui/`
  const adminUrl = `${serverUrl}/admin/models/list`

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
            <Server className="w-6 h-6" />
            启动服务
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            配置并启动模型推理服务（支持多模型并行）
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* 配置区 */}
          <Card className="p-4 space-y-4">
            <h3 className="text-sm font-medium">启动配置</h3>

            {/* 多模型选择 */}
            <div className="space-y-2">
              <Label>模型（可多选）</Label>
              <div className="border rounded-md max-h-48 overflow-y-auto p-1 space-y-0.5">
                {downloadedModels.length === 0 ? (
                  <p className="text-xs text-muted-foreground p-2">
                    没有已下载的模型，请先在"模型列表"页面下载模型
                  </p>
                ) : (
                  Object.entries(groupedModels).map(([type, typeModels]) => (
                    <div key={type}>
                      <div className="text-xs text-muted-foreground px-2 py-1 font-medium">
                        {TYPE_LABELS[type] || type}
                      </div>
                      {typeModels.map((m) => {
                        const checked = selectedModels.includes(m.model_id)
                        return (
                          <label
                            key={m.model_id}
                            onClick={() =>
                              setSelectedModels((prev) =>
                                prev.includes(m.model_id)
                                  ? prev.filter((id) => id !== m.model_id)
                                  : [...prev, m.model_id],
                              )
                            }
                            className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm cursor-pointer transition-colors ${
                              checked
                                ? "bg-primary/10 text-primary"
                                : "hover:bg-muted/50"
                            }`}
                          >
                            <div
                              className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors ${
                                checked
                                  ? "bg-primary border-primary text-primary-foreground"
                                  : "border-input"
                              }`}
                            >
                              {checked && <Check className="w-3 h-3" />}
                            </div>
                            <span className="truncate">{m.model_id}</span>
                          </label>
                        )
                      })}
                    </div>
                  ))
                )}
              </div>
              {selectedModels.length > 0 && (
                <p className="text-xs text-muted-foreground">
                  已选 {selectedModels.length} 个模型
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="address">监听地址</Label>
                <Input
                  id="address"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="port">端口</Label>
                <Input
                  id="port"
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="weight-path">
                权重路径{" "}
                <span className="text-muted-foreground font-normal">(可选)</span>
              </Label>
              <Input
                id="weight-path"
                value={weightPath}
                onChange={(e) => setWeightPath(e.target.value)}
                placeholder="留空则使用默认路径"
              />
            </div>
          </Card>

          {/* 状态 + 控制 */}
          <Card className="p-4 flex flex-col">
            <h3 className="text-sm font-medium mb-3">服务状态</h3>
            <div className="flex-1 space-y-2 text-sm mb-4">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2.5 h-2.5 rounded-full inline-block ${
                    status.running ? "bg-green-500" : "bg-gray-300"
                  }`}
                />
                <span>{status.running ? "运行中" : "已停止"}</span>
              </div>
              {status.pid && (
                <div className="text-muted-foreground">PID: {status.pid}</div>
              )}
              {selectedModels.length > 0 && (
                <div className="text-muted-foreground">
                  模型: {selectedModels.join(", ")}
                </div>
              )}
              {selectedModels.length > 1 && (
                <Badge variant="secondary" className="mt-1">
                  多模型并行 ({selectedModels.length} 个)
                </Badge>
              )}
            </div>

            <div className="flex gap-2">
                <Button
                  onClick={handleStart}
                  disabled={selectedModels.length === 0}
                  className="flex-1"
                >
                  {status.running ? (
                    <><RotateCw className="w-4 h-4 mr-1.5" />重启服务</>
                  ) : (
                    <><Play className="w-4 h-4 mr-1.5" />启动服务</>
                  )}
                </Button>
                {status.running && (
                  <Button onClick={handleStop} variant="destructive">
                    <Square className="w-4 h-4" />
                  </Button>
                )}
              </div>

            {/* 服务运行信息 */}
            {status.running && (
              <div className="mt-4 p-3 bg-muted/30 rounded-lg space-y-2 text-sm">
                <h4 className="font-medium text-xs text-muted-foreground uppercase tracking-wider">
                  服务链接
                </h4>

                {/* API Base URL */}
                <div className="flex items-center gap-2">
                  <ExternalLink className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground text-xs shrink-0">
                    API:
                  </span>
                  <code className="text-xs bg-background px-1.5 py-0.5 rounded truncate">
                    {serverUrl}
                  </code>
                </div>

                {/* Swagger UI / OpenAPI */}
                <a
                  href={swaggerUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 text-primary hover:underline"
                >
                  <BookOpen className="w-3.5 h-3.5 shrink-0" />
                  <span className="text-xs whitespace-nowrap">
                    OpenAPI 文档
                  </span>
                  <code className="text-xs bg-background px-1.5 py-0.5 rounded truncate">
                    /swagger-ui/
                  </code>
                </a>

                {/* OpenAPI JSON */}
                <a
                  href={`${serverUrl}/api-docs/openapi.json`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 text-primary hover:underline"
                >
                  <ExternalLink className="w-3.5 h-3.5 shrink-0" />
                  <span className="text-xs whitespace-nowrap">
                    OpenAPI JSON
                  </span>
                  <code className="text-xs bg-background px-1.5 py-0.5 rounded truncate">
                    /api-docs/openapi.json
                  </code>
                </a>

                {/* Admin: loaded models */}
                <a
                  href={adminUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 text-primary hover:underline"
                >
                  <Server className="w-3.5 h-3.5 shrink-0" />
                  <span className="text-xs whitespace-nowrap">
                    已加载模型
                  </span>
                  <code className="text-xs bg-background px-1.5 py-0.5 rounded truncate">
                    /admin/models/list
                  </code>
                </a>

                {/* 提示信息 */}
                <p className="text-xs text-muted-foreground mt-1">
                  接口格式与 OpenAI API 完全兼容，可直接集成
                </p>
              </div>
            )}
          </Card>
        </div>

        {/* 日志区域 */}
        <Card className="flex-1 min-h-0">
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Terminal className="w-4 h-4" />
              日志输出
            </div>
            <Button variant="outline" size="sm" onClick={handleClearLogs}>
              <RotateCw className="w-3 h-3 mr-1" />
              清空
            </Button>
          </div>
          <div className="bg-[#1e1e2e] rounded-b-lg p-4 h-64 overflow-y-auto font-mono text-sm leading-relaxed">
            {status.logs.length === 0 ? (
              <div className="text-gray-500 italic">等待日志输出...</div>
            ) : (
              <div className="space-y-0.5">
                {status.logs.map((line, i) => (
                  <div
                    key={i}
                    className="text-gray-300 whitespace-pre-wrap break-all"
                  >
                    {line}
                  </div>
                ))}
                <div ref={logEndRef} />
              </div>
            )}
          </div>
        </Card>
      </Main>
    </>
  )
}
