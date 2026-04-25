import { useState, useEffect, useRef, useCallback } from "react"
import { invoke } from "@tauri-apps/api/core"
import { listen } from "@tauri-apps/api/event"
import { Play, Square, Terminal, RotateCw, Server } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
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
  model_id: string
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

export function LaunchPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState("")
  const [address, setAddress] = useState("127.0.0.1")
  const [port, setPort] = useState("10100")
  const [weightPath, setWeightPath] = useState("")
  const [status, setStatus] = useState<ServerStatus>({ running: false, pid: null, logs: [] })
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
    if (!selectedModel) return
    try {
      await invoke("start_server", {
        config: {
          model_id: selectedModel,
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

  const downloadedModels = models.filter((m) => m.downloaded)

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
            配置并启动模型推理服务
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* 配置区 */}
          <Card className="p-4 space-y-4">
            <h3 className="text-sm font-medium">启动配置</h3>

            <div className="space-y-2">
              <Label htmlFor="model-select">模型</Label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                disabled={status.running}
              >
                <option value="">选择模型...</option>
                {downloadedModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.model_id}
                  </option>
                ))}
              </select>
              {downloadedModels.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  没有已下载的模型，请先在"模型列表"页面下载模型
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
                  disabled={status.running}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="port">端口</Label>
                <Input
                  id="port"
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                  disabled={status.running}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="weight-path">
                权重路径 <span className="text-muted-foreground font-normal">(可选)</span>
              </Label>
              <Input
                id="weight-path"
                value={weightPath}
                onChange={(e) => setWeightPath(e.target.value)}
                placeholder="留空则使用默认路径"
                disabled={status.running}
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
              {selectedModel && (
                <div className="text-muted-foreground">模型: {selectedModel}</div>
              )}
            </div>

            <div className="flex gap-2">
              {!status.running ? (
                <Button
                  onClick={handleStart}
                  disabled={!selectedModel}
                  className="flex-1"
                >
                  <Play className="w-4 h-4 mr-1.5" />
                  启动服务
                </Button>
              ) : (
                <Button
                  onClick={handleStop}
                  variant="destructive"
                  className="flex-1"
                >
                  <Square className="w-4 h-4 mr-1.5" />
                  停止服务
                </Button>
              )}
            </div>
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
                  <div key={i} className="text-gray-300 whitespace-pre-wrap break-all">
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
