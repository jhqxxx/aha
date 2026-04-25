import { useState, useEffect, useRef, useCallback } from "react"
import { invoke } from "@tauri-apps/api/core"
import { listen } from "@tauri-apps/api/event"
import { Play, Square, Terminal, RotateCw, Server } from "lucide-react"
import { cn } from "../lib/utils"

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

export default function LaunchPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState("")
  const [address, setAddress] = useState("127.0.0.1")
  const [port, setPort] = useState("10100")
  const [weightPath, setWeightPath] = useState("")
  const [status, setStatus] = useState<ServerStatus>({ running: false, pid: null, logs: [] })
  const logEndRef = useRef<HTMLDivElement>(null)

  // 加载模型列表
  useEffect(() => {
    invoke<ModelInfo[]>("list_models").then(setModels).catch(() => {})
  }, [])

  // 获取服务状态
  const refreshStatus = useCallback(async () => {
    try {
      const s = await invoke<ServerStatus>("get_server_status")
      setStatus(s)
    } catch {}
  }, [])

  // 监听日志和状态事件
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

  // 初始加载状态
  useEffect(() => {
    refreshStatus()
  }, [refreshStatus])

  // 自动滚动到底部
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
    <div className="p-6 flex flex-col h-full">
      <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Server className="w-6 h-6" />
        启动服务
      </h1>

      {/* 配置区 */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="space-y-4">
          {/* 模型选择 */}
          <div>
            <label className="block text-sm font-medium mb-1">模型</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full h-9 px-3 rounded-md border bg-background text-sm
                focus:outline-none focus:ring-2 focus:ring-ring"
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
              <p className="text-xs text-muted-foreground mt-1">
                没有已下载的模型，请先在"模型列表"页面下载模型
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium mb-1">监听地址</label>
              <input
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                className="w-full h-9 px-3 rounded-md border bg-background text-sm
                  focus:outline-none focus:ring-2 focus:ring-ring"
                disabled={status.running}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">端口</label>
              <input
                value={port}
                onChange={(e) => setPort(e.target.value)}
                className="w-full h-9 px-3 rounded-md border bg-background text-sm
                  focus:outline-none focus:ring-2 focus:ring-ring"
                disabled={status.running}
              />
            </div>
          </div>

          {/* 权重路径 */}
          <div>
            <label className="block text-sm font-medium mb-1">
              权重路径 <span className="text-muted-foreground font-normal">(可选，默认自动)</span>
            </label>
            <input
              value={weightPath}
              onChange={(e) => setWeightPath(e.target.value)}
              placeholder="留空则使用默认路径"
              className="w-full h-9 px-3 rounded-md border bg-background text-sm
                focus:outline-none focus:ring-2 focus:ring-ring"
              disabled={status.running}
            />
          </div>
        </div>

        {/* 状态 + 控制 */}
        <div className="flex flex-col">
          <div className="flex-1 p-4 rounded-lg border bg-secondary/20">
            <div className="text-sm font-medium mb-3">服务状态</div>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className={cn(
                  "w-2.5 h-2.5 rounded-full",
                  status.running ? "bg-green-500" : "bg-gray-300"
                )} />
                <span>{status.running ? "运行中" : "已停止"}</span>
              </div>
              {status.pid && (
                <div className="text-muted-foreground">PID: {status.pid}</div>
              )}
              {selectedModel && (
                <div className="text-muted-foreground">
                  模型: {selectedModel}
                </div>
              )}
            </div>
          </div>

          <div className="flex gap-2 mt-4">
            {!status.running ? (
              <button
                onClick={handleStart}
                disabled={!selectedModel}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2
                  rounded-md bg-primary text-primary-foreground text-sm font-medium
                  hover:opacity-90 disabled:opacity-50 transition-opacity cursor-pointer"
              >
                <Play className="w-4 h-4" />
                启动服务
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2
                  rounded-md bg-destructive text-destructive-foreground text-sm font-medium
                  hover:opacity-90 transition-opacity cursor-pointer"
              >
                <Square className="w-4 h-4" />
                停止服务
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 日志区域 */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Terminal className="w-4 h-4" />
            日志输出
          </div>
          <button
            onClick={handleClearLogs}
            className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md
              border hover:bg-secondary transition-colors cursor-pointer"
          >
            <RotateCw className="w-3 h-3" />
            清空
          </button>
        </div>

        <div className="flex-1 bg-[#1e1e2e] rounded-lg p-4 overflow-y-auto log-area">
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
      </div>
    </div>
  )
}
