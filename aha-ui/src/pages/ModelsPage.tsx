import { useState, useEffect } from "react"
import { invoke } from "@tauri-apps/api/core"
import { Download, Trash2, RefreshCw, FolderOpen, HardDrive, Package } from "lucide-react"
import { cn } from "../lib/utils"

interface ModelInfo {
  model_id: string
  owner: string
  model_type: string
  downloaded: boolean
  size: number | null
  size_human: string | null
  path: string | null
}

const typeColors: Record<string, string> = {
  llm: "bg-blue-100 text-blue-800",
  vlm: "bg-purple-100 text-purple-800",
  ocr: "bg-green-100 text-green-800",
  asr: "bg-yellow-100 text-yellow-800",
  tts: "bg-pink-100 text-pink-800",
  image: "bg-orange-100 text-orange-800",
  embedding: "bg-teal-100 text-teal-800",
  reranker: "bg-indigo-100 text-indigo-800",
}

const typeLabels: Record<string, string> = {
  llm: "LLM",
  vlm: "VLM",
  ocr: "OCR",
  asr: "ASR",
  tts: "TTS",
  image: "图像",
  embedding: "嵌入",
  reranker: "重排序",
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadModels = async () => {
    setLoading(true)
    try {
      const data = await invoke<ModelInfo[]>("list_models")
      setModels(data)
    } catch (e) {
      setError(String(e))
    }
    setLoading(false)
  }

  useEffect(() => {
    loadModels()
  }, [])

  const handleDownload = async (modelId: string) => {
    setDownloading(modelId)
    setError(null)
    try {
      await invoke("download_model", { modelId })
      await loadModels()
    } catch (e) {
      setError(String(e))
    }
    setDownloading(null)
  }

  const handleDelete = async (modelId: string) => {
    if (!confirm(`确定删除模型 ${modelId}？`)) return
    try {
      await invoke("delete_model", { modelId })
      await loadModels()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Package className="w-6 h-6" />
          模型列表
        </h1>
        <button
          onClick={loadModels}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border
            hover:bg-secondary transition-colors cursor-pointer"
        >
          <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          刷新
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="border rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-secondary/50 border-b">
                <th className="text-left px-4 py-3 font-medium">模型 ID</th>
                <th className="text-left px-4 py-3 font-medium">类型</th>
                <th className="text-left px-4 py-3 font-medium">大小</th>
                <th className="text-left px-4 py-3 font-medium">状态</th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model_id} className="border-b last:border-0 hover:bg-secondary/30">
                  <td className="px-4 py-3">
                    <div className="font-medium">{m.model_id}</div>
                    {m.path && (
                      <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1">
                        <FolderOpen className="w-3 h-3 flex-shrink-0" />
                        <span className="truncate max-w-[400px]">{m.path}</span>
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium",
                      typeColors[m.model_type] || "bg-gray-100 text-gray-800"
                    )}>
                      {typeLabels[m.model_type] || m.model_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {m.size_human ? (
                      <span className="flex items-center gap-1">
                        <HardDrive className="w-3.5 h-3.5" />
                        {m.size_human}
                      </span>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {m.downloaded ? (
                      <span className="inline-flex items-center gap-1 text-green-700">
                        <span className="w-2 h-2 rounded-full bg-green-500" />
                        已下载
                      </span>
                    ) : (
                      <span className="text-muted-foreground">未下载</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {!m.downloaded ? (
                        <button
                          onClick={() => handleDownload(m.model_id)}
                          disabled={downloading === m.model_id}
                          className="inline-flex items-center gap-1 px-3 py-1.5 text-xs rounded-md
                            bg-primary text-primary-foreground hover:opacity-90
                            disabled:opacity-50 transition-opacity cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5" />
                          {downloading === m.model_id ? "下载中..." : "下载"}
                        </button>
                      ) : (
                        <button
                          onClick={() => handleDelete(m.model_id)}
                          className="inline-flex items-center gap-1 px-3 py-1.5 text-xs rounded-md
                            border border-red-200 text-red-600 hover:bg-red-50
                            transition-colors cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          删除
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
