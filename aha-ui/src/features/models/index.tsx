import { useState, useEffect } from "react"
import { invoke } from "@tauri-apps/api/core"
import { Download, Trash2, RefreshCw, HardDrive, Package } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Header } from "@/components/layout/header"
import { Main } from "@/components/layout/main"
import { ProfileDropdown } from "@/components/profile-dropdown"
import { ThemeSwitch } from "@/components/theme-switch"

const SAVE_DIR_KEY = "aha-model-save-dir"

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
  llm: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  vlm: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  ocr: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  asr: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  tts: "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200",
  image: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  embedding: "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200",
  reranker: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200",
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

export function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadModels = async () => {
    setLoading(true)
    try {
      const data = await invoke<ModelInfo[]>("list_models")
      setModels(data)
      setError(null)
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
      const saveDir = localStorage.getItem(SAVE_DIR_KEY) || null
      await invoke("download_model", { modelId, saveDir })
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
    <>
      <Header>
        <div className="flex items-center gap-2 ms-auto">
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <Package className="w-6 h-6" />
              模型列表
            </h1>
            <p className="text-muted-foreground text-sm mt-1">
              管理已下载和可用的模型
            </p>
          </div>
          <Button variant="outline" onClick={loadModels} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-1.5 ${loading ? "animate-spin" : ""}`} />
            刷新
          </Button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
          </div>
        )}

        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-muted/50 border-b">
                  <th className="text-left px-4 py-3 font-medium">模型 ID</th>
                  <th className="text-left px-4 py-3 font-medium">类型</th>
                  <th className="text-left px-4 py-3 font-medium">大小</th>
                  <th className="text-left px-4 py-3 font-medium">状态</th>
                  <th className="text-right px-4 py-3 font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {models.length === 0 && !loading ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                      暂无模型数据
                    </td>
                  </tr>
                ) : (
                  models.map((m) => (
                    <tr key={m.model_id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-4 py-3">
                        <div className="font-medium">{m.model_id}</div>
                        {m.path && (
                          <div className="text-xs text-muted-foreground mt-0.5 truncate max-w-[400px]">
                            {m.path}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <Badge
                          variant="secondary"
                          className={typeColors[m.model_type] || ""}
                        >
                          {typeLabels[m.model_type] || m.model_type}
                        </Badge>
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
                          <Badge variant="outline" className="text-green-600 border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-500 mr-1.5 inline-block" />
                            已下载
                          </Badge>
                        ) : (
                          <span className="text-muted-foreground text-sm">未下载</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {!m.downloaded ? (
                            <Button
                              size="sm"
                              onClick={() => handleDownload(m.model_id)}
                              disabled={downloading === m.model_id}
                            >
                              <Download className="w-3.5 h-3.5 mr-1" />
                              {downloading === m.model_id ? "下载中..." : "下载"}
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={() => handleDelete(m.model_id)}
                            >
                              <Trash2 className="w-3.5 h-3.5 mr-1" />
                              删除
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </Main>
    </>
  )
}
