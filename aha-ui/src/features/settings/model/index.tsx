import { useState, useEffect } from "react"
import { invoke } from "@tauri-apps/api/core"
import { FolderOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ContentSection } from "../components/content-section"

const SAVE_DIR_KEY = "aha-model-save-dir"

export function SettingsModel() {
  const [saveDir, setSaveDir] = useState("")
  const [defaultDir, setDefaultDir] = useState("")
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    invoke<string>("get_default_save_dir")
      .then(setDefaultDir)
      .catch(() => {})

    const stored = localStorage.getItem(SAVE_DIR_KEY)
    if (stored) setSaveDir(stored)
  }, [])

  const handleSave = () => {
    if (saveDir) {
      localStorage.setItem(SAVE_DIR_KEY, saveDir)
    } else {
      localStorage.removeItem(SAVE_DIR_KEY)
    }
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    setSaveDir("")
    localStorage.removeItem(SAVE_DIR_KEY)
  }

  return (
    <ContentSection
      title='Model'
      desc='配置模型下载位置和相关设置'
    >
      <div className='space-y-6'>
        <div className='space-y-2'>
          <Label htmlFor='save-dir' className='flex items-center gap-1.5'>
            <FolderOpen className='w-4 h-4' />
            模型下载路径
          </Label>
          <Input
            id='save-dir'
            value={saveDir}
            onChange={(e) => setSaveDir(e.target.value)}
            placeholder={defaultDir || "~/.aha/"}
          />
          <p className='text-xs text-muted-foreground'>
            留空则使用默认路径：{defaultDir || "~/.aha/"}
            。模型将下载到该目录下的 {`{model_id}`} 子文件夹中。
          </p>
        </div>

        <div className='flex gap-2'>
          <Button onClick={handleSave}>
            {saved ? "已保存" : "保存"}
          </Button>
          <Button variant='outline' onClick={handleReset}>
            恢复默认
          </Button>
        </div>
      </div>
    </ContentSection>
  )
}
