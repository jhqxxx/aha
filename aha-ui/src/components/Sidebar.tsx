import { Package, Play, Settings } from "lucide-react"
import { cn } from "../lib/utils"

export type PageView = "models" | "launch"

interface SidebarProps {
  activePage: PageView
  onNavigate: (page: PageView) => void
  address: string
  port: string
  onAddressChange: (v: string) => void
  onPortChange: (v: string) => void
}

const navItems: { id: PageView; label: string; icon: typeof Package }[] = [
  { id: "models", label: "模型列表", icon: Package },
  { id: "launch", label: "启动服务", icon: Play },
]

export default function Sidebar({
  activePage,
  onNavigate,
  address,
  port,
  onAddressChange,
  onPortChange,
}: SidebarProps) {
  return (
    <aside className="w-64 border-r bg-sidebar-background flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 border-b">
        <h1 className="text-lg font-bold flex items-center gap-2">
          <Play className="w-5 h-5 text-primary" />
          AHA Launcher
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">模型推理服务启动器</p>
      </div>

      {/* Navigation */}
      <nav className="p-2 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                "w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors cursor-pointer",
                activePage === item.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50"
              )}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </button>
          )
        })}
      </nav>

      {/* Settings */}
      <div className="mt-auto p-4 border-t">
        <div className="flex items-center gap-2 text-sm font-medium mb-3 text-muted-foreground">
          <Settings className="w-4 h-4" />
          默认设置
        </div>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">监听端口</label>
            <input
              value={port}
              onChange={(e) => onPortChange(e.target.value)}
              className="w-full h-8 px-2.5 rounded-md border bg-background text-xs
                focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">监听地址</label>
            <input
              value={address}
              onChange={(e) => onAddressChange(e.target.value)}
              className="w-full h-8 px-2.5 rounded-md border bg-background text-xs
                focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
        </div>
      </div>
    </aside>
  )
}
