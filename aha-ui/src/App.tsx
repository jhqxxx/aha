import { useState } from "react"
import Sidebar, { type PageView } from "./components/Sidebar"
import ModelsPage from "./pages/ModelsPage"
import LaunchPage from "./pages/LaunchPage"
import "./index.css"

function App() {
  const [activePage, setActivePage] = useState<PageView>("models")
  const [address, setAddress] = useState("127.0.0.1")
  const [port, setPort] = useState("10100")

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        address={address}
        port={port}
        onAddressChange={setAddress}
        onPortChange={setPort}
      />
      <main className="flex-1 overflow-y-auto">
        {activePage === "models" ? (
          <ModelsPage />
        ) : (
          <LaunchPage key={`${address}:${port}`} />
        )}
      </main>
    </div>
  )
}

export default App
