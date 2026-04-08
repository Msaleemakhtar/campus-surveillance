import Link from "next/link"
import { CameraGrid } from "@/components/CameraGrid"

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 flex flex-col p-2 gap-2">
      <div className="flex items-center justify-between shrink-0 px-1">
        <h1 className="text-white text-sm font-semibold tracking-wide">
          Campus Surveillance
        </h1>
        <nav className="flex items-center gap-4">
          <span className="text-blue-400 text-xs border-b border-blue-400 pb-0.5">Live Feed</span>
          <Link href="/analytics" className="text-gray-500 text-xs hover:text-gray-300 transition-colors">
            Analytics →
          </Link>
          <Link href="/search" className="text-gray-500 text-xs hover:text-gray-300 transition-colors">
            Search →
          </Link>
        </nav>
      </div>

      <CameraGrid />
    </main>
  )
}
