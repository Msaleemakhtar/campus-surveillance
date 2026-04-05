import { CameraGrid } from "@/components/CameraGrid"

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-white text-lg font-semibold tracking-wide">
          Campus Surveillance — Live Feed
        </h1>
        <span className="text-gray-500 text-xs">
          20 cameras · AI inference via Colab T4
        </span>
      </div>

      <CameraGrid />
    </main>
  )
}
