"use client"

import { useCameraFeed } from "@/hooks/useCameraFeed"
import type { Alert } from "@/lib/types"

interface Props {
  camId: string
  label: string
}

const SEVERITY_COLOR: Record<Alert["severity"], string> = {
  HIGH: "bg-red-600",
  MEDIUM: "bg-yellow-500",
  LOW: "bg-blue-500",
}

export function CameraFeed({ camId, label }: Props) {
  const { canvasRef, lastMessage, connected } = useCameraFeed(camId)

  const latestAlert = lastMessage?.alerts?.[0] ?? null

  return (
    <div className="relative bg-black w-full h-full overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
      />

      {!connected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70">
          <span className="text-gray-400 text-xs">Connecting…</span>
        </div>
      )}

      <div className="absolute top-1 left-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded">
        {camId.toUpperCase()} · {label}
      </div>

      <div
        className={`absolute top-1 right-1 w-2 h-2 rounded-full ${
          connected ? "bg-green-400" : "bg-red-500"
        }`}
      />

      {latestAlert && (
        <div
          className={`absolute bottom-0 left-0 right-0 ${SEVERITY_COLOR[latestAlert.severity]} text-white text-[10px] px-2 py-0.5 truncate`}
        >
          {latestAlert.message}
        </div>
      )}
    </div>
  )
}
