"use client"

import { useState } from "react"
import { CameraFeed } from "@/components/CameraFeed"

const CAMERAS: { camId: string; label: string }[] = [
  { camId: "cam01", label: "Main Entrance" },
  { camId: "cam02", label: "Hallway A" },
  { camId: "cam03", label: "Hallway B" },
  { camId: "cam04", label: "Library" },
  { camId: "cam11", label: "Main Gate" },
  { camId: "cam15", label: "Sports Ground" },
  { camId: "cam19", label: "THREAT FEED" },
  { camId: "cam20", label: "INTRUSION FEED" },
]

interface ExpandedCamera {
  camId: string
  label: string
}

export function CameraGrid() {
  const [expanded, setExpanded] = useState<ExpandedCamera | null>(null)

  return (
    <>
      {/* 4×2 grid — each cell is 25vw wide, height maintains 16:9 via aspect-video */}
      <div className="w-full grid grid-cols-4 gap-1">
        {CAMERAS.map(({ camId, label }) => (
          <button
            key={camId}
            onClick={() => setExpanded({ camId, label })}
            className="block w-full aspect-video text-left focus:outline-none focus:ring-2 focus:ring-blue-500 rounded overflow-hidden"
            aria-label={`Expand ${camId} — ${label}`}
          >
            <CameraFeed camId={camId} label={label} />
          </button>
        ))}
      </div>

      {expanded && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setExpanded(null)}
        >
          <div
            className="relative w-full max-w-5xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setExpanded(null)}
              className="absolute -top-8 right-0 text-gray-300 hover:text-white text-sm z-10"
            >
              ✕ Close
            </button>
            <div className="w-full aspect-video rounded overflow-hidden">
              <CameraFeed camId={expanded.camId} label={expanded.label} />
            </div>
            <p className="text-gray-400 text-xs mt-2 text-center">
              Click outside or press ✕ to close
            </p>
          </div>
        </div>
      )}
    </>
  )
}
