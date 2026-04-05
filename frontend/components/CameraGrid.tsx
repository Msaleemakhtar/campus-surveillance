"use client"

import { useState } from "react"
import { CameraFeed } from "@/components/CameraFeed"

// 10-camera setup — best coverage across indoor, outdoor, and anomaly feeds
const CAMERAS: { camId: string; label: string }[] = [
  { camId: "cam01", label: "Main Entrance" },
  { camId: "cam02", label: "Hallway A" },
  { camId: "cam03", label: "Hallway B" },
  { camId: "cam04", label: "Library" },
  { camId: "cam05", label: "Cafeteria" },
  { camId: "cam11", label: "Main Gate" },
  { camId: "cam12", label: "Parking A" },
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
      {/* 5-column grid — 2 rows = 10 cameras */}
      <div className="grid grid-cols-5 gap-2">
        {CAMERAS.map(({ camId, label }) => (
          <button
            key={camId}
            onClick={() => setExpanded({ camId, label })}
            className="block w-full text-left focus:outline-none focus:ring-2 focus:ring-blue-500 rounded overflow-hidden"
            aria-label={`Expand ${camId} — ${label}`}
          >
            <CameraFeed camId={camId} label={label} />
          </button>
        ))}
      </div>

      {/* Expanded modal — full-screen overlay */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setExpanded(null)}
        >
          <div
            className="relative w-full max-w-4xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={() => setExpanded(null)}
              className="absolute -top-8 right-0 text-gray-300 hover:text-white text-sm"
            >
              ✕ Close
            </button>

            {/* Full-size feed — second WS connection for the expanded view */}
            <CameraFeed camId={expanded.camId} label={expanded.label} />

            <p className="text-gray-400 text-xs mt-2 text-center">
              Click outside or press ✕ to close
            </p>
          </div>
        </div>
      )}
    </>
  )
}
