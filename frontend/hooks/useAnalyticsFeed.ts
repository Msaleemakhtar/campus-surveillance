"use client"

import { useEffect, useRef, useState } from "react"
import { CameraWebSocket } from "@/lib/websocket"
import type { WSMessage } from "@/lib/types"

const CAMERAS = ["cam01", "cam02", "cam03", "cam04", "cam11", "cam15", "cam19", "cam20"]
const MAX_INCIDENTS = 50
const DEDUP_MS = 60_000

const SS_INCIDENTS = "analytics:incidents"
const SS_LAST_FIRED = "analytics:lastFired"

export interface TrackPoint {
  track_id: number
  cx: number
  cy: number
  status: string
}

export interface IncidentEntry {
  id: string
  timestamp: string
  cam_id: string
  type: string
  severity: string
  track_id: number | null
  message: string
}

export interface AnalyticsSummary {
  visitor_count: number
  authorized: number
  unknown: number
  cameras_active: number
  camera_tracks: Record<string, TrackPoint[]>
}

function ssRead<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch { return fallback }
}

function ssWrite(key: string, value: unknown) {
  try { sessionStorage.setItem(key, JSON.stringify(value)) } catch {}
}

export function useAnalyticsFeed() {
  const [cameraMessages, setCameraMessages] = useState<Record<string, WSMessage>>({})

  // Start with empty state for SSR/hydration parity; restore from sessionStorage after mount
  const [incidents, setIncidents] = useState<IncidentEntry[]>([])
  const lastFired = useRef<Record<string, number>>({})

  useEffect(() => {
    setIncidents(ssRead<IncidentEntry[]>(SS_INCIDENTS, []))
    const saved = ssRead<Record<string, number>>(SS_LAST_FIRED, {})
    const now = Date.now()
    const filtered: Record<string, number> = {}
    for (const [k, t] of Object.entries(saved)) {
      if (now - t < DEDUP_MS) filtered[k] = t
    }
    lastFired.current = filtered
  }, [])

  const lastFrameIdx = useRef<Record<string, number>>({})

  // Persist incidents + lastFired whenever incidents list changes
  useEffect(() => {
    ssWrite(SS_INCIDENTS, incidents)
    ssWrite(SS_LAST_FIRED, lastFired.current)
  }, [incidents])

  useEffect(() => {
    const sockets = CAMERAS.map((camId) => {
      const ws = new CameraWebSocket(camId, (msg: WSMessage) => {
        setCameraMessages((prev) => ({ ...prev, [camId]: msg }))

        // Only process alerts on a new YOLO result
        const prevIdx = lastFrameIdx.current[camId] ?? -1
        if (msg.frame_idx <= prevIdx || msg.alerts.length === 0) return
        lastFrameIdx.current[camId] = msg.frame_idx

        const now = Date.now()
        const newEntries: IncidentEntry[] = []
        for (const a of msg.alerts) {
          const dedupKey = `${camId}-${a.type}-${a.track_id ?? "x"}`
          const lastTime = lastFired.current[dedupKey] ?? 0
          if (now - lastTime < DEDUP_MS) continue
          lastFired.current[dedupKey] = now
          newEntries.push({
            id: `${dedupKey}-${now}`,
            timestamp: new Date().toISOString(),
            cam_id: camId,
            type: a.type,
            severity: a.severity,
            track_id: a.track_id ?? null,
            message: a.message,
          })
        }

        if (newEntries.length === 0) return
        setIncidents((prev) => [...newEntries, ...prev].slice(0, MAX_INCIDENTS))
      })
      ws.connect()
      return ws
    })

    return () => { sockets.forEach((ws) => ws.disconnect()) }
  }, [])

  const summary: AnalyticsSummary = (() => {
    let visitor_count = 0, authorized = 0, unknown = 0
    const camera_tracks: Record<string, TrackPoint[]> = {}

    for (const [camId, msg] of Object.entries(cameraMessages)) {
      const points: TrackPoint[] = []
      for (const t of msg.tracks) {
        visitor_count++
        if (t.status === "AUTHORIZED") authorized++
        else unknown++
        const [x, y, w, h] = t.bbox
        points.push({ track_id: t.track_id, cx: x + w / 2, cy: y + h / 2, status: t.status })
      }
      camera_tracks[camId] = points
    }

    return { visitor_count, authorized, unknown, cameras_active: Object.keys(cameraMessages).length, camera_tracks }
  })()

  return { summary, incidents, connected: Object.keys(cameraMessages).length > 0 }
}
