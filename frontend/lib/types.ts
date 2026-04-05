export interface Track {
  track_id: number
  bbox: [number, number, number, number]  // x, y, w, h
  status: "AUTHORIZED" | "UNAUTHORIZED" | "UNKNOWN"
  person_name: string | null
  zone: string | null
  confidence: number
  behavior: string | null
}

export interface Alert {
  type: string
  severity: "LOW" | "MEDIUM" | "HIGH"
  track_id: number | null
  message: string
}

export interface WSMessage {
  cam_id: string
  timestamp: string
  frame: string           // base64 JPEG
  tracks: Track[]
  alerts: Alert[]
  zone_counts: Record<string, number>
  heatmap: string | null  // base64 PNG or null
}
