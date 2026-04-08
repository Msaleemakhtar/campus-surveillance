"use client"

import { useEffect, useState } from "react"
import { useAnalyticsFeed, type IncidentEntry } from "@/hooks/useAnalyticsFeed"
import { HeatmapCanvas, type HeatBlob } from "@/components/HeatmapCanvas"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// Column indices 0–3 map to x-quadrants 0-25%, 25-50%, 50-75%, 75-100% via col/4
const CAM_COL: Record<string, number> = { cam01: 0, cam02: 1, cam03: 2, cam04: 3, cam11: 0, cam15: 1, cam19: 2, cam20: 3 }
const CAM_ROW: Record<string, number> = { cam01: 0, cam02: 0, cam03: 0, cam04: 0, cam11: 1, cam15: 1, cam19: 1, cam20: 1 }

interface Responder { initials: string; name: string; status: string; color: string }

function incidentStyle(severity: string) {
  if (severity === "HIGH")   return { bar: "border-l-rose-500",   text: "text-rose-700",   bg: "bg-rose-50",   badge: "bg-rose-100 text-rose-600"   }
  if (severity === "MEDIUM") return { bar: "border-l-amber-500",  text: "text-amber-700",  bg: "bg-amber-50",  badge: "bg-amber-100 text-amber-600"  }
  return                            { bar: "border-l-indigo-400", text: "text-indigo-700", bg: "bg-indigo-50", badge: "bg-indigo-100 text-indigo-600" }
}

function incidentLabel(type: string) {
  const map: Record<string, string> = {
    UNAUTHORIZED_VISITOR: "Unauthorized",
    AFTER_HOURS_PRESENCE: "After Hours",
    LOITERING:            "Loitering",
    RUNNING:              "Running",
  }
  return map[type] ?? type
}

const RESPONDER_STATUS: Record<string, { dot: string; color: string; label: string }> = {
  ON_DUTY:    { dot: "●", color: "text-emerald-600", label: "On Duty"    },
  RESPONDING: { dot: "◆", color: "text-amber-600",   label: "Responding" },
  ON_CALL:    { dot: "○", color: "text-slate-400",   label: "On Call"    },
}

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString("en-GB", {
      hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC",
    }) + " UTC"
  } catch { return iso }
}

export function AnalyticsDashboard() {
  const { summary, incidents, connected } = useAnalyticsFeed()
  const [responders, setResponders] = useState<Responder[]>([])

  useEffect(() => {
    fetch(`${API}/analytics/responders`)
      .then((r) => r.json())
      .then((d) => setResponders(d.responders ?? []))
      .catch(() => {})
  }, [])

  const heatBlobs: HeatBlob[] = Object.entries(summary.camera_tracks).flatMap(
    ([camId, tracks]) => {
      const col = CAM_COL[camId] ?? 0
      const row = CAM_ROW[camId] ?? 0
      return tracks.map((t) => ({
        x: ((col / 4) + (t.cx / 640) / 4) * 100,
        y: ((row / 2) + (t.cy / 480) / 2) * 100,
      }))
    }
  )

  const hasHighAlert = incidents.some(
    (i) => i.severity === "HIGH" && (i.cam_id === "cam19" || i.cam_id === "cam20")
  )
  const alertZones =
    (summary.camera_tracks["cam19"]?.length ?? 0) > 0 ||
    (summary.camera_tracks["cam20"]?.length ?? 0) > 0

  const authPct = summary.visitor_count > 0
    ? Math.round((summary.authorized / summary.visitor_count) * 100)
    : 0

  return (
    <div className="flex-1 grid grid-cols-[1fr_248px] gap-3 min-h-0">

      {/* ══ LEFT: heatmap ═══════════════════════════════════════════════════ */}
      <div className="flex flex-col gap-3 min-h-0">

        {/* Section label */}
        <div className="flex items-center justify-between shrink-0">
          <span className="text-[11px] text-slate-700 font-bold tracking-widest uppercase">
            Foot Traffic · All 8 Cameras
          </span>
          <span className={`text-[11px] flex items-center gap-1.5 font-medium ${connected ? "text-emerald-600" : "text-slate-400"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-500" : "bg-slate-300"}`} />
            {connected ? "Live" : "Connecting…"}
          </span>
        </div>

        {/* Heatmap — keeps its own dark bg so inferno colors pop */}
        <div
          className="rounded-xl border border-slate-200 shadow-sm relative flex-1 overflow-hidden"
          style={{ background: "#0b1424", minHeight: 240 }}
        >
          {/* Quadrant dividers */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-0 bottom-8 left-1/2 border-l border-dashed border-white/10" />
            <div className="absolute left-0 right-0 top-1/2 border-t border-dashed border-white/10" />
          </div>

          {/* Zone labels */}
          <div className="absolute top-2.5 left-3 text-[10px] text-white/30">cam01 · cam02</div>
          <div className="absolute top-2.5 right-3 text-[10px] text-white/30">cam03 · cam04</div>
          <div className="absolute bottom-10 left-3 text-[10px] text-white/30">cam11 · cam15</div>
          <div className={`absolute bottom-10 right-3 text-[10px] font-medium ${alertZones ? "text-rose-400" : "text-white/30"}`}>
            cam19 · cam20{alertZones ? " ⚠" : ""}
          </div>

          <HeatmapCanvas blobs={heatBlobs} className="absolute inset-0" />

          {/* Legend */}
          <div className="absolute bottom-0 left-0 right-0 flex items-center gap-2.5 px-3 py-2 text-[10px]"
            style={{ background: "rgba(6,12,22,0.8)", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <span className="text-white/40">Density</span>
            <div className="w-20 h-1.5 rounded-full" style={{ background: "linear-gradient(to right,#312e81,#7c3aed,#dc2626,#f97316,#eab308)" }} />
            <span className="text-white/40">Low → High</span>
            <span className="ml-auto text-white/25">8 cameras · live</span>
          </div>
        </div>

        {/* High-alert banner */}
        {hasHighAlert && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 flex items-center gap-3 px-4 py-2.5 shrink-0">
            <span className="text-rose-500">⚠</span>
            <span className="text-xs text-rose-700 font-medium">
              High activity in cam19 (THREAT) and cam20 (INTRUSION) zones
            </span>
          </div>
        )}
      </div>

      {/* ══ RIGHT: sidebar ══════════════════════════════════════════════════ */}
      <div className="flex flex-col gap-3 min-h-0 overflow-y-auto" style={{ scrollbarWidth: "thin", scrollbarColor: "#e2e8f0 transparent" }}>

        {/* Section label — mirrors left column header for vertical alignment */}
        <div className="flex items-center justify-between shrink-0">
          <span className="text-[11px] text-slate-700 font-bold tracking-widest uppercase">
            Overview · Live Stats
          </span>
          <span className="text-[11px] text-slate-700 font-bold">
            {summary.cameras_active} / 8 active
          </span>
        </div>

        {/* KPI row */}
        <div className="grid grid-cols-2 gap-3 shrink-0">
          {/* Visitors */}
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-3.5">
            <p className="text-[11px] text-slate-700 uppercase tracking-widest font-bold mb-1.5">Visitors</p>
            <span className="text-3xl font-bold text-slate-800 leading-none">{summary.visitor_count}</span>
          </div>
          {/* Badge */}
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-3.5">
            <p className="text-[11px] text-slate-700 uppercase tracking-widest font-bold mb-2">Badge</p>
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
              <span className="text-xs font-semibold text-emerald-700">{summary.authorized}</span>
              <span className="text-[9px] text-slate-400">Authorized</span>
            </div>
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0" />
              <span className="text-xs font-semibold text-rose-600">{summary.unknown}</span>
              <span className="text-[9px] text-slate-400">Unknown</span>
            </div>
            {summary.visitor_count > 0 && (
              <div className="w-full h-1 rounded-full bg-slate-100 overflow-hidden">
                <div className="h-full rounded-full bg-emerald-500 transition-all duration-500" style={{ width: `${authPct}%` }} />
              </div>
            )}
          </div>
        </div>

        {/* Incident timeline */}
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-3.5 flex flex-col min-h-0 flex-1" style={{ minHeight: 80, maxHeight: 220 }}>
          <div className="flex items-center justify-between mb-3 shrink-0">
            <p className="text-[11px] text-slate-700 uppercase tracking-widest font-bold">Incidents</p>
            {incidents.length > 0 && (
              <span className="text-[9px] font-semibold bg-rose-100 text-rose-600 rounded-full px-2 py-0.5">
                {incidents.length}
              </span>
            )}
          </div>

          <div
            className="flex-1 overflow-y-auto space-y-1.5 pr-0.5"
            style={{ scrollbarWidth: "thin", scrollbarColor: "#cbd5e1 transparent" }}
          >
            {incidents.length === 0 ? (
              <p className="text-[11px] text-slate-400 italic pt-1">No incidents recorded.</p>
            ) : incidents.map((inc: IncidentEntry) => {
              const s = incidentStyle(inc.severity)
              return (
                <div key={inc.id} className={`border-l-2 pl-2.5 py-1.5 rounded-r ${s.bar} ${s.bg}`}>
                  <div className="text-[9px] text-slate-400 mb-0.5">{formatTime(inc.timestamp)}</div>
                  <div className={`text-[11px] font-semibold ${s.text}`}>{incidentLabel(inc.type)}</div>
                  <div className="text-[10px] text-slate-500">
                    {inc.cam_id}{inc.track_id != null ? ` · #${inc.track_id}` : ""}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Response team */}
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-3.5 shrink-0">
          <p className="text-[11px] text-slate-700 uppercase tracking-widest font-bold mb-3">Response Team</p>
          <div className="space-y-2.5">
            {responders.map((r) => {
              const s = RESPONDER_STATUS[r.status] ?? RESPONDER_STATUS.ON_CALL
              // Ensure avatar is always visible on white background
              const avatarBg = r.color === "#1e293b" ? "#475569" : r.color
              return (
                <div key={r.initials} className="flex items-center gap-2.5">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white shrink-0 ring-2 ring-white shadow-sm"
                    style={{ background: avatarBg }}
                  >
                    {r.initials}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-slate-700 font-medium truncate">{r.name}</div>
                    <div className={`text-[10px] ${s.color} flex items-center gap-1`}>
                      <span className="text-[8px]">{s.dot}</span>
                      <span>{s.label}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

      </div>
    </div>
  )
}
