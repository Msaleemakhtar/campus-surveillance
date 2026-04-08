"use client"

import { useEffect, useRef } from "react"

// ── Inferno colormap (matplotlib) — 9 key stops, interpolated to 256 entries ──
// Maps intensity 0→1 to RGB: black → violet → red → orange → yellow-white
const INFERNO_STOPS: [number, number, number, number][] = [
  [0.00,   0,   0,   4],
  [0.13,  31,  12,  72],
  [0.26,  85,  15, 109],
  [0.38, 136,  34,  85],
  [0.50, 186,  54,  42],
  [0.63, 227, 105,  19],
  [0.75, 249, 167,  20],
  [0.88, 246, 219, 101],
  [1.00, 252, 255, 164],
]

function buildColormap(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    // Find surrounding stops
    let lo = INFERNO_STOPS[0], hi = INFERNO_STOPS[INFERNO_STOPS.length - 1]
    for (let s = 0; s < INFERNO_STOPS.length - 1; s++) {
      if (t >= INFERNO_STOPS[s][0] && t <= INFERNO_STOPS[s + 1][0]) {
        lo = INFERNO_STOPS[s]
        hi = INFERNO_STOPS[s + 1]
        break
      }
    }
    const span = hi[0] - lo[0]
    const f = span > 0 ? (t - lo[0]) / span : 0
    lut[i * 3]     = Math.round(lo[1] + f * (hi[1] - lo[1]))
    lut[i * 3 + 1] = Math.round(lo[2] + f * (hi[2] - lo[2]))
    lut[i * 3 + 2] = Math.round(lo[3] + f * (hi[3] - lo[3]))
  }
  return lut
}

const COLORMAP = buildColormap()

// ── Pre-built radial gradient stamp for each kernel size ──
// We stamp one per point; additive blending accumulates intensity.
function makeStamp(radius: number): ImageData {
  const size = radius * 2
  const stamp = new ImageData(size, size)
  const cx = radius, cy = radius
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
      // Gaussian falloff: e^(-3d²) clipped to 0 outside radius
      const v = d <= 1 ? Math.exp(-3 * d * d) : 0
      const idx = (y * size + x) * 4
      stamp.data[idx]     = 255
      stamp.data[idx + 1] = 255
      stamp.data[idx + 2] = 255
      stamp.data[idx + 3] = Math.round(v * 255)
    }
  }
  return stamp
}

export interface HeatBlob {
  /** 0–100 percent of container width */
  x: number
  /** 0–100 percent of container height */
  y: number
}

interface Props {
  blobs: HeatBlob[]
  className?: string
}

export function HeatmapCanvas({ blobs, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Cache stamp per radius size
  const stampRef = useRef<{ radius: number; stamp: ImageData } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d", { willReadFrequently: true })
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    if (W === 0 || H === 0) return

    const radius = Math.round(Math.min(W, H) * 0.10)   // 10% of shorter side
    if (!stampRef.current || stampRef.current.radius !== radius) {
      stampRef.current = { radius, stamp: makeStamp(radius) }
    }
    const { stamp } = stampRef.current
    const stampSize = radius * 2

    // ── 1. Accumulate intensity into a Float32 buffer ────────────────────────
    const intensity = new Float32Array(W * H)

    for (const blob of blobs) {
      const bx = Math.round((blob.x / 100) * W)
      const by = Math.round((blob.y / 100) * H)
      const x0 = bx - radius
      const y0 = by - radius

      for (let sy = 0; sy < stampSize; sy++) {
        const cy = y0 + sy
        if (cy < 0 || cy >= H) continue
        for (let sx = 0; sx < stampSize; sx++) {
          const cx = x0 + sx
          if (cx < 0 || cx >= W) continue
          const si = (sy * stampSize + sx) * 4 + 3   // alpha channel of stamp
          intensity[cy * W + cx] += stamp.data[si] / 255
        }
      }
    }

    // ── 2. Find max for normalisation ────────────────────────────────────────
    let maxVal = 0
    for (let i = 0; i < intensity.length; i++) {
      if (intensity[i] > maxVal) maxVal = intensity[i]
    }

    // ── 3. Map through inferno colormap → ImageData ──────────────────────────
    const output = ctx.createImageData(W, H)
    const od = output.data

    if (maxVal > 0) {
      for (let i = 0; i < intensity.length; i++) {
        const t = Math.min(intensity[i] / maxVal, 1)
        const lutIdx = Math.round(t * 255) * 3
        const alpha = t > 0.02 ? Math.round(t * 220) : 0   // fade out near-zero
        od[i * 4]     = COLORMAP[lutIdx]
        od[i * 4 + 1] = COLORMAP[lutIdx + 1]
        od[i * 4 + 2] = COLORMAP[lutIdx + 2]
        od[i * 4 + 3] = alpha
      }
    }

    ctx.putImageData(output, 0, 0)
  }, [blobs])

  // Keep canvas pixel dimensions in sync with display size
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width  = Math.round(width)
        canvas.height = Math.round(height)
      }
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  )
}
