"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { CameraWebSocket } from "@/lib/websocket"
import type { WSMessage } from "@/lib/types"

interface UseCameraFeedResult {
  canvasRef: React.RefObject<HTMLCanvasElement>
  lastMessage: WSMessage | null
  connected: boolean
}

export function useCameraFeed(camId: string): UseCameraFeedResult {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<CameraWebSocket | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const [connected, setConnected] = useState(false)

  const drawFrame = useCallback((msg: WSMessage) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    if (!imgRef.current) {
      imgRef.current = new Image()
    }
    const img = imgRef.current

    img.onload = () => {
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width
        canvas.height = img.height
      }
      ctx.drawImage(img, 0, 0)
    }
    img.src = `data:image/jpeg;base64,${msg.frame}`
  }, [])

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      drawFrame(msg)
      setLastMessage(msg)
    },
    [drawFrame],
  )

  useEffect(() => {
    const ws = new CameraWebSocket(camId, handleMessage, setConnected)
    wsRef.current = ws
    ws.connect()

    return () => {
      ws.disconnect()
      wsRef.current = null
      imgRef.current = null
    }
  }, [camId, handleMessage])

  return { canvasRef, lastMessage, connected }
}
