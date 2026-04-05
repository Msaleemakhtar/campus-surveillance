import type { WSMessage } from "./types"

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000"

export type MessageHandler = (msg: WSMessage) => void

export class CameraWebSocket {
  private ws: WebSocket | null = null
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private readonly reconnectDelay = 3000

  constructor(
    private readonly camId: string,
    private readonly onMessage: MessageHandler,
    private readonly onStatusChange?: (connected: boolean) => void,
  ) {}

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return

    this.ws = new WebSocket(`${WS_BASE}/ws/stream/${this.camId}`)

    this.ws.onopen = () => {
      this.onStatusChange?.(true)
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer)
        this.reconnectTimer = null
      }
    }

    this.ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage
        this.onMessage(msg)
      } catch {
        // malformed message — ignore
      }
    }

    this.ws.onclose = () => {
      this.onStatusChange?.(false)
      this.scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, this.reconnectDelay)
  }
}
