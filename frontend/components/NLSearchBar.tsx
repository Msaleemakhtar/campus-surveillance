"use client"

import { useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

const CAM_LABEL: Record<string, string> = {
  cam01: "Main Entrance", cam02: "Hallway A", cam03: "Hallway B",
  cam04: "Library", cam11: "Main Gate", cam15: "Sports Ground",
  cam19: "THREAT FEED", cam20: "INTRUSION FEED",
}

interface SearchResult {
  cam_id: string
  timestamp: string
  frame_url: string | null
}

function formatTs(iso: string) {
  try {
    return new Date(iso).toLocaleString("en-GB", {
      day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit",
    })
  } catch { return iso }
}

export function NLSearchBar() {
  const [query, setQuery]     = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [searched, setSearched] = useState(false)

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/search?query=${encodeURIComponent(query)}&limit=12`)
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResults(data.results ?? [])
      setSearched(true)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Search failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6 flex-1 min-h-0">

      {/* Search bar */}
      <form onSubmit={handleSearch} className="flex gap-3 shrink-0">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. person running near gate, two people fighting, someone loitering…"
          className="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white text-sm font-semibold px-5 py-2.5 shadow-sm transition-colors"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      {/* Results */}
      {searched && !loading && results.length === 0 && (
        <p className="text-sm text-slate-400 italic">No matching frames found.</p>
      )}

      {searched && !loading && results.length > 0 && (
        // Scroll wrapper is separate from the grid — grid always derives its
        // width from its block parent, avoiding flex-1 column-width quirks.
        <div className="overflow-y-auto flex-1 min-h-0 pr-1"
          style={{ scrollbarWidth: "thin", scrollbarColor: "#cbd5e1 transparent" }}>
          <div className="grid grid-cols-2 gap-4">
            {results.map((r, i) => (
              <div key={i} className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
                {/* padding-top 56.25% = 9/16 forces 16:9 landscape regardless of source frame dimensions */}
                <div className="relative w-full bg-black" style={{ paddingTop: "56.25%" }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API}${r.frame_url}`}
                    alt={`${r.cam_id} at ${r.timestamp}`}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                </div>
                <div className="px-3 py-2.5">
                  <p className="text-[11px] text-slate-700 font-bold uppercase tracking-widest">
                    {CAM_LABEL[r.cam_id] ?? r.cam_id}
                  </p>
                  <p className="text-[10px] text-slate-400 mt-0.5">{formatTs(r.timestamp)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!searched && !loading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2">
          <p className="text-slate-700 text-base font-medium">Type a description of what you&apos;re looking for.</p>
          <p className="text-slate-500 text-sm">
            CLIP matches frame content semantically — no exact keywords needed.
          </p>
        </div>
      )}
    </div>
  )
}
