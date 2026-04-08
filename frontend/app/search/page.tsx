import Link from "next/link"
import { NLSearchBar } from "@/components/NLSearchBar"

export default function SearchPage() {
  return (
    <main className="h-screen overflow-hidden flex flex-col bg-slate-50" style={{ padding: "12px 16px", gap: 12 }}>
      <div className="flex items-center justify-between shrink-0">
        <h1 className="text-indigo-900 text-lg font-extrabold tracking-tight">Campus Surveillance</h1>
        <nav className="flex items-center gap-5">
          <Link href="/" className="text-slate-400 text-xs hover:text-slate-600 transition-colors">
            ← Live Feed
          </Link>
          <Link href="/analytics" className="text-slate-400 text-xs hover:text-slate-600 transition-colors">
            Analytics
          </Link>
          <span className="text-indigo-600 text-xs border-b border-indigo-400 pb-0.5 font-medium">Search</span>
        </nav>
      </div>

      <div className="shrink-0">
        <p className="text-[11px] text-slate-700 font-bold tracking-widest uppercase mb-1">
          Natural Language Search
        </p>
        <p className="text-[11px] text-slate-400">
          Searches recorded frames using CLIP semantic similarity — results update as frames are indexed.
        </p>
      </div>

      <NLSearchBar />
    </main>
  )
}
