import { NLSearchBar } from "@/components/NLSearchBar"

export default function SearchPage() {
  return (
    <main className="h-full overflow-hidden flex flex-col bg-slate-50" style={{ padding: "12px 16px", gap: 12 }}>
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
