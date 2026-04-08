import Link from "next/link"
import { AnalyticsDashboard } from "@/components/AnalyticsDashboard"

export default function AnalyticsPage() {
  return (
    <main className="h-screen overflow-hidden flex flex-col bg-slate-50" style={{ padding: "12px 16px", gap: 12 }}>
      <div className="flex items-center justify-between shrink-0">
        <h1 className="text-indigo-900 text-lg font-extrabold tracking-tight">Campus Surveillance</h1>
        <nav className="flex items-center gap-5">
          <Link href="/" className="text-slate-400 text-xs hover:text-slate-600 transition-colors">
            ← Live Feed
          </Link>
          <span className="text-indigo-600 text-xs border-b border-indigo-400 pb-0.5 font-medium">Analytics</span>
        </nav>
      </div>

      <AnalyticsDashboard />
    </main>
  )
}
