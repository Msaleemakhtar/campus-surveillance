import { AnalyticsDashboard } from "@/components/AnalyticsDashboard"

export default function AnalyticsPage() {
  return (
    <main className="h-full overflow-hidden flex flex-col bg-slate-50" style={{ padding: "12px 16px", gap: 12 }}>
      <AnalyticsDashboard />
    </main>
  )
}
