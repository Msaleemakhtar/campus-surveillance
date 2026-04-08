"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ShieldCheck, Video, BarChart3, Search } from "lucide-react"

const links = [
  { href: "/", label: "Live Feed", icon: Video },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/search", label: "Search", icon: Search },
]

export function Navbar() {
  const pathname = usePathname()

  return (
    <header className="shrink-0 bg-slate-900 border-b border-slate-700 px-5 h-14 flex items-center justify-between">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-md bg-indigo-600 flex items-center justify-center">
          <ShieldCheck className="w-5 h-5 text-white" strokeWidth={2.5} />
        </div>
        <span className="text-white text-base font-bold tracking-tight">
          Campus<span className="text-indigo-400">AI</span>
        </span>
        <span className="hidden sm:inline ml-1 text-slate-500 text-sm font-medium">Surveillance</span>
      </div>

      {/* Nav links */}
      <nav className="flex items-center gap-1.5">
        {links.map(({ href, label, icon: Icon }) => {
          const active = pathname === href
          return (
            <Link
              key={href}
              href={href}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors
                ${active
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
                }
              `}
            >
              <Icon className="w-4 h-4" strokeWidth={active ? 2.5 : 2} />
              <span>{label}</span>
            </Link>
          )
        })}
      </nav>

      {/* Status pill */}
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-slate-400 text-sm hidden sm:inline">Live</span>
      </div>
    </header>
  )
}
