"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  CreditCard,
  Brain,
  BarChart3,
  Zap,
  Settings,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/payments", label: "Payments", icon: CreditCard },
  { href: "/pipeline", label: "AI Pipeline", icon: Brain },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/settings", label: "Model Control", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-gray-900/80 border-r border-gray-800 flex flex-col z-50 backdrop-blur-sm">
      {/* Logo */}
      <div className="p-6 border-b border-gray-800">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">PayRevive</h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest">
              AI Recovery Engine
            </p>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "bg-blue-500/15 text-blue-400 border border-blue-500/20"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
              }`}
            >
              <Icon className="w-[18px] h-[18px]" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="px-4 py-3 bg-gray-800/40 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-dot" />
            <span className="text-xs font-medium text-green-400">
              System Online
            </span>
          </div>
          <p className="text-[10px] text-gray-500">
            Track 03 · Razorpay AI Buildathon
          </p>
        </div>
      </div>
    </aside>
  );
}
