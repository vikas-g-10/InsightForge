import { motion } from 'motion/react'
import { Sparkles, ThumbsUp, AlertTriangle, ShieldCheck } from 'lucide-react'

export function SceneFallback() {
  return (
    <div className="relative w-full h-full min-h-[420px] flex flex-col justify-between p-8 sm:p-12 overflow-hidden bg-radial from-slate-900 via-slate-950 to-slate-950 text-slate-100 select-none">
      {/* Ambient background glows */}
      <div className="absolute -top-24 -left-24 size-96 rounded-full bg-emerald-500/15 blur-3xl pointer-events-none" />
      <div className="absolute top-1/2 -right-24 size-96 rounded-full bg-rose-500/15 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 left-1/3 size-96 rounded-full bg-indigo-500/20 blur-3xl pointer-events-none" />

      {/* Header Info */}
      <div className="relative z-10 space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-semibold backdrop-blur-md">
          <Sparkles size={14} className="text-indigo-400 animate-pulse" />
          <span>The Good. The Bad. The Insight.</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
          Intelligence from Customer Voice
        </h2>
        <p className="text-xs sm:text-sm text-slate-400 max-w-md leading-relaxed">
          Transform unstructured reviews, defect images, and complaint signals into prioritized engineering action.
        </p>
      </div>

      {/* Floating Sample Feedback Cards */}
      <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 gap-4 my-8 max-w-lg">
        {/* Positive Review Badge */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="rounded-2xl border border-emerald-500/30 bg-emerald-950/40 backdrop-blur-md p-4 shadow-xl shadow-emerald-950/30"
        >
          <div className="flex items-center gap-2 mb-2 text-emerald-400 text-xs font-bold">
            <ThumbsUp size={14} />
            <span>Positive Feedback</span>
          </div>
          <p className="text-xs text-emerald-200 italic">"★ 5.0 Battery lasts 3 days and build feels premium."</p>
          <div className="mt-3 flex items-center justify-between text-[10px] text-emerald-400 font-medium">
            <span>Battery Life · Build Quality</span>
            <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">+18% NPS</span>
          </div>
        </motion.div>

        {/* Negative Review Badge */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="rounded-2xl border border-rose-500/30 bg-rose-950/40 backdrop-blur-md p-4 shadow-xl shadow-rose-950/30"
        >
          <div className="flex items-center gap-2 mb-2 text-rose-400 text-xs font-bold">
            <AlertTriangle size={14} />
            <span>Critical Defect Signal</span>
          </div>
          <p className="text-xs text-rose-200 italic">"★ 1.0 Bluetooth disconnects every 10 minutes."</p>
          <div className="mt-3 flex items-center justify-between text-[10px] text-rose-400 font-medium">
            <span>Connectivity · Firmware</span>
            <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold">P0 Alert</span>
          </div>
        </motion.div>
      </div>

      {/* Footer Pill Status */}
      <div className="relative z-10 flex flex-wrap items-center gap-4 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} className="text-indigo-400" />
          <span>Enterprise RLS Isolation</span>
        </div>
        <div className="size-1 rounded-full bg-slate-700" />
        <div>Ollama Vision Diagnosis</div>
        <div className="size-1 rounded-full bg-slate-700" />
        <div>Fix-This-First Simulation</div>
      </div>
    </div>
  )
}
