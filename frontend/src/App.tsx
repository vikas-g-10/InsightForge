import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  Calendar,
  Camera,
  Check,
  CheckCircle2,
  Clock3,
  Eye,
  FileSpreadsheet,
  FileText,
  Filter,
  Flame,
  Gauge,
  GitCompare,
  History,
  Image as ImageIcon,
  Info,
  LayoutDashboard,
  Lightbulb,
  LogOut,
  Moon,
  Play,
  RefreshCw,
  Scale,
  Settings as SettingsIcon,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Sun,
  TrendingUp,
  Upload,
  Users,
  X,
} from 'lucide-react'
import { useAuth } from './context/useAuth'
import { AuthView } from './components/auth/AuthView'
import { IntroSplash } from './components/IntroSplash'
import { ModelEvaluationPage } from './components/ModelEvaluationPage'
import { fetchUserHistory, saveUserHistory, type HistoryItem } from './services/historyService'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  API_BASE_URL,
  analyzeFile,
  analyzeProductImage,
  analyzeReviews,
  compareProducts,
  compareVersions,
  fetchEmergingIssues,
  fetchHealth,
  fetchRootCauseHypotheses,
  fetchVisionHealth,
  simulateFixes,
} from './api'
import type {
  AnalyzeResponse,
  EmergingIssueFinding,
  EmergingIssuesResponse,
  HealthResponse,
  ImageDefectAnalysis,
  IssueCluster,
  ProductCompareResponse,
  ReviewInputItem,
  RootCauseAnalysisResult,
  SimulateResponse,
  VersionCompareResponse,
  VisionHealthResponse,
} from './api'
import { SAMPLE_REVIEWS } from './sampleData'


type Theme = 'light' | 'dark'

type Page =
  | 'Overview'
  | 'Analyze Reviews'
  | 'Insights'
  | 'Emerging Issues'
  | 'Defect Vision'
  | 'Compare'
  | 'Model Evaluation'
  | 'Analysis History'
  | 'Settings'

interface SharedProps {
  dark: boolean
  surface: string
  muted: string
  control: string
}

interface PageProps extends SharedProps {
  onOpenPage: (page: Page) => void
}

// HistoryItem interface imported from services/historyService

const navItems = [
  { name: 'Overview' as Page, icon: LayoutDashboard },
  { name: 'Analyze Reviews' as Page, icon: FileSpreadsheet },
  { name: 'Insights' as Page, icon: Lightbulb },
  { name: 'Emerging Issues' as Page, icon: TrendingUp },
  { name: 'Defect Vision' as Page, icon: Camera },
  { name: 'Compare' as Page, icon: GitCompare },
  { name: 'Model Evaluation' as Page, icon: Brain },
  { name: 'Analysis History' as Page, icon: Clock3 },
  { name: 'Settings' as Page, icon: SettingsIcon },
]

function PageHeading({
  title,
  description,
  muted,
}: {
  title: string
  description: string
  muted: string
}) {
  return (
    <div className="mb-6">
      <h2 className="text-2xl font-bold tracking-tight md:text-3xl">{title}</h2>
      <p className={`mt-2 text-sm leading-6 ${muted}`}>{description}</p>
    </div>
  )
}

function EvidenceModal({
  dark,
  surface,
  muted,
  control,
  cluster,
  analysis,
  imageFindings,
  onClose,
  onInspectVision,
}: SharedProps & {
  cluster: IssueCluster | null
  analysis: AnalyzeResponse | null
  imageFindings?: ImageDefectAnalysis[]
  onClose: () => void
  onInspectVision?: (cluster: IssueCluster) => void
}) {
  const [activeTab, setActiveTab] = useState<'excerpts' | 'root_cause'>('excerpts')
  const [rootCauseResult, setRootCauseResult] = useState<RootCauseAnalysisResult | null>(null)
  const [loadingRootCause, setLoadingRootCause] = useState(false)
  const [rootCauseError, setRootCauseError] = useState<string | null>(null)

  useEffect(() => {
    if (activeTab === 'root_cause' && !rootCauseResult && cluster && analysis) {
      let isMounted = true
      const timer = setTimeout(() => {
        if (isMounted) setLoadingRootCause(true)
      }, 0)

      const reviews = analysis.reviews.map((r) => ({
        id: r.review_id,
        text: r.original_text,
        rating: r.rating ?? r.pseudo_rating,
        date: r.date,
        version: r.version,
        product: r.product,
      }))

      fetchRootCauseHypotheses(reviews, imageFindings || [], cluster.cluster_id)
        .then((res) => {
          if (isMounted) {
            const match = res.results.find((r) => r.cluster_id === cluster.cluster_id)
            if (match) {
              setRootCauseResult(match)
            } else if (res.results.length > 0) {
              setRootCauseResult(res.results[0])
            }
            setLoadingRootCause(false)
          }
        })
        .catch((err) => {
          if (isMounted) {
            setRootCauseError(err instanceof Error ? err.message : 'Failed to deduce root-cause hypotheses.')
            setLoadingRootCause(false)
          }
        })

      return () => {
        isMounted = false
        clearTimeout(timer)
      }
    }
  }, [activeTab, cluster, analysis, imageFindings, rootCauseResult])

  if (!cluster) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="evidence-title"
        className={`w-full max-w-3xl max-h-[88vh] overflow-y-auto rounded-2xl border p-6 shadow-2xl ${surface}`}
        onClick={(event) => event.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase ${
                  cluster.polarity === 'neg'
                    ? 'bg-rose-500/10 text-rose-500'
                    : cluster.polarity === 'pos'
                      ? 'bg-emerald-500/10 text-emerald-500'
                      : 'bg-amber-500/10 text-amber-500'
                }`}
              >
                {cluster.aspect} · {cluster.polarity}
              </span>
              {cluster.is_safety_alert && (
                <span className="rounded-full bg-rose-500/15 px-2.5 py-0.5 text-xs font-bold text-rose-500">
                  {cluster.safety_tier.toUpperCase()} SAFETY ALERT
                </span>
              )}
            </div>
            <h3 id="evidence-title" className="mt-2 text-xl font-bold">
              {cluster.label}
            </h3>
            <p className={`mt-1 text-xs ${muted}`}>
              Cluster ID: {cluster.cluster_id} · {cluster.size} supporting clauses across{' '}
              {cluster.review_count} unique customer reviews
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close evidence preview"
            className={`rounded-lg border p-2 ${control}`}
          >
            <X size={17} />
          </button>
        </div>

        {/* Tab Toggle */}
        <div className={`mt-5 flex border-b pb-1 gap-2 ${dark ? 'border-slate-700/60' : 'border-slate-200'}`}>
          <button
            onClick={() => setActiveTab('excerpts')}
            className={`px-4 py-2 text-xs font-bold rounded-lg transition ${
              activeTab === 'excerpts'
                ? 'bg-indigo-600 text-white shadow'
                : `${control} hover:border-slate-500`
            }`}
          >
            Review Evidence Excerpts ({cluster.evidence.length})
          </button>
          <button
            onClick={() => setActiveTab('root_cause')}
            className={`px-4 py-2 text-xs font-bold rounded-lg flex items-center gap-1.5 transition ${
              activeTab === 'root_cause'
                ? 'bg-indigo-600 text-white shadow'
                : `${control} hover:border-slate-500 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`
            }`}
          >
            <Sparkles size={13} />
            Root-Cause Hypotheses & Verification
          </button>
        </div>

        {/* Tab 1: Review Evidence Excerpts */}
        {activeTab === 'excerpts' && (
          <div className="mt-4 space-y-4">
            <div className={`rounded-xl border p-4 text-sm ${dark ? 'border-indigo-500/20 bg-indigo-500/5' : 'border-indigo-200 bg-indigo-50/70'}`}>
              <div className={`text-xs font-semibold uppercase tracking-wider ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                Representative Medoid Quote
              </div>
              <p className="mt-1.5 italic font-medium">"{cluster.example_quote}"</p>
              <div className={`mt-2 flex gap-4 text-xs ${muted}`}>
                <span>Example Clause ID: {cluster.example_clause_id}</span>
                <span>Assigned Team: {cluster.team || 'Product/Engineering'}</span>
                <span>Priority: {cluster.priority || 'P1'}</span>
              </div>

              <div className={`mt-3 pt-3 border-t flex items-center justify-between ${dark ? 'border-indigo-500/20' : 'border-indigo-200'}`}>
                <span className={`text-xs ${muted}`}>Corroborate customer complaint with physical imagery:</span>
                <button
                  onClick={() => {
                    onInspectVision?.(cluster)
                    onClose()
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                    dark
                      ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20'
                      : 'border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100'
                  }`}
                >
                  <Camera size={14} /> Attach or Inspect Defect Photo Evidence →
                </button>
              </div>
            </div>

            <h4 className={`mt-4 mb-2 text-xs font-bold uppercase tracking-wider ${dark ? 'text-slate-400' : 'text-slate-600'}`}>
              Supporting Customer Review Excerpts ({cluster.evidence.length})
            </h4>

            <div className="space-y-3">
              {cluster.evidence.map((ev, idx) => (
                <div
                  key={`${ev.review_id}-${ev.clause_id}-${idx}`}
                  className={`rounded-xl border p-3.5 text-sm ${
                    dark
                      ? 'border-slate-800 bg-slate-950/70'
                      : 'border-slate-200 bg-slate-50/80'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className={`font-semibold ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>
                      Review ID: {ev.review_id}
                    </span>
                    <div className="flex items-center gap-2">
                      <span
                        className={`font-medium ${
                          ev.polarity === 'neg'
                            ? (dark ? 'text-rose-400' : 'text-rose-600 font-semibold')
                            : ev.polarity === 'pos'
                              ? (dark ? 'text-emerald-400' : 'text-emerald-700 font-semibold')
                              : (dark ? 'text-amber-400' : 'text-amber-700 font-semibold')
                        }`}
                      >
                        Sentiment: {ev.score > 0 ? `+${ev.score.toFixed(2)}` : ev.score.toFixed(2)}
                      </span>
                      {ev.safety_tier !== 'none' && (
                        <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                          dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                        }`}>
                          {ev.safety_tier}
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="leading-relaxed">"{ev.text}"</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 2: Root-Cause Hypotheses & Verification */}
        {activeTab === 'root_cause' && (
          <div className="mt-4 space-y-4">
            {loadingRootCause ? (
              <div className="py-12 text-center">
                <RefreshCw size={24} className={`mx-auto animate-spin mb-3 ${dark ? 'text-indigo-400' : 'text-indigo-600'}`} />
                <p className="text-xs font-semibold">Deducing evidence-grounded hypotheses...</p>
                <p className={`text-[11px] mt-1 ${muted}`}>
                  Analyzing review cluster symptoms, temporal frequency, and associated visual observations.
                </p>
              </div>
            ) : rootCauseError ? (
              <div className={`rounded-xl border p-4 text-xs ${dark ? 'border-rose-500/30 bg-rose-500/10 text-rose-400' : 'border-rose-200 bg-rose-50 text-rose-700'}`}>
                {rootCauseError}
              </div>
            ) : rootCauseResult ? (
              <div className="space-y-4">
                {/* Disclaimer Banner */}
                <div className={`rounded-xl border p-3.5 text-xs ${
                  dark ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-300' : 'border-indigo-200 bg-indigo-50/70 text-indigo-900'
                }`}>
                  <div className="font-bold flex items-center gap-1.5">
                    <Info size={14} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Engineering Hypothesis Disclaimer
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed opacity-90">
                    {rootCauseResult.disclaimer}
                  </p>
                </div>

                {/* Insufficient Evidence State */}
                {!rootCauseResult.has_sufficient_evidence ? (
                  <div className={`rounded-2xl border p-6 text-center ${dark ? 'border-amber-500/30 bg-amber-500/5' : 'border-amber-200 bg-amber-50/70'}`}>
                    <AlertTriangle size={28} className={`mx-auto mb-2 ${dark ? 'text-amber-400' : 'text-amber-600'}`} />
                    <h4 className={`text-sm font-bold ${dark ? 'text-amber-400' : 'text-amber-700'}`}>Insufficient Evidence for Root-Cause Inference</h4>
                    <p className={`mt-2 text-xs max-w-md mx-auto leading-relaxed ${muted}`}>
                      {rootCauseResult.insufficient_evidence_reason}
                    </p>
                    <div className="mt-4">
                      <button
                        onClick={() => {
                          onInspectVision?.(cluster)
                          onClose()
                        }}
                        className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-bold text-white shadow hover:bg-indigo-500"
                      >
                        <Camera size={14} /> Attach Photo Evidence to Corroborate
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Summary Context */}
                    <div className={`rounded-xl border p-3.5 text-xs ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className={`font-semibold ${dark ? 'text-slate-200' : 'text-slate-900'}`}>{rootCauseResult.summary}</span>
                        <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                          dark ? 'bg-emerald-500/15 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                        }`}>
                          {rootCauseResult.hypotheses.length} Hypotheses Ranked
                        </span>
                      </div>
                      {rootCauseResult.temporal_pattern_summary && (
                        <div className={`mt-2 text-[11px] ${muted}`}>
                          <strong>Temporal Context:</strong> {rootCauseResult.temporal_pattern_summary}
                        </div>
                      )}
                    </div>

                    {/* Hypotheses List */}
                    <div className="space-y-4">
                      {rootCauseResult.hypotheses.map((hyp) => (
                        <div
                          key={hyp.hypothesis_id}
                          className={`rounded-2xl border p-5 space-y-3 ${
                            dark ? 'border-slate-800 bg-slate-900/80' : 'border-slate-200 bg-white shadow-sm'
                          }`}
                        >
                          {/* Hypothesis Header */}
                          <div className={`flex flex-wrap items-center justify-between gap-2 border-b pb-2.5 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                            <div className="flex items-center gap-2">
                              <span className="rounded-md bg-indigo-600 px-2.5 py-0.5 text-xs font-extrabold text-white">
                                Rank #{hyp.rank}
                              </span>
                              <span className={`rounded-md px-2.5 py-0.5 text-xs font-semibold capitalize ${
                                dark ? 'bg-indigo-500/15 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                              }`}>
                                {hyp.category.replace('_', ' ')}
                              </span>
                            </div>

                            <span
                              className={`rounded-md px-2.5 py-0.5 text-xs font-bold uppercase ${
                                hyp.confidence_level === 'high'
                                  ? (dark ? 'bg-emerald-500/15 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                                  : hyp.confidence_level === 'medium'
                                    ? (dark ? 'bg-amber-500/15 text-amber-400' : 'bg-amber-100 text-amber-800 border border-amber-200')
                                    : (dark ? 'bg-slate-500/15 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200')
                              }`}
                            >
                              {hyp.confidence_level} Confidence ({Math.round(hyp.confidence_score * 100)}%)
                            </span>
                          </div>

                          {/* Proposed Mechanism */}
                          <div>
                            <div className={`text-[11px] font-bold uppercase tracking-wider ${muted}`}>
                              Proposed Failure Mechanism
                            </div>
                            <p className={`mt-1 text-sm font-bold ${dark ? 'text-slate-100' : 'text-slate-900'}`}>
                              {hyp.proposed_cause}
                            </p>
                            <p className={`mt-1 text-xs leading-relaxed ${muted}`}>
                              {hyp.explanation}
                            </p>
                          </div>

                          {/* Supporting Evidence References */}
                          <div>
                            <div className={`text-xs font-semibold flex items-center gap-1 ${dark ? 'text-emerald-400' : 'text-emerald-700'}`}>
                              <CheckCircle2 size={13} /> Supporting Review & Photo Evidence ({hyp.supporting_evidence.length})
                            </div>
                            <div className="mt-1.5 space-y-1">
                              {hyp.supporting_evidence.map((ev, idx) => (
                                <div
                                  key={idx}
                                  className={`rounded-lg border px-2.5 py-1.5 text-[11px] flex items-center justify-between gap-2 ${
                                    dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'
                                  }`}
                                >
                                  <span className="truncate italic">"{ev.excerpt_or_observation}"</span>
                                  <span className={`shrink-0 font-mono text-[10px] ${dark ? 'text-indigo-400' : 'text-indigo-600 font-semibold'}`}>
                                    {ev.source_type.toUpperCase()}:{ev.reference_id}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Contradicting Evidence & Missing Info */}
                          {hyp.contradicting_evidence.length > 0 && (
                            <div className={`rounded-xl border p-3 text-xs ${dark ? 'border-amber-500/20 bg-amber-500/5' : 'border-amber-200 bg-amber-50/70'}`}>
                              <span className={`font-bold flex items-center gap-1 ${dark ? 'text-amber-400' : 'text-amber-800'}`}>
                                <AlertTriangle size={13} /> Contradicting Reports (Defect Is Non-Universal)
                              </span>
                              <div className="mt-1.5 space-y-1">
                                {hyp.contradicting_evidence.map((c, idx) => (
                                  <div key={idx} className={`text-[11px] ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
                                    • Review {c.reference_id}: {c.excerpt_or_observation}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Missing Information */}
                          {hyp.missing_information.length > 0 && (
                            <div className={`text-[11px] ${muted}`}>
                              <strong>Missing Variables:</strong> {hyp.missing_information.join(' ')}
                            </div>
                          )}

                          {/* Engineering Verification Steps */}
                          <div className={`border-t pt-2.5 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                            <div className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1 ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                              <SlidersHorizontal size={13} /> Recommended Engineering Bench Verification
                            </div>
                            <ul className={`mt-1.5 space-y-1 text-xs ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
                              {hyp.verification_steps.map((stp, idx) => (
                                <li key={idx}>• {stp}</li>
                              ))}
                            </ul>
                          </div>

                          {/* Falsification Criteria */}
                          <div className={`rounded-xl border p-3 text-xs ${dark ? 'border-rose-500/20 bg-rose-500/5' : 'border-rose-200 bg-rose-50/70'}`}>
                            <span className={`font-bold uppercase tracking-wider text-[11px] ${dark ? 'text-rose-400' : 'text-rose-700'}`}>
                              Falsification Criterion (Reject If Observed)
                            </span>
                            <p className={`mt-1 text-xs ${dark ? 'text-slate-200' : 'text-slate-800'}`}>
                              {hyp.falsification_criteria}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ) : null}
          </div>
        )}

        {/* Footer */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-indigo-500"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}


function OverviewPage({
  dark,
  surface,
  muted,
  onOpenPage,
  analysis,
  onShowEvidence,
}: PageProps & {
  analysis: AnalyzeResponse | null
  onShowEvidence: (cluster: IssueCluster) => void
}) {
  const summary = analysis?.meta?.summary
  const reviews = analysis?.reviews || []
  const clusters = analysis?.clusters || []
  const insights = analysis?.insights || []

  // Dynamic metrics from actual analysis
  const totalReviews = summary?.total_reviews ?? 0
  const analyzedReviews = summary?.analyzed_reviews ?? 0
  const uniqueReviews = summary?.unique_reviews ?? 0
  const duplicateCount = summary?.duplicate_count ?? 0

  const negativeCount = summary?.sentiment_breakdown?.neg ?? 0
  const negativePct =
    analyzedReviews > 0
      ? ((negativeCount / analyzedReviews) * 100).toFixed(1)
      : '0.0'

  const issuesCount = clusters.length
  const criticalSafety = summary?.safety_breakdown?.critical ?? 0
  const highSafety = summary?.safety_breakdown?.high ?? 0
  const safetyCount = criticalSafety + highSafety

  // Chart data: Aspect sentiment counts from actual analysis
  const aspectData = Object.entries(summary?.aspect_mentions || {})
    .slice(0, 6)
    .map(([aspect, mentions]) => {
      const aspNeg = clusters
        .filter((c) => c.aspect.toLowerCase() === aspect.toLowerCase() && c.polarity === 'neg')
        .reduce((sum, c) => sum + c.size, 0)
      const aspPos = clusters
        .filter((c) => c.aspect.toLowerCase() === aspect.toLowerCase() && c.polarity === 'pos')
        .reduce((sum, c) => sum + c.size, 0)
      return {
        aspect: aspect.charAt(0).toUpperCase() + aspect.slice(1),
        negative: aspNeg,
        positive: aspPos,
        mentions,
      }
    })

  // Safety alert review matches
  const safetyAlerts = reviews
    .filter((r) => r.safety.is_alert && r.safety.tier !== 'none')
    .slice(0, 3)

  return (
    <>
      <section className="flex flex-col justify-between gap-4 xl:flex-row xl:items-end">
        <div className="max-w-3xl">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-indigo-500/10 px-3 py-1 text-xs font-semibold text-indigo-500">
              INSIGHTFORGE
            </span>
            <span
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                analysis
                  ? (dark ? 'bg-emerald-500/10 text-emerald-400' : 'bg-emerald-50 text-emerald-700 border border-emerald-200')
                  : (dark ? 'bg-amber-400/10 text-amber-300' : 'bg-amber-50 text-amber-800 border border-amber-200')
              }`}
            >
              {analysis ? `Live Analysis (${analysis.meta.mode.toUpperCase()} mode)` : 'Awaiting Review Analysis'}
            </span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
            Turn customer feedback into your next best action.
          </h2>
          <p className={`mt-2 max-w-2xl text-sm leading-6 ${muted}`}>
            Discover recurring product issues, inspect supporting reviews, and
            prioritize improvements using empirical evidence from customer feedback.
          </p>
        </div>

        <button
          onClick={() => onOpenPage('Analyze Reviews')}
          className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
        >
          <Sparkles size={16} /> Analyze New Reviews
        </button>
      </section>

      {/* Top 4 KPI Metrics Cards */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          {
            label: 'Reviews Analyzed',
            value: totalReviews > 0 ? totalReviews.toLocaleString() : '—',
            note:
              totalReviews > 0
                ? `${uniqueReviews} unique · ${duplicateCount} duplicate`
                : 'Upload reviews to begin',
            icon: Users,
            tone: 'text-indigo-500',
          },
          {
            label: 'Negative Complaints',
            value: totalReviews > 0 ? `${negativePct}%` : '—',
            note: totalReviews > 0 ? `${negativeCount} negative reviews` : 'Awaiting feedback',
            icon: BarChart3,
            tone: 'text-rose-500',
          },
          {
            label: 'Issue Clusters',
            value: totalReviews > 0 ? `${issuesCount}` : '—',
            note: totalReviews > 0 ? `Across ${aspectData.length} detected aspects` : 'No clusters formed',
            icon: Gauge,
            tone: 'text-violet-500',
          },
          {
            label: 'Safety Alerts',
            value: totalReviews > 0 ? `${safetyCount}` : '—',
            note: safetyCount > 0 ? `${criticalSafety} critical · ${highSafety} high` : 'Zero hazard alerts',
            icon: ShieldAlert,
            tone: safetyCount > 0 ? 'text-rose-500' : 'text-emerald-500',
          },
        ].map(({ label, value, note, icon: Icon, tone }) => (
          <article key={label} className={`rounded-xl border p-5 ${surface}`}>
            <div className="flex items-center justify-between">
              <span className={`text-sm ${muted}`}>{label}</span>
              <Icon size={19} className={tone} />
            </div>
            <div className="mt-4 text-3xl font-bold tracking-tight">{value}</div>
            <div className={`mt-2 text-xs ${muted}`}>{note}</div>
          </article>
        ))}
      </section>

      {/* Main Grid: Sentiment Chart + Top Fix This First Issues */}
      <section className="grid gap-6 xl:grid-cols-5">
        <article className={`rounded-xl border p-5 xl:col-span-2 ${surface}`}>
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-semibold">Aspect Sentiment Breakdown</h3>
              <p className={`mt-1 text-xs ${muted}`}>
                Complaints vs praises across detected aspects
              </p>
            </div>
            <BarChart3 size={19} className="text-indigo-500" />
          </div>

          <div className="mt-4 flex flex-wrap gap-4 text-xs font-medium">
            <span className="text-emerald-500">● Positive</span>
            <span className="text-rose-500">● Negative</span>
          </div>

          <div className="mt-4 h-60 w-full">
            {aspectData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={aspectData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={dark ? '#334155' : '#e2e8f0'} />
                  <XAxis
                    dataKey="aspect"
                    tick={{ fontSize: 11, fill: dark ? '#94a3b8' : '#64748b' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: dark ? '#94a3b8' : '#64748b' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: dark ? '#0f172a' : '#fff',
                      borderColor: dark ? '#334155' : '#e2e8f0',
                      borderRadius: 10,
                      color: dark ? '#f1f5f9' : '#0f172a',
                    }}
                    itemStyle={{
                      color: dark ? '#f1f5f9' : '#0f172a',
                    }}
                  />
                  <Bar dataKey="positive" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="negative" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full flex-col items-center justify-center text-center text-xs text-slate-500">
                <BarChart3 size={32} className="mb-2 opacity-30" />
                No aspect sentiment data available yet.
              </div>
            )}
          </div>
        </article>

        {/* Fix This First Priority Card */}
        <article className={`rounded-xl border p-5 xl:col-span-3 ${surface}`}>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="font-semibold">Fix This First — Priority Issues</h3>
              <p className={`mt-1 text-xs ${muted}`}>
                Ranked by customer rating damage and safety hazard level
              </p>
            </div>
            <button
              onClick={() => onOpenPage('Insights')}
              className="text-sm font-medium text-indigo-500 hover:text-indigo-400"
            >
              All insights →
            </button>
          </div>

          <div className="mt-4 space-y-3">
            {insights.slice(0, 3).map((ins, index) => {
              const matchingCluster = clusters.find(
                (c) => c.aspect.toLowerCase() === ins.aspect.toLowerCase()
              )
              return (
                <div
                  key={ins.aspect}
                  className={`rounded-xl border p-4 transition ${
                    dark
                      ? 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
                      : 'border-slate-200 bg-slate-50/70 hover:border-slate-300'
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="flex gap-3">
                      <div
                        className={`flex size-8 shrink-0 items-center justify-center rounded-lg text-sm font-bold ${
                          ins.priority_code === 'P0'
                            ? 'bg-rose-500/15 text-rose-500'
                            : index === 0
                              ? 'bg-amber-500/15 text-amber-500'
                              : 'bg-indigo-500/15 text-indigo-500'
                        }`}
                      >
                        {ins.priority_code || `#${ins.priority_rank}`}
                      </div>
                      <div>
                        <div className="font-semibold capitalize flex items-center gap-2">
                          {ins.aspect} Issues
                          {ins.safety_tier !== 'none' && (
                            <span className="rounded bg-rose-500/15 px-2 py-0.5 text-[10px] font-bold text-rose-500 uppercase">
                              {ins.safety_tier} Hazard
                            </span>
                          )}
                        </div>
                        <div className={`mt-1 text-xs ${muted}`}>
                          Assigned to: {ins.team || 'Product/Engineering'} · {ins.mentions} mentions (
                          {ins.complaint_count} complaints)
                        </div>
                      </div>
                    </div>

                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                        ins.confidence_label === 'low'
                          ? (dark ? 'bg-amber-500/10 text-amber-400' : 'bg-amber-100 text-amber-800 border border-amber-200')
                          : (dark ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200')
                      }`}
                    >
                      {ins.confidence_label === 'low' ? 'Low confidence (<30)' : 'High confidence'}
                    </span>
                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                    <div className={`flex flex-wrap gap-4 text-xs ${muted}`}>
                      <span>
                        <strong className="text-rose-500">{ins.pct_negative}%</strong> negative
                      </span>
                      <span>
                        Rating damage: <strong>{ins.damage} pts</strong>
                      </span>
                      <span>
                        Rating impact: <strong>-{ins.beta}★/complaint</strong>
                      </span>
                    </div>

                    {matchingCluster && (
                      <button
                        onClick={() => onShowEvidence(matchingCluster)}
                        className="text-xs font-semibold text-indigo-500 hover:text-indigo-400"
                      >
                        View evidence ({matchingCluster.size}) →
                      </button>
                    )}
                  </div>
                </div>
              )
            })}

            {insights.length === 0 && (
              <div className="py-8 text-center text-sm text-slate-500">
                No issues analyzed yet. Upload or paste reviews to view prioritized problems.
              </div>
            )}
          </div>
        </article>
      </section>

      {/* Safety Watch Section */}
      <section className={`rounded-xl border p-5 ${surface}`}>
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="font-semibold flex items-center gap-2">
              <ShieldAlert size={19} className="text-rose-500" /> Safety & Compliance Escalations
            </h3>
            <p className={`mt-1 text-xs ${muted}`}>
              Audited keywords with context and negation guards (fire, explosion, overheating, injury)
            </p>
          </div>
          {safetyAlerts.length > 0 && (
            <span className="rounded-full bg-rose-500/15 px-3 py-1 text-xs font-bold text-rose-500">
              {safetyAlerts.length} Action Required
            </span>
          )}
        </div>

        {safetyAlerts.length > 0 ? (
          <div className="mt-4 space-y-3">
            {safetyAlerts.map((rev) => (
              <div
                key={rev.review_id}
                className={`rounded-xl border p-4 ${
                  dark
                    ? 'border-rose-900/60 bg-rose-950/25'
                    : 'border-rose-200 bg-rose-50/70'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 font-semibold text-rose-500 text-sm">
                    <AlertTriangle size={16} /> Tier: {rev.safety.tier.toUpperCase()} Hazard (Review {rev.review_id})
                  </div>
                  <span className={`rounded-md px-2 py-0.5 text-xs font-bold ${
                    dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                  }`}>
                    P0 Escalation
                  </span>
                </div>
                <p className="mt-2 text-sm italic">"{rev.original_text}"</p>
                <div className={`mt-2 text-xs font-medium ${muted}`}>
                  Suggested Action:{' '}
                  <span className={`font-normal ${dark ? 'text-slate-300' : 'text-slate-800'}`}>
                    {rev.safety.suggested_action || 'Escalate to QA and Compliance within 24 hours.'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm text-emerald-500 flex items-center gap-2">
            <CheckCircle2 size={18} /> No active safety incidents detected in customer reviews.
          </div>
        )}
      </section>
    </>
  )
}

function AnalyzePage({
  dark,
  surface,
  muted,
  control,
  onAnalyzeComplete,
  analysis,
}: PageProps & {
  onAnalyzeComplete: (res: AnalyzeResponse) => void
  analysis: AnalyzeResponse | null
}) {
  const [activeTab, setActiveTab] = useState<'upload' | 'paste'>('paste')
  const [pastedText, setPastedText] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  function loadSampleData() {
    setPastedText(
      SAMPLE_REVIEWS.map(
        (r) => `${r.rating ? `[${r.rating}★] ` : ''}${r.text}`
      ).join('\n\n')
    )
    setSelectedFile(null)
    setActiveTab('paste')
    setErrorMessage(null)
  }

  async function handleRunAnalysis() {
    setLoading(true)
    setErrorMessage(null)

    try {
      if (activeTab === 'upload') {
        if (!selectedFile) {
          throw new Error('Please select a CSV or XLSX file first.')
        }
        const res = await analyzeFile(selectedFile)
        onAnalyzeComplete(res)
      } else {
        if (!pastedText.trim()) {
          throw new Error('Please paste review text or load sample reviews.')
        }
        // Split by lines or double newlines into reviews
        const rawLines = pastedText
          .split(/\n\s*\n|\n/)
          .map((s) => s.trim())
          .filter(Boolean)

        const items: ReviewInputItem[] = rawLines.map((line, idx) => {
          // Check for rating prefix like [5★] or 5/5
          const starMatch = line.match(/^\[?(\d(?:\.\d)?)[/★*]?\]?\s*(.*)$/)
          if (starMatch && Number(starMatch[1]) >= 1 && Number(starMatch[1]) <= 5) {
            return {
              id: `paste-${idx + 1}`,
              text: starMatch[2] || line,
              rating: Number(starMatch[1]),
            }
          }
          return {
            id: `paste-${idx + 1}`,
            text: line,
            rating: null,
          }
        })

        const res = await analyzeReviews(items)
        onAnalyzeComplete(res)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setErrorMessage(msg || 'Analysis failed. Please check backend status.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <PageHeading
        title="Analyze Customer Reviews"
        description="Upload CSV/XLSX review files or paste customer feedback to run NLP normalization, safety screening, aspect clustering, and impact ranking."
        muted={muted}
      />

      {errorMessage && (
        <div className={`mb-6 flex items-center justify-between rounded-xl border p-4 text-sm ${
          dark ? 'border-rose-500/40 bg-rose-500/10 text-rose-400' : 'border-rose-200 bg-rose-50 text-rose-700'
        }`}>
          <div className="flex items-center gap-2">
            <AlertTriangle size={18} />
            <span>{errorMessage}</span>
          </div>
          <button onClick={() => setErrorMessage(null)} className={dark ? 'text-rose-400 hover:text-rose-300' : 'text-rose-600 hover:text-rose-800'}>
            <X size={16} />
          </button>
        </div>
      )}

      <div className={`rounded-2xl border p-5 md:p-6 ${surface}`}>
        {/* Tab switcher: File upload vs Paste text */}
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 border-b pb-4">
          <div className={`flex rounded-lg border p-1 ${control}`}>
            <button
              onClick={() => setActiveTab('paste')}
              className={`flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold ${
                activeTab === 'paste' ? 'bg-indigo-600 text-white shadow' : ''
              }`}
            >
              <FileText size={15} /> Paste Reviews
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold ${
                activeTab === 'upload' ? 'bg-indigo-600 text-white shadow' : ''
              }`}
            >
              <Upload size={15} /> Upload CSV/XLSX
            </button>
          </div>

          <button
            onClick={loadSampleData}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
              dark
                ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20'
                : 'border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100'
            }`}
          >
            <Sparkles size={14} /> Load 12 Sample Reviews
          </button>
        </div>

        {activeTab === 'paste' ? (
          <div>
            <label className={`block text-xs font-semibold uppercase tracking-wider mb-2 ${muted}`}>
              Customer Review Text (one per paragraph or line)
            </label>
            <textarea
              rows={8}
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder="Paste reviews here, for example:&#10;&#10;[1★] The battery caught fire and started sparking while plugged in!&#10;[2★] Camera photos are blurry and low-light photos are noisy.&#10;[5★] Amazing display and lovely bright screen colors!&#10;Yeh app bahut accha hai aur fast hai."
              className={`w-full rounded-xl border p-4 text-sm font-mono focus:outline-hidden focus:ring-2 focus:ring-indigo-500 ${
                dark ? 'border-slate-800 bg-slate-950 text-slate-100' : 'border-slate-200 bg-slate-50 text-slate-900'
              }`}
            />
          </div>
        ) : (
          <div>
            <label
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                if (e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0])
              }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-5 py-12 text-center transition ${
                dark
                  ? 'border-slate-800 bg-slate-950 hover:border-indigo-500'
                  : 'border-slate-300 bg-slate-50 hover:border-indigo-400'
              }`}
            >
              <Upload size={28} className="text-indigo-500 mb-3" />
              <div className="text-sm font-semibold">
                Drag and drop your CSV or XLSX file here, or{' '}
                <span className="text-indigo-500">browse</span>
              </div>
              <div className={`mt-1.5 text-xs ${muted}`}>CSV or XLSX · Max 50MB / 2,000 rows</div>
              <input
                type="file"
                accept=".csv,.xlsx"
                className="sr-only"
                onChange={(e) => {
                  if (e.target.files?.[0]) setSelectedFile(e.target.files[0])
                }}
              />
            </label>

            {selectedFile && (
              <div className={`mt-4 flex items-center justify-between rounded-lg border p-3 ${control}`}>
                <div className="flex items-center gap-2 text-sm font-medium">
                  <FileSpreadsheet size={16} className="text-indigo-400" />
                  {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-xs text-rose-500 hover:text-rose-400"
                >
                  Remove
                </button>
              </div>
            )}
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center justify-between gap-4 pt-4 border-t">
          <div className={`flex items-center gap-4 text-xs ${muted}`}>
            <span>✓ Phone & Email PII Auto-Redacted</span>
            <span>✓ Indian Languages (Hinglish/Tanglish)</span>
            <span>✓ Safety Keyword Guard</span>
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={loading}
            className={`inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow transition hover:bg-indigo-500 disabled:opacity-50`}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> Analyzing NLP Pipeline...
              </>
            ) : (
              <>
                <Play size={16} /> Run Review Analysis
              </>
            )}
          </button>
        </div>
      </div>

      {/* Preview Table of Analyzed Reviews */}
      {analysis && (
        <section className={`mt-8 rounded-2xl border p-5 ${surface}`}>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-semibold text-base">
              Analyzed Reviews Output ({analysis.reviews.length})
            </h3>
            <span className={`text-xs ${muted}`}>
              Processing time: {analysis.meta.processing_time_ms}ms · Mode: {analysis.meta.mode}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className={`border-b text-[11px] font-semibold uppercase ${muted}`}>
                <tr>
                  <th className="pb-3 pl-2">ID</th>
                  <th className="pb-3">Rating</th>
                  <th className="pb-3">Language</th>
                  <th className="pb-3">Polarity</th>
                  <th className="pb-3">Aspects</th>
                  <th className="pb-3">Safety</th>
                  <th className="pb-3 pr-2">Sanitized Text</th>
                </tr>
              </thead>
              <tbody className={`divide-y ${dark ? 'divide-slate-800/40' : 'divide-slate-200'}`}>
                {analysis.reviews.slice(0, 10).map((rev) => (
                  <tr key={rev.review_id} className={dark ? 'hover:bg-slate-800/20' : 'hover:bg-slate-50'}>
                    <td className="py-3 pl-2 font-mono font-medium">{rev.review_id}</td>
                    <td className="py-3">
                      {rev.rating !== null ? (
                        <span className={`font-bold ${dark ? 'text-amber-400' : 'text-amber-600'}`}>{rev.rating}★</span>
                      ) : rev.pseudo_rating !== null ? (
                        <span className={`italic ${dark ? 'text-slate-400' : 'text-slate-600'}`}>~{rev.pseudo_rating}★</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3">
                      <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                        dark ? 'bg-slate-800 text-slate-200' : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        {rev.language_badge}
                      </span>
                    </td>
                    <td className="py-3">
                      <span
                        className={`rounded px-2 py-0.5 text-[10px] font-semibold ${
                          rev.polarity === 'pos'
                            ? (dark ? 'bg-emerald-500/10 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                            : rev.polarity === 'neg'
                              ? (dark ? 'bg-rose-500/10 text-rose-400' : 'bg-rose-100 text-rose-800 border border-rose-200')
                              : (dark ? 'bg-slate-700 text-slate-300' : 'bg-slate-100 text-slate-700 border border-slate-200')
                        }`}
                      >
                        {rev.polarity.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 capitalize">
                      {rev.aspects.length > 0 ? rev.aspects.join(', ') : 'general'}
                    </td>
                    <td className="py-3">
                      {rev.safety.is_alert ? (
                        <span className={`rounded px-2 py-0.5 font-bold uppercase text-[10px] ${
                          dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                        }`}>
                          {rev.safety.tier}
                        </span>
                      ) : (
                        <span className={muted}>None</span>
                      )}
                    </td>
                    <td className={`py-3 pr-2 max-w-xs truncate ${dark ? 'text-slate-300' : 'text-slate-800'}`}>
                      {rev.original_text}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  )
}

function InsightsPage({
  dark,
  surface,
  muted,
  control,
  analysis,
  onShowEvidence,
}: SharedProps & {
  analysis: AnalyzeResponse | null
  onShowEvidence: (cluster: IssueCluster) => void
}) {
  const [selectedAspect, setSelectedAspect] = useState<string>('All')
  const [selectedPolarity, setSelectedPolarity] = useState<string>('All')

  // Simulator slider states
  const [simulatorFixes, setSimulatorFixes] = useState<Record<string, number>>({})
  const [simResult, setSimResult] = useState<SimulateResponse | null>(null)
  const [simulating, setSimulating] = useState(false)

  const clusters = useMemo(() => analysis?.clusters || [], [analysis])
  const insights = useMemo(() => analysis?.insights || [], [analysis])
  const reviews = useMemo(() => analysis?.reviews || [], [analysis])

  // Compute effective fixes: top 4 insights default to 0.5 (50%) unless user adjusted
  const effectiveFixes = useMemo(() => {
    const fixes: Record<string, number> = {}
    insights.slice(0, 4).forEach((ins) => {
      fixes[ins.aspect] = simulatorFixes[ins.aspect] !== undefined ? simulatorFixes[ins.aspect] : 0.5
    })
    Object.entries(simulatorFixes).forEach(([k, v]) => {
      if (fixes[k] === undefined) {
        fixes[k] = v
      }
    })
    return fixes
  }, [insights, simulatorFixes])

  // Run simulation whenever effectiveFixes or reviews change
  useEffect(() => {
    if (!analysis || reviews.length === 0 || Object.keys(effectiveFixes).length === 0) return

    let isMounted = true

    simulateFixes(
      effectiveFixes,
      reviews.map((r) => ({
        id: r.review_id,
        text: r.original_text,
        rating: r.rating ?? r.pseudo_rating,
      }))
    )
      .then((res) => {
        if (isMounted) {
          setSimResult(res)
          setSimulating(false)
        }
      })
      .catch((err) => {
        console.error('Simulation error:', err)
        if (isMounted) setSimulating(false)
      })

    return () => {
      isMounted = false
    }
  }, [analysis, reviews, effectiveFixes])

  const aspectList = ['All', ...Array.from(new Set(clusters.map((c) => c.aspect)))]

  const filteredClusters = clusters.filter((cl) => {
    if (selectedAspect !== 'All' && cl.aspect.toLowerCase() !== selectedAspect.toLowerCase()) {
      return false
    }
    if (selectedPolarity !== 'All' && cl.polarity !== selectedPolarity) {
      return false
    }
    return true
  })

  return (
    <>
      <PageHeading
        title="Clustered Issues & Fix-This-First Prioritization"
        description="Explainable issue clusters grouped by aspect and polarity with traceable clause evidence, impact ranking, and the Fix-This-First simulator."
        muted={muted}
      />

      {/* Simulator Hero Section */}
      <section className={`mb-8 rounded-2xl border p-6 ${dark ? 'border-indigo-500/30' : 'border-indigo-200'} ${surface}`}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className={`flex items-center gap-2 text-xs font-bold uppercase tracking-wider ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
              <SlidersHorizontal size={15} /> Fix-This-First Simulator (PRD Section 7.1)
              {simulating && (
                <span className={`flex items-center gap-1 text-[11px] font-normal ${dark ? 'text-indigo-300' : 'text-indigo-600'}`}>
                  <RefreshCw className="h-3 w-3 animate-spin" /> Simulating...
                </span>
              )}
            </div>
            <h3 className="mt-1 text-xl font-bold">Simulate Projected Rating Uplift</h3>
            <p className={`mt-1 text-xs max-w-xl ${muted}`}>
              Adjust the sliders below to model expected star rating gains when fixing specific product complaints.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                const updated: Record<string, number> = {}
                insights.slice(0, 3).forEach((ins) => {
                  updated[ins.aspect] = 1.0
                })
                setSimulating(true)
                setSimulatorFixes(updated)
              }}
              className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                dark
                  ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20'
                  : 'border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100'
              }`}
            >
              Fix Top 3 (100%)
            </button>
            <button
              onClick={() => {
                const updated: Record<string, number> = {}
                insights.forEach((ins) => {
                  updated[ins.aspect] = 0.0
                })
                setSimulating(true)
                setSimulatorFixes(updated)
              }}
              className={`rounded-lg border px-3 py-1.5 text-xs ${control}`}
            >
              Reset
            </button>
          </div>
        </div>

        {/* Live Simulation Scoreboard */}
        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <div className={`rounded-xl border p-4 ${control}`}>
            <span className={`text-xs ${muted}`}>Current Baseline Rating</span>
            <div className={`mt-2 text-2xl font-bold ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
              {simResult?.current_rating ? `${simResult.current_rating.toFixed(2)} ★` : '—'}
            </div>
          </div>

          <div className={`rounded-xl border p-4 ${control}`}>
            <span className={`text-xs ${muted}`}>Projected Rating After Fixes</span>
            <div className={`mt-2 flex items-center gap-2 text-2xl font-bold ${dark ? 'text-emerald-400' : 'text-emerald-700'}`}>
              {simResult?.projected_rating ? `${simResult.projected_rating.toFixed(2)} ★` : '—'}
              {simResult && simResult.delta > 0 && (
                <span className={`rounded px-2 py-0.5 text-xs font-semibold ${
                  dark ? 'bg-emerald-500/20 text-emerald-300' : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                }`}>
                  +{simResult.delta.toFixed(2)}
                </span>
              )}
            </div>
          </div>

          <div className={`rounded-xl border p-4 ${control}`}>
            <span className={`text-xs ${muted}`}>Unhappy Reviews Resolved (≤2★)</span>
            <div className={`mt-2 text-2xl font-bold ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>
              {simResult?.unhappy_reviews_resolved ?? 0} Customers
            </div>
          </div>
        </div>

        {/* Slider Controls */}
        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          {insights.slice(0, 4).map((ins) => {
            const currentVal = effectiveFixes[ins.aspect] ?? 0.5
            return (
              <div
                key={ins.aspect}
                className={`rounded-xl border p-3.5 text-xs ${
                  dark ? 'border-slate-800 bg-slate-950/40' : 'border-slate-200 bg-slate-50/80'
                }`}
              >
                <div className="flex items-center justify-between font-semibold capitalize mb-1">
                  <span>{ins.aspect} Complaints ({ins.complaint_count})</span>
                  <span className={`font-mono font-bold ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>
                    {Math.round(currentVal * 100)}% fixed
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={currentVal}
                  onChange={(e) => {
                    setSimulating(true)
                    setSimulatorFixes((prev) => ({
                      ...prev,
                      [ins.aspect]: Number(e.target.value),
                    }))
                  }}
                  className="w-full accent-indigo-500 cursor-pointer mt-2"
                />
                <div className={`mt-2 flex justify-between text-[11px] ${muted}`}>
                  <span>Impact: -{ins.beta}★</span>
                  <span>Damage: {ins.damage} pts</span>
                </div>
              </div>
            )
          })}
        </div>

        {simResult?.notes && simResult.notes.length > 0 && (
          <div className={`mt-4 text-[11px] flex items-center gap-1.5 ${dark ? 'text-amber-400/90' : 'text-amber-800'}`}>
            <Info size={13} /> {simResult.notes.join(' · ')}
          </div>
        )}
      </section>

      {/* Filter Toolbar */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-xs font-semibold uppercase ${muted} flex items-center gap-1`}>
            <Filter size={13} /> Aspect:
          </span>
          {aspectList.map((asp) => (
            <button
              key={asp}
              onClick={() => setSelectedAspect(asp)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold capitalize transition ${
                selectedAspect === asp
                  ? 'bg-indigo-600 text-white shadow'
                  : `${control} hover:border-slate-400`
              }`}
            >
              {asp}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className={`text-xs font-semibold uppercase ${muted}`}>Polarity:</span>
          {['All', 'neg', 'pos'].map((pol) => (
            <button
              key={pol}
              onClick={() => setSelectedPolarity(pol)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold uppercase ${
                selectedPolarity === pol
                  ? 'bg-indigo-600 text-white shadow'
                  : `${control} hover:border-slate-400`
              }`}
            >
              {pol}
            </button>
          ))}
        </div>
      </div>

      {/* Clustered Issues Cards */}
      <section className="space-y-4">
        {filteredClusters.map((cl) => (
          <div
            key={cl.cluster_id}
            className={`rounded-2xl border p-5 transition ${
              dark
                ? 'border-slate-800 bg-slate-900/80 hover:border-slate-700'
                : 'border-slate-200 bg-white hover:border-slate-300'
            }`}
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-md px-2.5 py-0.5 text-xs font-bold uppercase ${
                      cl.polarity === 'neg'
                        ? (dark ? 'bg-rose-500/15 text-rose-400' : 'bg-rose-100 text-rose-800 border border-rose-200')
                        : cl.polarity === 'pos'
                          ? (dark ? 'bg-emerald-500/15 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                          : (dark ? 'bg-amber-500/15 text-amber-400' : 'bg-amber-100 text-amber-800 border border-amber-200')
                    }`}
                  >
                    {cl.aspect} · {cl.polarity}
                  </span>
                  <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${
                    dark ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                  }`}>
                    {cl.team || 'Product/Engineering'} ({cl.priority || 'P1'})
                  </span>
                  {cl.is_safety_alert && (
                    <span className={`rounded-md px-2 py-0.5 text-xs font-bold ${
                      dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                    }`}>
                      {cl.safety_tier.toUpperCase()} SAFETY ALERT
                    </span>
                  )}
                </div>

                <h3 className="text-lg font-bold pt-1">{cl.label}</h3>
                <p className={`text-xs ${muted}`}>
                  Cluster: {cl.cluster_id} · {cl.size} clauses · {cl.review_count} unique reviews
                </p>
              </div>

              <button
                onClick={() => onShowEvidence(cl)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-500"
              >
                View Evidence ({cl.size}) →
              </button>
            </div>

            <div
              className={`mt-4 rounded-xl border p-3.5 text-xs italic ${
                dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50/80'
              }`}
            >
              "{cl.example_quote}"
            </div>

            <div className={`mt-4 flex flex-wrap gap-x-6 gap-y-2 text-xs ${muted}`}>
              {cl.average_rating !== null && (
                <span>
                  Avg Rating: <strong className={dark ? 'text-amber-400' : 'text-amber-600 font-semibold'}>{cl.average_rating}★</strong>
                </span>
              )}
              <span>
                Avg Sentiment:{' '}
                <strong className={
                  cl.average_sentiment < 0
                    ? (dark ? 'text-rose-400' : 'text-rose-600 font-semibold')
                    : (dark ? 'text-emerald-400' : 'text-emerald-700 font-semibold')
                }>
                  {cl.average_sentiment > 0 ? `+${cl.average_sentiment}` : cl.average_sentiment}
                </strong>
              </span>
              <span>
                Clause IDs: <code className="font-mono text-[11px]">{cl.clause_ids.slice(0, 3).join(', ')}{cl.clause_ids.length > 3 ? '...' : ''}</code>
              </span>
            </div>
          </div>
        ))}

        {filteredClusters.length === 0 && (
          <div className={`rounded-xl border p-12 text-center text-sm ${muted} ${surface}`}>
            No clusters match the selected aspect or polarity filter.
          </div>
        )}
      </section>
    </>
  )
}

function EmergingIssuesPage({
  dark,
  surface,
  muted,
  control,
  onOpenPage,
  analysis,
  onShowEvidence,
  onInspectVision,
}: PageProps & {
  analysis: AnalyzeResponse | null
  onShowEvidence: (cluster: IssueCluster) => void
  onInspectVision?: (label: string, aspect: string) => void
}) {
  const [recentDays, setRecentDays] = useState(14)
  const [baselineDays, setBaselineDays] = useState(30)

  const [minRecentCount, setMinRecentCount] = useState(2)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [emergingData, setEmergingData] = useState<EmergingIssuesResponse | null>(null)
  const [selectedStatus, setSelectedStatus] = useState<string>('All')
  const [selectedAspect, setSelectedAspect] = useState<string>('All')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const reviews = useMemo(() => analysis?.reviews || [], [analysis])
  const clusters = useMemo(() => analysis?.clusters || [], [analysis])

  useEffect(() => {
    if (!analysis || reviews.length === 0) return

    let isMounted = true
    const timer = setTimeout(() => {
      if (isMounted) {
        setLoading(true)
        setError(null)
      }
    }, 0)

    const items: ReviewInputItem[] = reviews.map((r) => ({
      id: r.review_id,
      text: r.original_text,
      rating: r.rating ?? r.pseudo_rating,
      date: r.date,
      version: r.version,
      product: r.product,
    }))

    fetchEmergingIssues(items, recentDays, baselineDays, undefined, minRecentCount)
      .then((res) => {
        if (isMounted) {
          setEmergingData(res)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : String(err)
          setError(msg)
          setLoading(false)
        }
      })

    return () => {
      isMounted = false
      clearTimeout(timer)
    }
  }, [analysis, reviews, recentDays, baselineDays, minRecentCount])

  if (!analysis || reviews.length === 0) {
    return (
      <div className={`rounded-2xl border p-12 text-center ${surface}`}>
        <TrendingUp className="mx-auto size-12 text-indigo-400 opacity-60" />
        <h3 className="mt-4 text-lg font-bold">No Review Analysis Available</h3>
        <p className={`mt-2 text-xs max-w-md mx-auto ${muted}`}>
          Run an analysis on reviews with timestamps to detect emerging defect clusters, accelerating complaints, and critical safety hazards.
        </p>
        <button
          onClick={() => onOpenPage('Analyze Reviews')}
          className="mt-6 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow-md hover:bg-indigo-500"
        >
          Go to Analyze Reviews →
        </button>
      </div>
    )
  }

  const findings: EmergingIssueFinding[] = emergingData?.findings || []
  const timeWindows = emergingData?.time_windows

  const aspectList = ['All', ...Array.from(new Set(findings.map((f) => f.aspect)))]

  const filteredFindings = findings.filter((f) => {
    if (selectedStatus === 'new' && f.status !== 'new') return false
    if (selectedStatus === 'accelerating' && f.status !== 'accelerating') return false
    if (selectedStatus === 'safety' && !f.is_safety_escalated) return false
    if (selectedStatus === 'insufficient' && f.status !== 'insufficient_data') return false
    if (selectedAspect !== 'All' && f.aspect.toLowerCase() !== selectedAspect.toLowerCase()) return false
    return true
  })

  return (
    <>
      <PageHeading
        title="Emerging Issues & Trend Velocity"
        description="Detect newly surfaced defect patterns, accelerating complaints, and prioritized safety escalations by comparing recent and baseline time windows."
        muted={muted}
      />

      {error && (
        <div className="mb-6 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-400 flex items-center justify-between">
          <span>{error}</span>
          <button
            onClick={() => {
              if (analysis && reviews.length > 0) {
                setLoading(true)
                const items: ReviewInputItem[] = reviews.map((r) => ({
                  id: r.review_id,
                  text: r.original_text,
                  rating: r.rating ?? r.pseudo_rating,
                  date: r.date,
                  version: r.version,
                  product: r.product,
                }))
                fetchEmergingIssues(items, recentDays, baselineDays, undefined, minRecentCount)
                  .then((res) => {
                    setEmergingData(res)
                    setLoading(false)
                  })
                  .catch(() => setLoading(false))
              }
            }}
            className="underline font-bold"
          >
            Retry
          </button>
        </div>
      )}

      {/* Scoreboard Cards */}
      <section className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className={`rounded-2xl border p-5 ${surface}`}>
          <div className="flex items-center justify-between">
            <span className={`text-xs font-semibold ${muted}`}>Total Emerging Alerts</span>
            <TrendingUp size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
          </div>
          <div className={`mt-3 text-3xl font-extrabold ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>
            {emergingData?.total_emerging_issues ?? 0}
          </div>
          <div className={`mt-1 text-[11px] ${muted}`}>New or accelerating trends</div>
        </div>

        <div className={`rounded-2xl border p-5 ${surface}`}>
          <div className="flex items-center justify-between">
            <span className={`text-xs font-semibold ${muted}`}>New Issues</span>
            <Sparkles size={16} className={dark ? 'text-emerald-400' : 'text-emerald-700'} />
          </div>
          <div className={`mt-3 text-3xl font-extrabold ${dark ? 'text-emerald-400' : 'text-emerald-700'}`}>
            {emergingData?.new_issues_count ?? 0}
          </div>
          <div className={`mt-1 text-[11px] ${muted}`}>0 in baseline, active recently</div>
        </div>

        <div className={`rounded-2xl border p-5 ${surface}`}>
          <div className="flex items-center justify-between">
            <span className={`text-xs font-semibold ${muted}`}>Accelerating Issues</span>
            <Flame size={16} className={dark ? 'text-amber-400' : 'text-amber-600'} />
          </div>
          <div className={`mt-3 text-3xl font-extrabold ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
            {emergingData?.accelerating_issues_count ?? 0}
          </div>
          <div className={`mt-1 text-[11px] ${muted}`}>Surging complaint share</div>
        </div>

        <div className={`rounded-2xl border p-5 ${surface}`}>
          <div className="flex items-center justify-between">
            <span className={`text-xs font-semibold ${muted}`}>Safety Escalations</span>
            <ShieldAlert size={16} className={dark ? 'text-rose-400' : 'text-rose-600'} />
          </div>
          <div className={`mt-3 text-3xl font-extrabold ${dark ? 'text-rose-400' : 'text-rose-600'}`}>
            {emergingData?.safety_escalations_count ?? 0}
          </div>
          <div className={`mt-1 text-[11px] ${muted}`}>Prioritized regardless of frequency</div>
        </div>
      </section>

      {/* Timeline Controls & Settings */}
      <section className={`mb-8 rounded-2xl border p-6 ${surface}`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
              <Calendar size={15} /> Time Window Configuration
              {loading && <RefreshCw size={12} className="animate-spin text-indigo-400 ml-1" />}
            </div>
            <h3 className="mt-1 text-lg font-bold">Configurable Recent vs. Baseline Windows</h3>
            {timeWindows && (
              <div className={`mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs ${muted}`}>
                <span>Recent: <strong>{timeWindows.recent_start || '—'}</strong> to <strong>{timeWindows.recent_end || '—'}</strong> ({timeWindows.total_recent_reviews} reviews)</span>
                <span>Baseline: <strong>{timeWindows.baseline_start || '—'}</strong> to <strong>{timeWindows.baseline_end || '—'}</strong> ({timeWindows.total_baseline_reviews} reviews)</span>
                {timeWindows.undated_reviews > 0 && <span>Undated: {timeWindows.undated_reviews} reviews</span>}
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <label htmlFor="recent-days-input" className={`text-xs ${muted}`}>Recent Days:</label>
              <input
                id="recent-days-input"
                type="number"
                min="1"
                max="90"
                value={recentDays}
                onChange={(e) => setRecentDays(Number(e.target.value) || 14)}
                className={`w-16 rounded-lg border px-2 py-1 text-xs font-semibold ${control}`}
              />
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="baseline-days-input" className={`text-xs ${muted}`}>Baseline Days:</label>
              <input
                id="baseline-days-input"
                type="number"
                min="1"
                max="180"
                value={baselineDays}
                onChange={(e) => setBaselineDays(Number(e.target.value) || 30)}
                className={`w-16 rounded-lg border px-2 py-1 text-xs font-semibold ${control}`}
              />
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="min-recent-count-input" className={`text-xs ${muted}`}>Min Count:</label>
              <input
                id="min-recent-count-input"
                type="number"
                min="1"
                max="10"
                value={minRecentCount}
                onChange={(e) => setMinRecentCount(Number(e.target.value) || 2)}
                className={`w-14 rounded-lg border px-2 py-1 text-xs font-semibold ${control}`}
              />
            </div>
          </div>
        </div>

        {timeWindows && !timeWindows.has_sufficient_timeline && (
          <div className={`mt-4 rounded-xl border p-3.5 text-xs flex items-start gap-2 ${
            dark ? 'border-amber-500/30 bg-amber-500/10 text-amber-300' : 'border-amber-200 bg-amber-50/70 text-amber-900'
          }`}>
            <Info size={16} className={`shrink-0 mt-0.5 ${dark ? 'text-amber-400' : 'text-amber-700'}`} />
            <div>
              <strong>Missing or insufficient review timestamps.</strong> Historical baseline comparison requires reviews with valid date metadata.
              InsightForge honestly displays unconfirmed status rather than fabricating trends; critical safety hazards remain actively escalated.
            </div>
          </div>
        )}
      </section>

      {/* Filter Toolbar */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-xs font-semibold uppercase ${muted}`}>Filter Status:</span>
          {[
            { id: 'All', label: 'All Findings' },
            { id: 'new', label: 'New Issues' },
            { id: 'accelerating', label: 'Accelerating' },
            { id: 'safety', label: 'Safety Escalations' },
            { id: 'insufficient', label: 'Insufficient Data' },
          ].map((st) => (
            <button
              key={st.id}
              onClick={() => setSelectedStatus(st.id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                selectedStatus === st.id
                  ? 'bg-indigo-600 text-white shadow'
                  : `${control} hover:border-slate-400`
              }`}
            >
              {st.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-xs font-semibold uppercase ${muted} flex items-center gap-1`}>
            <Filter size={13} /> Aspect:
          </span>
          {aspectList.map((asp) => (
            <button
              key={asp}
              onClick={() => setSelectedAspect(asp)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold capitalize transition ${
                selectedAspect === asp
                  ? 'bg-indigo-600 text-white shadow'
                  : `${control} hover:border-slate-400`
              }`}
            >
              {asp}
            </button>
          ))}
        </div>
      </div>

      {/* Findings List */}
      <section className="space-y-4">
        {filteredFindings.map((finding) => {
          const isExpanded = expandedId === finding.issue_id
          const matchingCluster = clusters.find((c) => c.cluster_id === finding.issue_id || c.aspect === finding.aspect)

          return (
            <div
              key={finding.issue_id}
              className={`rounded-2xl border p-5 transition ${
                dark
                  ? 'border-slate-800 bg-slate-900/80 hover:border-slate-700'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    {finding.is_safety_escalated && (
                      <span className={`rounded-md px-2.5 py-0.5 text-xs font-bold flex items-center gap-1 ${
                        dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                      }`}>
                        <ShieldAlert size={13} /> CRITICAL SAFETY ESCALATION (P{finding.priority})
                      </span>
                    )}

                    {finding.status === 'new' && (
                      <span className={`rounded-md px-2.5 py-0.5 text-xs font-bold flex items-center gap-1 ${
                        dark ? 'bg-emerald-500/20 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                      }`}>
                        <Sparkles size={13} /> NEW DEFECT ISSUE (P{finding.priority})
                      </span>
                    )}

                    {finding.status === 'accelerating' && (
                      <span className={`rounded-md px-2.5 py-0.5 text-xs font-bold flex items-center gap-1 ${
                        dark ? 'bg-amber-500/20 text-amber-400' : 'bg-amber-100 text-amber-800 border border-amber-200'
                      }`}>
                        <Flame size={13} /> ACCELERATING
                        {finding.velocity_ratio ? ` · ${finding.velocity_ratio}x VELOCITY` : ''} (P{finding.priority})
                      </span>
                    )}

                    {finding.status === 'insufficient_data' && (
                      <span className={`rounded-md px-2.5 py-0.5 text-xs font-semibold ${
                        dark ? 'bg-slate-500/20 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        INSUFFICIENT DATA · UNCONFIRMED
                      </span>
                    )}

                    {finding.status === 'steady' && (
                      <span className={`rounded-md px-2.5 py-0.5 text-xs font-medium ${
                        dark ? 'bg-slate-500/10 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        STEADY
                      </span>
                    )}

                    <span className={`rounded-md px-2 py-0.5 text-xs font-medium capitalize ${
                      dark ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                    }`}>
                      {finding.category} · {finding.polarity}
                    </span>

                    <span className={`text-[11px] ${muted}`}>
                      Confidence: <strong className="capitalize">{finding.confidence}</strong>
                    </span>
                  </div>

                  <h3 className="text-lg font-bold pt-1">{finding.label}</h3>

                  <div className={`flex flex-wrap gap-x-4 gap-y-1 text-xs ${muted}`}>
                    <span>First Seen: <strong>{finding.first_observed_date || 'Unknown'}</strong></span>
                    <span>Recent Count: <strong className={dark ? 'text-indigo-400' : 'text-indigo-600'}>{finding.recent_count}</strong> ({Math.round(finding.recent_share * 100)}% share)</span>
                    <span>Baseline Count: <strong>{finding.baseline_count}</strong> ({Math.round(finding.baseline_share * 100)}% share)</span>
                    <span>Change: <strong className={finding.change_count > 0 ? (dark ? 'text-amber-400' : 'text-amber-600') : ''}>{finding.change_count >= 0 ? `+${finding.change_count}` : finding.change_count}</strong></span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : finding.issue_id)}
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${control}`}
                  >
                    {isExpanded ? 'Hide Evidence' : `Evidence (${finding.evidence.length})`}
                  </button>

                  {matchingCluster && (
                    <button
                      onClick={() => onShowEvidence(matchingCluster)}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow hover:bg-indigo-500"
                    >
                      Cluster View <ArrowUpRight size={13} />
                    </button>
                  )}

                  <button
                    onClick={() => onInspectVision?.(finding.label, finding.aspect)}
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${control}`}
                    title="Upload or inspect photo defect evidence"
                  >
                    <Camera size={13} className="text-indigo-400" /> Photo Defect
                  </button>
                </div>

              </div>

              {/* Limitations alerts */}
              {finding.limitations.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {finding.limitations.map((lim, idx) => (
                    <span
                      key={idx}
                      className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] ${
                        dark ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      <Info size={11} className="text-amber-400" /> {lim}
                    </span>
                  ))}
                </div>
              )}

              {/* Expandable Evidence Drawer */}
              {isExpanded && (
                <div className={`mt-4 rounded-xl border p-4 space-y-3 ${
                  dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50/80'
                }`}>
                  <div className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                    <FileText size={13} /> Traceable Clause Evidence ({finding.evidence.length} samples)
                  </div>
                  {finding.evidence.map((ev, idx) => (
                    <div
                      key={idx}
                      className={`rounded-lg border p-3 text-xs ${
                        dark ? 'border-slate-800 bg-slate-900/90' : 'border-slate-200 bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[11px] mb-1">
                        <span className={`font-mono font-semibold ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>Review {ev.review_id} (Clause {ev.clause_id})</span>
                        <div className="flex items-center gap-2">
                          {ev.date && <span>{ev.date}</span>}
                          {ev.rating !== null && <span className={`font-bold ${dark ? 'text-amber-400' : 'text-amber-600'}`}>{ev.rating}★</span>}
                          {ev.safety_tier !== 'none' && (
                            <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                              dark ? 'bg-rose-500/20 text-rose-400' : 'bg-rose-100 text-rose-700 border border-rose-200'
                            }`}>
                              {ev.safety_tier}
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="italic">"{ev.text}"</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}

        {filteredFindings.length === 0 && (
          <div className={`rounded-xl border p-12 text-center text-sm ${muted} ${surface}`}>
            No emerging findings match the current status and aspect filters.
          </div>
        )}
      </section>
    </>
  )
}

function DefectVisionPage({
  dark,
  surface,
  muted,
  control,
  analysis,
  initialContext,
  gallery: externalGallery,
  onNewFinding,
  onShowEvidence,
}: PageProps & {
  analysis: AnalyzeResponse | null
  initialContext?: { cluster?: string; aspect?: string } | null
  gallery?: ImageDefectAnalysis[]
  onNewFinding?: (finding: ImageDefectAnalysis) => void
  onShowEvidence?: (cluster: IssueCluster) => void
}) {
  const [health, setHealth] = useState<VisionHealthResponse | null>(null)
  const [loadingHealth, setLoadingHealth] = useState(false)

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [clusterContext, setClusterContext] = useState<string>(initialContext?.cluster || '')
  const [aspectContext, setAspectContext] = useState<string>(initialContext?.aspect || '')

  const [analyzing, setAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [currentResult, setCurrentResult] = useState<ImageDefectAnalysis | null>(null)
  const [gallery, setGallery] = useState<ImageDefectAnalysis[]>(externalGallery || [])

  const checkVision = async () => {
    setLoadingHealth(true)
    try {
      const data = await fetchVisionHealth()
      setHealth(data)
    } catch {
      setHealth({
        status: 'unavailable',
        provider: 'ollama',
        base_url: 'http://localhost:11434',
        model: 'gemma3:4b',
        vision_supported: false,
        message: 'Ollama is unreachable at http://localhost:11434. Ensure Ollama is running locally via `ollama serve`.',
        available_models: [],
      })
    } finally {
      setLoadingHealth(false)
    }
  }

  useEffect(() => {
    let isMounted = true
    fetchVisionHealth()
      .then((data) => {
        if (isMounted) setHealth(data)
      })
      .catch(() => {
        if (isMounted) {
          setHealth({
            status: 'unavailable',
            provider: 'ollama',
            base_url: 'http://localhost:11434',
            model: 'gemma3:4b',
            vision_supported: false,
            message: 'Ollama is unreachable at http://localhost:11434. Ensure Ollama is running locally via `ollama serve`.',
            available_models: [],
          })
        }
      })

    return () => {
      isMounted = false
    }
  }, [])


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setAnalysisError(null)

    if (file.size > 10 * 1024 * 1024) {
      setAnalysisError('File size exceeds the 10 MB maximum limit.')
      return
    }

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setAnalysisError('Invalid file format. Please upload a JPEG, PNG, or WebP image.')
      return
    }

    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setAnalysisError(null)
  }

  const handleRunAnalysis = async () => {
    if (!selectedFile) {
      setAnalysisError('Please select a product image first.')
      return
    }

    setAnalyzing(true)
    setAnalysisError(null)

    try {
      const res = await analyzeProductImage(
        selectedFile,
        clusterContext.trim() || undefined,
        aspectContext.trim() || undefined,
      )
      setCurrentResult(res)
      setGallery((prev) => [res, ...prev.filter((item) => item.image_id !== res.image_id)])
      onNewFinding?.(res)
    } catch (err) {
      setAnalysisError(err instanceof Error ? err.message : 'Defect vision inference failed.')
    } finally {
      setAnalyzing(false)
    }
  }

  const clusters = useMemo(() => analysis?.clusters || [], [analysis])
  const matchingCluster = useMemo(() => {
    if (!currentResult?.relevant_issue_cluster) return null
    return clusters.find((c) => c.label.toLowerCase() === currentResult.relevant_issue_cluster?.toLowerCase())
  }, [clusters, currentResult])

  const getSeverityBadgeClass = (sev: string) => {
    switch (sev.toLowerCase()) {
      case 'critical':
        return dark
          ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
          : 'bg-rose-100 text-rose-800 border border-rose-200'
      case 'high':
        return dark
          ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
          : 'bg-orange-100 text-orange-800 border border-orange-200'
      case 'medium':
        return dark
          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
          : 'bg-amber-100 text-amber-800 border border-amber-200'
      case 'low':
        return dark
          ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
          : 'bg-blue-100 text-blue-800 border border-blue-200'
      default:
        return dark
          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
          : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
    }
  }

  return (
    <>
      <PageHeading
        title="Visual Defect Evidence"
        description="Corroborate customer review complaints with physical product photos using local vision inference (Ollama gemma3:4b). Grounded strictly in observable features."
        muted={muted}
      />

      {/* Ollama Health Status Bar */}
      <section className={`mb-6 rounded-2xl border p-4.5 ${surface}`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400">
              <Eye size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm">Ollama Vision Inference</span>
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                    health?.status === 'ready'
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      : health?.status === 'model_not_installed'
                        ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                        : health?.status === 'vision_unsupported'
                          ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                          : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                  }`}
                >
                  <span
                    className={`size-2 rounded-full ${
                      health?.status === 'ready'
                        ? 'bg-emerald-400 animate-pulse'
                        : health?.status === 'model_not_installed'
                          ? 'bg-amber-400'
                          : 'bg-rose-400'
                    }`}
                  />
                  {health?.status === 'ready'
                    ? `Ready (${health.model})`
                    : health?.status === 'model_not_installed'
                      ? `Model Missing (${health.model})`
                      : health?.status === 'vision_unsupported'
                        ? 'Vision Unsupported'
                        : 'Ollama Offline'}
                </span>
              </div>
              <p className={`mt-0.5 text-xs ${muted}`}>
                Provider: <strong>{health?.provider || 'ollama'}</strong> · Endpoint: <code>{health?.base_url || 'http://localhost:11434'}</code> · Model: <code>{health?.model || 'gemma3:4b'}</code>
              </p>
            </div>
          </div>

          <button
            onClick={checkVision}
            disabled={loadingHealth}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold ${control} hover:border-slate-400`}
            title="Check Ollama service status"
          >
            <RefreshCw size={13} className={loadingHealth ? 'animate-spin' : ''} />
            {loadingHealth ? 'Checking...' : 'Refresh Health'}
          </button>
        </div>

        {health && health.status !== 'ready' && (
          <div className={`mt-4 rounded-xl border p-4 text-xs ${
            dark ? 'border-amber-500/30 bg-amber-500/5' : 'border-amber-200 bg-amber-50/70'
          }`}>
            <div className={`flex items-center gap-2 font-bold ${dark ? 'text-amber-400' : 'text-amber-800'}`}>
              <Info size={15} /> Local Setup Instructions for Ollama Vision
            </div>
            <p className={`mt-1 ${dark ? 'text-slate-300' : 'text-slate-800'}`}>
              {health.message}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className={`text-[11px] font-semibold ${muted}`}>Run in your local terminal:</span>
              <code className={`rounded px-2.5 py-1 font-mono ${dark ? 'bg-black/50 text-emerald-400' : 'bg-slate-100 text-emerald-800 border border-slate-200'}`}>ollama serve</code>
              <code className={`rounded px-2.5 py-1 font-mono ${dark ? 'bg-black/50 text-emerald-400' : 'bg-slate-100 text-emerald-800 border border-slate-200'}`}>ollama pull gemma3:4b</code>
            </div>
            <div className={`mt-2.5 text-[11px] ${muted}`}>
              <strong>Deployment Architecture Note:</strong> An Ollama instance on a developer machine is unreachable from cloud hosting via localhost. In production, configure <code>OLLAMA_BASE_URL</code> to point to a secured remote inference host.
            </div>
          </div>
        )}
      </section>

      {/* Main Analysis Workspace */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Image Upload & Context Form (5 cols) */}
        <div className="space-y-6 lg:col-span-5">
          <section className={`rounded-2xl border p-5 ${surface}`}>
            <h3 className="text-base font-bold flex items-center gap-2">
              <Camera size={18} className="text-indigo-400" /> Upload Product Photo
            </h3>
            <p className={`mt-1 text-xs ${muted}`}>
              Upload a customer product image (JPEG, PNG, WebP up to 10 MB) to inspect visible defect evidence.
            </p>

            {/* Dropzone */}
            <div className="mt-4">
              {previewUrl ? (
                <div className={`relative overflow-hidden rounded-xl border p-2 ${dark ? 'border-slate-700 bg-black/30' : 'border-slate-200 bg-slate-100/60'}`}>
                  <img
                    src={previewUrl}
                    alt="Upload Preview"
                    className={`max-h-56 w-full rounded-lg object-contain ${dark ? 'bg-black/40' : 'bg-white'}`}
                  />
                  <button
                    onClick={handleClearFile}
                    className="absolute top-4 right-4 rounded-full bg-black/70 p-1.5 text-white shadow hover:bg-black"
                    title="Remove image"
                  >
                    <X size={15} />
                  </button>
                  <div className={`mt-2 flex items-center justify-between px-1 text-[11px] ${muted}`}>
                    <span className="truncate font-mono">{selectedFile?.name}</span>
                    <span>{selectedFile ? `${Math.round(selectedFile.size / 1024)} KB` : ''}</span>
                  </div>
                </div>
              ) : (
                <label className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition ${
                  dark ? 'border-slate-700/80 bg-slate-900/20 hover:border-indigo-500/60' : 'border-slate-300 bg-slate-50 hover:border-indigo-400'
                }`}>
                  <Upload size={32} className="text-indigo-400/80 mb-2" />
                  <span className="text-xs font-semibold">Click to select photo or drag and drop</span>
                  <span className={`mt-1 text-[11px] ${muted}`}>JPEG, PNG, or WebP (max 10 MB)</span>
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>
              )}
            </div>

            {/* Context Linking */}
            <div className="mt-4 space-y-3">
              <div>
                <label className={`block text-xs font-semibold ${muted}`}>
                  Link to Issue Cluster (Optional)
                </label>
                {clusters.length > 0 ? (
                  <select
                    value={clusterContext}
                    onChange={(e) => {
                      setClusterContext(e.target.value)
                      const matched = clusters.find((c) => c.label === e.target.value)
                      if (matched?.aspect) setAspectContext(matched.aspect)
                    }}
                    className={`mt-1 w-full rounded-lg border p-2 text-xs ${control}`}
                  >
                    <option value="">-- No issue linked --</option>
                    {clusters.map((c) => (
                      <option key={c.cluster_id} value={c.label}>
                        {c.label} ({c.aspect} · {c.review_count} reviews)
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={clusterContext}
                    onChange={(e) => setClusterContext(e.target.value)}
                    placeholder="e.g. Scratched casing, cracked bezel"
                    className={`mt-1 w-full rounded-lg border p-2 text-xs ${control}`}
                  />
                )}
              </div>

              <div>
                <label className={`block text-xs font-semibold ${muted}`}>
                  Product Aspect / Component (Optional)
                </label>
                <input
                  type="text"
                  value={aspectContext}
                  onChange={(e) => setAspectContext(e.target.value)}
                  placeholder="e.g. build_quality, hardware, packaging"
                  className={`mt-1 w-full rounded-lg border p-2 text-xs ${control}`}
                />
              </div>
            </div>

            {analysisError && (
              <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-400">
                {analysisError}
              </div>
            )}

            <button
              onClick={handleRunAnalysis}
              disabled={!selectedFile || analyzing}
              className={`mt-5 w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white shadow-md transition ${
                !selectedFile || analyzing
                  ? 'opacity-50 cursor-not-allowed'
                  : 'hover:bg-indigo-500'
              }`}
            >
              {analyzing ? (
                <>
                  <RefreshCw size={14} className="animate-spin" />
                  Inspecting Defect with Ollama...
                </>
              ) : (
                <>
                  <Eye size={14} />
                  Inspect Defect with Ollama
                </>
              )}
            </button>
          </section>

          {/* Guidelines box */}
          <div className={`rounded-xl border p-4 text-xs ${surface}`}>
            <h4 className={`font-semibold flex items-center gap-1.5 ${dark ? 'text-slate-300' : 'text-slate-900'}`}>
              <CheckCircle2 size={14} className={dark ? 'text-emerald-400' : 'text-emerald-700'} /> Factual Quality Inspection Protocol
            </h4>
            <ul className={`mt-2 space-y-1.5 text-[11px] ${muted}`}>
              <li>• <strong>Visual Evidence Only:</strong> The model evaluates observable physical features.</li>
              <li>• <strong>No Unsubstantiated Claims:</strong> Hidden electrical faults, supplier blame, or internal causes are never assumed.</li>
              <li>• <strong>Normal Wear vs Defect:</strong> If no defect is visible, category is recorded as none_detected.</li>
              <li>• <strong>Local Privacy:</strong> Images are never sent to third-party cloud AI APIs.</li>
            </ul>
          </div>
        </div>

        {/* Right Column: Structured Findings (7 cols) */}
        <div className="space-y-6 lg:col-span-7">
          {analyzing ? (
            <div className={`rounded-2xl border p-12 text-center ${surface}`}>
              <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-400 animate-pulse">
                <Camera size={28} />
              </div>
              <h3 className="mt-4 text-base font-bold">Analyzing Product Image</h3>
              <p className={`mx-auto mt-2 max-w-sm text-xs ${muted}`}>
                Ollama vision model is performing structured physical feature extraction and observable defect detection...
              </p>
            </div>
          ) : currentResult ? (
            <section className={`rounded-2xl border p-6 ${surface}`}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`rounded-md px-2.5 py-0.5 text-xs font-bold uppercase ${getSeverityBadgeClass(
                        currentResult.severity,
                      )}`}
                    >
                      {currentResult.severity} SEVERITY
                    </span>

                    <span
                      className={`rounded-md px-2.5 py-0.5 text-xs font-semibold capitalize ${
                        currentResult.is_defect_observed
                          ? (dark ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' : 'bg-rose-100 text-rose-800 border border-rose-200')
                          : (dark ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                      }`}
                    >
                      {currentResult.is_defect_observed ? 'Defect Identified' : 'No Visible Defect'}
                    </span>

                    <span className={`rounded-md px-2.5 py-0.5 text-xs font-medium ${
                      dark ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                    }`}>
                      Category: <strong>{currentResult.potential_defect_category}</strong>
                    </span>
                  </div>

                  <h3 className="mt-2 text-xl font-bold">{currentResult.filename}</h3>
                  <div className={`mt-1 flex flex-wrap gap-x-3 text-[11px] ${muted}`}>
                    <span>Image ID: <code>{currentResult.image_id}</code></span>
                    <span>Confidence: <strong>{Math.round(currentResult.confidence * 100)}%</strong></span>
                    {currentResult.relevant_aspect && (
                      <span>Aspect: <strong>{currentResult.relevant_aspect}</strong></span>
                    )}
                  </div>
                </div>

                {matchingCluster && onShowEvidence && (
                  <button
                    onClick={() => onShowEvidence(matchingCluster)}
                    className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow hover:bg-indigo-500"
                  >
                    View Clustered Evidence <ArrowUpRight size={13} />
                  </button>
                )}
              </div>

              {/* Side-by-Side: Image preview + Visual Description */}
              <div className={`mt-5 grid gap-4 sm:grid-cols-12 rounded-xl border p-4 ${
                dark ? 'border-slate-700/60 bg-black/20' : 'border-slate-200 bg-slate-50'
              }`}>
                <div className="sm:col-span-5 flex items-center justify-center">
                  <img
                    src={currentResult.thumbnail_data_url || previewUrl || ''}
                    alt="Analyzed defect thumbnail"
                    className={`max-h-48 rounded-lg object-contain border ${
                      dark ? 'border-slate-800 bg-black/40' : 'border-slate-200 bg-white'
                    }`}
                  />
                </div>
                <div className="sm:col-span-7 flex flex-col justify-center">
                  <span className={`text-xs font-semibold uppercase ${muted}`}>Visual Description</span>
                  <p className="mt-1 text-sm font-medium leading-relaxed">
                    "{currentResult.visual_description}"
                  </p>
                  {currentResult.relevant_issue_cluster && (
                    <div className={`mt-3 text-xs font-medium ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                      Linked Issue: <strong>{currentResult.relevant_issue_cluster}</strong>
                    </div>
                  )}
                </div>
              </div>

              {/* Exact Visual Observations */}
              <div className="mt-6">
                <h4 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${
                  dark ? 'text-slate-300' : 'text-slate-900'
                }`}>
                  <Check size={14} className={dark ? 'text-emerald-400' : 'text-emerald-700'} /> Directly Observable Visual Evidence
                </h4>
                <div className="mt-2 space-y-2">
                  {currentResult.visual_observations.map((obs, idx) => (
                    <div
                      key={idx}
                      className={`flex items-start gap-2.5 rounded-xl border p-3 text-xs ${
                        dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50/80'
                      }`}
                    >
                      <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-bold ${
                        dark ? 'bg-emerald-500/20 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                      }`}>
                        #{idx + 1}
                      </span>
                      <span className="leading-relaxed">{obs}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommended Physical Verification Steps */}
              {currentResult.suggested_manual_checks.length > 0 && (
                <div className="mt-6">
                  <h4 className={`text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 ${
                    dark ? 'text-slate-300' : 'text-slate-900'
                  }`}>
                    <SlidersHorizontal size={14} className={dark ? 'text-indigo-400' : 'text-indigo-700'} /> Suggested Manual Quality Checks
                  </h4>
                  <div className="mt-2 space-y-1.5">
                    {currentResult.suggested_manual_checks.map((chk, idx) => (
                      <div
                        key={idx}
                        className={`rounded-lg border px-3 py-2 text-xs ${
                          dark ? 'border-slate-800 bg-slate-900/40' : 'border-slate-200 bg-white'
                        }`}
                      >
                        • {chk}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Limitations & Caveats */}
              {currentResult.limitations.length > 0 && (
                <div className={`mt-6 rounded-xl border p-4 ${
                  dark ? 'border-amber-500/20 bg-amber-500/5' : 'border-amber-200 bg-amber-50/70'
                }`}>
                  <div className={`flex items-center gap-2 text-xs font-bold uppercase tracking-wider ${dark ? 'text-amber-400' : 'text-amber-700'}`}>
                    <AlertTriangle size={14} /> Observational Limitations & Caveats
                  </div>
                  <ul className={`mt-2 space-y-1 text-xs ${muted}`}>
                    {currentResult.limitations.map((lim, idx) => (
                      <li key={idx}>• {lim}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Non-speculation declaration */}
              <div className={`mt-5 border-t pt-3 flex items-center justify-between text-[11px] ${muted}`}>
                <span>Strict Grounding: No unobservable causes claimed</span>
                <span className={`font-semibold ${dark ? 'text-emerald-400' : 'text-emerald-700'}`}>✓ Verified Factual</span>
              </div>
            </section>
          ) : (
            <div className={`rounded-2xl border p-12 text-center ${surface}`}>
              <div className={`mx-auto flex size-14 items-center justify-center rounded-2xl ${dark ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-50 text-indigo-600'}`}>
                <ImageIcon size={28} />
              </div>
              <h3 className="mt-4 text-base font-bold">No Image Evidence Inspected Yet</h3>
              <p className={`mx-auto mt-2 max-w-sm text-xs ${muted}`}>
                Select a product photo and optional issue cluster context on the left, then click "Inspect Defect with Ollama" to run local vision analysis.
              </p>
            </div>
          )}

          {/* Session Gallery */}
          {gallery.length > 1 && (
            <section className={`rounded-2xl border p-5 ${surface}`}>
              <h4 className="text-sm font-bold flex items-center gap-2 mb-3">
                <History size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Analyzed In This Session ({gallery.length})
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {gallery.map((item) => (
                  <button
                    key={item.image_id}
                    onClick={() => setCurrentResult(item)}
                    className={`rounded-xl border p-2 text-left transition ${
                      currentResult?.image_id === item.image_id
                        ? 'border-indigo-500 ring-2 ring-indigo-500/30 bg-indigo-500/10'
                        : `${control} hover:border-slate-500`
                    }`}
                  >
                    <img
                      src={item.thumbnail_data_url || ''}
                      alt={item.filename}
                      className={`h-24 w-full rounded-lg object-contain mb-2 ${dark ? 'bg-black/40' : 'bg-slate-100'}`}
                    />
                    <div className="text-[11px] font-bold truncate">{item.filename}</div>
                    <div className="flex items-center justify-between mt-1 text-[10px]">
                      <span className={`capitalize ${dark ? 'text-slate-400' : 'text-slate-600'}`}>{item.potential_defect_category}</span>
                      <span
                        className={`font-bold ${
                          item.is_defect_observed
                            ? (dark ? 'text-rose-400' : 'text-rose-600')
                            : (dark ? 'text-emerald-400' : 'text-emerald-700')
                        }`}
                      >
                        {item.severity}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </>
  )
}

function ComparePage({
  dark,
  surface,
  muted,
  control,
  onOpenPage,
  analysis,
  onAnalysisComplete,
  onShowEvidence,
}: PageProps & {
  analysis: AnalyzeResponse | null
  onAnalysisComplete: (res: AnalyzeResponse) => void
  onShowEvidence?: (cluster: IssueCluster) => void
}) {
  const [activeTab, setActiveTab] = useState<'versions' | 'products'>('versions')

  // Version comparison state
  const [selectedProduct, setSelectedProduct] = useState<string>('')
  const [versionA, setVersionA] = useState<string>('')
  const [versionB, setVersionB] = useState<string>('')
  const [loadingVersion, setLoadingVersion] = useState(false)
  const [versionError, setVersionError] = useState<string | null>(null)
  const [versionResult, setVersionResult] = useState<VersionCompareResponse | null>(null)
  const [expandedCluster, setExpandedCluster] = useState<string | null>(null)

  // Product comparison state
  const [productA, setProductA] = useState<string>('')
  const [productB, setProductB] = useState<string>('')
  const [focusAspect, setFocusAspect] = useState<string>('All')
  const [loadingProduct, setLoadingProduct] = useState(false)
  const [productError, setProductError] = useState<string | null>(null)
  const [productResult, setProductResult] = useState<ProductCompareResponse | null>(null)

  const [loadingSample, setLoadingSample] = useState(false)

  const reviews = useMemo(() => analysis?.reviews || [], [analysis])

  // Extract products and versions from current real analyzed data
  const availableProducts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const r of reviews) {
      if (r.product && r.product.trim()) {
        const p = r.product.trim()
        counts[p] = (counts[p] || 0) + 1
      }
    }
    return Object.entries(counts).map(([name, count]) => ({ name, count }))
  }, [reviews])

  const availableVersions = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const r of reviews) {
      if (r.version && r.version.trim()) {
        const v = r.version.trim()
        counts[v] = (counts[v] || 0) + 1
      }
    }
    return Object.entries(counts).map(([version, count]) => ({ version, count }))
  }, [reviews])

  const versionsForSelectedProduct = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const r of reviews) {
      const pMatch = !selectedProduct || (r.product && r.product.trim().toLowerCase() === selectedProduct.toLowerCase())
      if (pMatch && r.version && r.version.trim()) {
        const v = r.version.trim()
        counts[v] = (counts[v] || 0) + 1
      }
    }
    return Object.entries(counts).map(([version, count]) => ({ version, count }))
  }, [reviews, selectedProduct])

  // Auto-init defaults asynchronously
  useEffect(() => {
    if (availableProducts.length > 0 && !selectedProduct) {
      const timer = setTimeout(() => setSelectedProduct(availableProducts[0].name), 0)
      return () => clearTimeout(timer)
    }
  }, [availableProducts, selectedProduct])

  useEffect(() => {
    if (versionsForSelectedProduct.length >= 2) {
      const v0 = versionsForSelectedProduct[0].version
      const v1 = versionsForSelectedProduct[1].version
      const timer = setTimeout(() => {
        if (!versionA || !versionsForSelectedProduct.some((v) => v.version === versionA)) setVersionA(v0)
        if (!versionB || !versionsForSelectedProduct.some((v) => v.version === versionB) || versionB === v0) setVersionB(v1)
      }, 0)
      return () => clearTimeout(timer)
    }
  }, [versionsForSelectedProduct, versionA, versionB])

  useEffect(() => {
    if (availableProducts.length >= 2) {
      const p0 = availableProducts[0].name
      const p1 = availableProducts[1].name
      const timer = setTimeout(() => {
        if (!productA || !availableProducts.some((p) => p.name === productA)) setProductA(p0)
        if (!productB || !availableProducts.some((p) => p.name === productB) || productB === p0) setProductB(p1)
      }, 0)
      return () => clearTimeout(timer)
    }
  }, [availableProducts, productA, productB])

  const handleRunVersionCompare = async () => {
    if (!versionA || !versionB) {
      setVersionError('Please select both Version A and Version B.')
      return
    }
    if (versionA.toLowerCase() === versionB.toLowerCase()) {
      setVersionError('Version A and Version B must be different versions.')
      return
    }

    setLoadingVersion(true)
    setVersionError(null)

    const items: ReviewInputItem[] = reviews.map((r) => ({
      id: r.review_id,
      text: r.original_text,
      rating: r.rating ?? r.pseudo_rating,
      date: r.date,
      version: r.version,
      product: r.product,
    }))

    try {
      const res = await compareVersions({
        reviews: items,
        version_a: versionA,
        version_b: versionB,
        product: selectedProduct || undefined,
      })
      setVersionResult(res)
    } catch (err) {
      setVersionError(err instanceof Error ? err.message : 'Version comparison failed.')
    } finally {
      setLoadingVersion(false)
    }
  }

  const handleRunProductCompare = async () => {
    if (!productA || !productB) {
      setProductError('Please select both Product A and Product B.')
      return
    }
    if (productA.toLowerCase() === productB.toLowerCase()) {
      setProductError('Product A and Product B must be different products.')
      return
    }

    setLoadingProduct(true)
    setProductError(null)

    const items: ReviewInputItem[] = reviews.map((r) => ({
      id: r.review_id,
      text: r.original_text,
      rating: r.rating ?? r.pseudo_rating,
      date: r.date,
      version: r.version,
      product: r.product,
    }))

    try {
      const res = await compareProducts({
        reviews: items,
        product_a: productA,
        product_b: productB,
        focus_aspect: focusAspect === 'All' ? undefined : focusAspect.toLowerCase(),
      })
      setProductResult(res)
    } catch (err) {
      setProductError(err instanceof Error ? err.message : 'Product comparison failed.')
    } finally {
      setLoadingProduct(false)
    }
  }

  const handleLoadSampleDataset = async () => {
    setLoadingSample(true)
    try {
      const res = await analyzeReviews(SAMPLE_REVIEWS)
      onAnalysisComplete(res)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingSample(false)
    }
  }

  return (
    <>
      <PageHeading
        title="Product & Version Comparison"
        description="Side-by-side comparative intelligence across software/firmware versions or distinct products."
        muted={muted}
      />

      {/* Tabs */}
      <div className={`flex border-b pb-1 gap-2 ${dark ? 'border-slate-700/60' : 'border-slate-200'}`}>
        <button
          onClick={() => setActiveTab('versions')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-bold rounded-lg transition ${
            activeTab === 'versions'
              ? 'bg-indigo-600 text-white shadow'
              : `${control} hover:border-slate-500`
          }`}
        >
          <GitCompare size={15} />
          Compare Product Versions
        </button>
        <button
          onClick={() => setActiveTab('products')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-bold rounded-lg transition ${
            activeTab === 'products'
              ? 'bg-indigo-600 text-white shadow'
              : `${control} hover:border-slate-500`
          }`}
        >
          <Scale size={15} />
          Compare Distinct Products
        </button>
      </div>

      {/* No reviews analyzed empty state */}
      {!analysis || reviews.length === 0 ? (
        <section className={`mt-6 rounded-2xl border p-8 text-center md:p-12 ${surface}`}>
          <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-500">
            <GitCompare size={28} />
          </div>
          <h3 className="mt-4 text-lg font-bold">No Active Review Analysis</h3>
          <p className={`mx-auto mt-2 max-w-md text-xs leading-relaxed ${muted}`}>
            Analyze your customer review file or load our multi-version, multi-product sample dataset to inspect comparative metrics.
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            <button
              onClick={handleLoadSampleDataset}
              disabled={loadingSample}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white shadow hover:bg-indigo-500 disabled:opacity-50"
            >
              {loadingSample ? <RefreshCw size={14} className="animate-spin" /> : <Sparkles size={14} />}
              Load & Analyze Multi-Version Sample (20 Reviews)
            </button>
            <button
              onClick={() => onOpenPage('Analyze Reviews')}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-4 py-2.5 text-xs font-semibold ${control}`}
            >
              Upload Review File →
            </button>
          </div>
        </section>
      ) : (
        <div className="mt-6 space-y-6">
          {/* TAB 1: COMPARE VERSIONS */}
          {activeTab === 'versions' && (
            <div className="space-y-6">
              {/* Controls Box */}
              <section className={`rounded-2xl border p-5 ${surface}`}>
                <div className={`flex flex-wrap items-center justify-between gap-2 border-b pb-3 mb-4 ${dark ? 'border-slate-700/50' : 'border-slate-200'}`}>
                  <div className="font-bold text-sm flex items-center gap-2">
                    <GitCompare size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Version Selection Controls
                  </div>
                  <span className={`text-xs ${muted}`}>
                    Dataset contains {availableVersions.length} distinct version(s) across {availableProducts.length || 1} product(s)
                  </span>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  {/* Product Filter */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Product Scope
                    </label>
                    <select
                      value={selectedProduct}
                      onChange={(e) => setSelectedProduct(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="">All Products ({availableProducts.length ? availableProducts.map(p => p.name).join(', ') : 'Single Product'})</option>
                      {availableProducts.map((p) => (
                        <option key={p.name} value={p.name}>
                          {p.name} ({p.count} reviews)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Version A */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Baseline (Version A)
                    </label>
                    <select
                      value={versionA}
                      onChange={(e) => setVersionA(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="">Select Version A...</option>
                      {versionsForSelectedProduct.map((v) => (
                        <option key={v.version} value={v.version}>
                          {v.version} ({v.count} reviews)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Version B */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Comparison Target (Version B)
                    </label>
                    <select
                      value={versionB}
                      onChange={(e) => setVersionB(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="">Select Version B...</option>
                      {versionsForSelectedProduct.map((v) => (
                        <option key={v.version} value={v.version}>
                          {v.version} ({v.count} reviews)
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {versionError && (
                  <div className={`mt-4 rounded-xl border p-3 text-xs ${dark ? 'border-rose-500/30 bg-rose-500/10 text-rose-400' : 'border-rose-200 bg-rose-50 text-rose-700'}`}>
                    {versionError}
                  </div>
                )}

                <div className={`mt-4 flex items-center justify-between pt-3 border-t ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                  <span className={`text-[11px] ${muted}`}>
                    * Compares aspect negative rates, cluster transitions, and safety escalation.
                  </span>
                  <button
                    onClick={handleRunVersionCompare}
                    disabled={loadingVersion || !versionA || !versionB || versionA === versionB}
                    className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-bold text-white shadow hover:bg-indigo-500 disabled:opacity-50"
                  >
                    {loadingVersion ? <RefreshCw size={14} className="animate-spin" /> : <GitCompare size={14} />}
                    Run Version Comparison
                  </button>
                </div>
              </section>

              {/* Version Comparison Results */}
              {versionResult && (
                <div className="space-y-6">
                  {/* Summary Banner */}
                  <div className={`rounded-2xl border p-5 ${dark ? 'border-indigo-500/30 bg-indigo-500/10' : 'border-indigo-200 bg-indigo-50/70'}`}>
                    <div className={`flex items-center gap-2 text-sm font-bold ${dark ? 'text-indigo-300' : 'text-indigo-800'}`}>
                      <Sparkles size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Evidence-Grounded Version Transition Summary
                    </div>
                    <p className={`mt-2 text-xs font-medium leading-relaxed ${dark ? 'text-slate-100' : 'text-slate-900'}`}>
                      {versionResult.summary_of_changes}
                    </p>
                    <p className={`mt-2 text-[11px] italic leading-relaxed ${dark ? 'text-indigo-300/80' : 'text-indigo-700'}`}>
                      {versionResult.disclaimer}
                    </p>
                  </div>

                  {/* Sparse Warning Banner */}
                  {versionResult.is_sparse_sample && (
                    <div className={`rounded-xl border p-3.5 text-xs flex items-start gap-2 ${
                      dark ? 'border-amber-500/30 bg-amber-500/10 text-amber-300' : 'border-amber-200 bg-amber-50/70 text-amber-900'
                    }`}>
                      <AlertTriangle size={16} className={`shrink-0 mt-0.5 ${dark ? 'text-amber-400' : 'text-amber-700'}`} />
                      <div>
                        <strong>Preliminary Sample Size:</strong> {versionResult.warnings.join(' ')}
                      </div>
                    </div>
                  )}

                  {/* Safety Changes Banner */}
                  {versionResult.safety_changes.length > 0 && (
                    <div className={`rounded-xl border p-4 text-xs ${
                      dark ? 'border-rose-500/30 bg-rose-500/10 text-rose-300' : 'border-rose-200 bg-rose-50/70 text-rose-800'
                    }`}>
                      <div className={`font-bold flex items-center gap-1.5 mb-1.5 ${dark ? 'text-rose-400' : 'text-rose-700'}`}>
                        <ShieldAlert size={16} /> Safety Posture Changes Between Versions
                      </div>
                      <ul className="space-y-1">
                        {versionResult.safety_changes.map((sc, idx) => (
                          <li key={idx}>• {sc}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Side-by-Side Metric Cards */}
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                    {/* Version A Card */}
                    <div className={`rounded-2xl border p-5 ${surface}`}>
                      <div className={`flex items-center justify-between border-b pb-3 mb-3 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                        <span className={`rounded-md px-2.5 py-1 text-xs font-bold ${
                          dark ? 'bg-slate-700/50 text-slate-100' : 'bg-slate-100 text-slate-800 border border-slate-200'
                        }`}>
                          Baseline: Version {versionResult.version_a.version}
                        </span>
                        <span className={`text-xs ${muted}`}>
                          {versionResult.version_a.review_count} Customer Reviews
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-center my-3">
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Average Star Rating</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
                            {typeof versionResult.version_a.average_rating === 'number'
                              ? `${versionResult.version_a.average_rating.toFixed(2)} ★`
                              : 'No Ratings'}
                          </div>
                        </div>
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Safety Alerts</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-rose-400' : 'text-rose-600'}`}>
                            {versionResult.version_a.safety_alert_count}
                          </div>
                        </div>
                      </div>

                      {/* Sentiment Proportions */}
                      <div className="mt-4">
                        <div className={`text-[11px] font-semibold mb-1.5 flex justify-between ${muted}`}>
                          <span>Sentiment Distribution</span>
                          <span>
                            {Math.round(versionResult.version_a.sentiment_proportions.pos_ratio * 100)}% Pos /{' '}
                            {Math.round(versionResult.version_a.sentiment_proportions.neg_ratio * 100)}% Neg
                          </span>
                        </div>
                        <div className={`h-3 w-full overflow-hidden rounded-full flex ${dark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                          <div
                            style={{ width: `${versionResult.version_a.sentiment_proportions.pos_ratio * 100}%` }}
                            className="bg-emerald-500"
                            title={`Positive: ${versionResult.version_a.sentiment_proportions.pos_count}`}
                          />
                          <div
                            style={{ width: `${versionResult.version_a.sentiment_proportions.neu_ratio * 100}%` }}
                            className="bg-slate-500"
                            title={`Neutral: ${versionResult.version_a.sentiment_proportions.neu_count}`}
                          />
                          <div
                            style={{ width: `${versionResult.version_a.sentiment_proportions.neg_ratio * 100}%` }}
                            className="bg-rose-500"
                            title={`Negative: ${versionResult.version_a.sentiment_proportions.neg_count}`}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Version B Card */}
                    <div className={`rounded-2xl border p-5 ${surface}`}>
                      <div className={`flex items-center justify-between border-b pb-3 mb-3 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                        <span className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-bold text-white">
                          Target: Version {versionResult.version_b.version}
                        </span>
                        <span className={`text-xs ${muted}`}>
                          {versionResult.version_b.review_count} Customer Reviews
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-center my-3">
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Average Star Rating</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
                            {typeof versionResult.version_b.average_rating === 'number'
                              ? `${versionResult.version_b.average_rating.toFixed(2)} ★`
                              : 'No Ratings'}
                            {typeof versionResult.version_a.average_rating === 'number' && typeof versionResult.version_b.average_rating === 'number' && (
                              <span className={`text-xs ml-1.5 font-bold ${
                                versionResult.version_b.average_rating >= versionResult.version_a.average_rating
                                  ? (dark ? 'text-emerald-400' : 'text-emerald-700')
                                  : (dark ? 'text-rose-400' : 'text-rose-700')
                              }`}>
                                ({versionResult.version_b.average_rating >= versionResult.version_a.average_rating ? '+' : ''}
                                {(versionResult.version_b.average_rating - versionResult.version_a.average_rating).toFixed(2)}★)
                              </span>
                            )}
                          </div>
                        </div>
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Safety Alerts</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-rose-400' : 'text-rose-600'}`}>
                            {versionResult.version_b.safety_alert_count}
                          </div>
                        </div>
                      </div>

                      {/* Sentiment Proportions */}
                      <div className="mt-4">
                        <div className={`text-[11px] font-semibold mb-1.5 flex justify-between ${muted}`}>
                          <span>Sentiment Distribution</span>
                          <span>
                            {Math.round(versionResult.version_b.sentiment_proportions.pos_ratio * 100)}% Pos /{' '}
                            {Math.round(versionResult.version_b.sentiment_proportions.neg_ratio * 100)}% Neg
                          </span>
                        </div>
                        <div className={`h-3 w-full overflow-hidden rounded-full flex ${dark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                          <div
                            style={{ width: `${versionResult.version_b.sentiment_proportions.pos_ratio * 100}%` }}
                            className="bg-emerald-500"
                            title={`Positive: ${versionResult.version_b.sentiment_proportions.pos_count}`}
                          />
                          <div
                            style={{ width: `${versionResult.version_b.sentiment_proportions.neu_ratio * 100}%` }}
                            className="bg-slate-500"
                            title={`Neutral: ${versionResult.version_b.sentiment_proportions.neu_count}`}
                          />
                          <div
                            style={{ width: `${versionResult.version_b.sentiment_proportions.neg_ratio * 100}%` }}
                            className="bg-rose-500"
                            title={`Negative: ${versionResult.version_b.sentiment_proportions.neg_count}`}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Aspect Comparison Table */}
                  <section className={`rounded-2xl border p-5 ${surface}`}>
                    <h4 className="font-bold text-sm mb-3">Aspect-Level Complaint Shifts</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className={`border-b ${dark ? 'border-slate-700/50' : 'border-slate-200'} ${muted}`}>
                            <th className="py-2.5 px-3">Product Aspect</th>
                            <th className="py-2.5 px-3">Version {versionResult.version_a.version} Complaints</th>
                            <th className="py-2.5 px-3">Version {versionResult.version_b.version} Complaints</th>
                            <th className="py-2.5 px-3">Complaint Delta</th>
                            <th className="py-2.5 px-3">Transition Status</th>
                          </tr>
                        </thead>
                        <tbody className={`divide-y ${dark ? 'divide-slate-800/50' : 'divide-slate-200'}`}>
                          {versionResult.aspect_comparisons.map((asp) => (
                            <tr key={asp.aspect} className={dark ? 'hover:bg-slate-800/30' : 'hover:bg-slate-50'}>
                              <td className={`py-3 px-3 font-semibold ${dark ? 'text-slate-100' : 'text-slate-900'}`}>{asp.aspect_label}</td>
                              <td className="py-3 px-3">
                                {asp.complaint_count_a} complaints ({Math.round(asp.negative_rate_a * 100)}% neg)
                              </td>
                              <td className="py-3 px-3">
                                {asp.complaint_count_b} complaints ({Math.round(asp.negative_rate_b * 100)}% neg)
                              </td>
                              <td className="py-3 px-3">
                                <span className={`font-bold ${
                                  asp.complaint_diff < 0
                                    ? (dark ? 'text-emerald-400' : 'text-emerald-700')
                                    : asp.complaint_diff > 0
                                      ? (dark ? 'text-rose-400' : 'text-rose-700')
                                      : (dark ? 'text-slate-400' : 'text-slate-600')
                                }`}>
                                  {asp.complaint_diff > 0 ? `+${asp.complaint_diff}` : asp.complaint_diff}
                                </span>
                              </td>
                              <td className="py-3 px-3">
                                <div>
                                  <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase ${
                                    asp.status === 'improved'
                                      ? (dark ? 'bg-emerald-500/15 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                                      : asp.status === 'worsened'
                                        ? (dark ? 'bg-rose-500/15 text-rose-400' : 'bg-rose-100 text-rose-800 border border-rose-200')
                                        : asp.status === 'newly_observed'
                                          ? (dark ? 'bg-purple-500/15 text-purple-400' : 'bg-purple-100 text-purple-800 border border-purple-200')
                                          : asp.status === 'not_observed'
                                            ? (dark ? 'bg-blue-500/15 text-blue-400' : 'bg-blue-100 text-blue-800 border border-blue-200')
                                            : asp.status === 'resolved'
                                              ? (dark ? 'bg-cyan-500/15 text-cyan-400' : 'bg-cyan-100 text-cyan-800 border border-cyan-200')
                                              : (dark ? 'bg-slate-500/15 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200')
                                  }`}>
                                    {asp.status.replace('_', ' ')}
                                  </span>
                                  {asp.sample_note && (
                                    <div className={`text-[10px] mt-1 ${muted}`}>
                                      {asp.sample_note}
                                    </div>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  {/* Issue Clusters Comparison */}
                  <section className={`rounded-2xl border p-5 ${surface}`}>
                    <h4 className="font-bold text-sm mb-3">Issue Cluster Evolution & Review Evidence</h4>
                    <div className="space-y-3">
                      {versionResult.cluster_comparisons.map((cl) => {
                        const isExpanded = expandedCluster === cl.cluster_label
                        return (
                          <div
                            key={cl.cluster_label}
                            className={`rounded-xl border p-4 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-white'}`}
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div>
                                <div className="flex items-center gap-2">
                                  <span className={`font-bold text-sm ${dark ? 'text-slate-100' : 'text-slate-900'}`}>{cl.cluster_label}</span>
                                  <span className={`text-[11px] rounded px-2 py-0.5 uppercase font-semibold ${
                                    dark ? 'bg-indigo-500/15 text-indigo-400' : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                                  }`}>
                                    {cl.aspect}
                                  </span>
                                </div>
                                <div className={`mt-1 text-xs ${muted}`}>
                                  Version {versionResult.version_a.version}: {cl.count_a} reviews ({Math.round(cl.share_a * 100)}%) → Version {versionResult.version_b.version}: {cl.count_b} reviews ({Math.round(cl.share_b * 100)}%)
                                </div>
                              </div>

                              <div className="flex items-center gap-2">
                                <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase ${
                                  cl.status === 'improved'
                                    ? (dark ? 'bg-emerald-500/15 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                                    : cl.status === 'worsened'
                                      ? (dark ? 'bg-rose-500/15 text-rose-400' : 'bg-rose-100 text-rose-800 border border-rose-200')
                                      : cl.status === 'newly_observed'
                                        ? (dark ? 'bg-purple-500/15 text-purple-400' : 'bg-purple-100 text-purple-800 border border-purple-200')
                                        : cl.status === 'not_observed'
                                          ? (dark ? 'bg-blue-500/15 text-blue-400' : 'bg-blue-100 text-blue-800 border border-blue-200')
                                          : cl.status === 'resolved'
                                            ? (dark ? 'bg-cyan-500/15 text-cyan-400' : 'bg-cyan-100 text-cyan-800 border border-cyan-200')
                                            : (dark ? 'bg-slate-500/15 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200')
                                }`}>
                                  {cl.status.replace('_', ' ')}
                                </span>
                                <button
                                  onClick={() => setExpandedCluster(isExpanded ? null : cl.cluster_label)}
                                  className={`rounded-md border p-1.5 text-xs ${control}`}
                                  aria-label="Toggle evidence excerpts"
                                >
                                  {isExpanded ? <X size={13} /> : <Eye size={13} />}
                                </button>
                                {onShowEvidence && (
                                  <button
                                    onClick={() => {
                                      const match = analysis?.clusters.find(
                                        (c) => c.label.toLowerCase() === cl.cluster_label.toLowerCase(),
                                      )
                                      if (match) onShowEvidence(match)
                                    }}
                                    className={`rounded-md border p-1.5 text-xs ${control} ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}
                                    title="Open Full Traceable Evidence & Root Cause Hypotheses"
                                  >
                                    <Sparkles size={13} />
                                  </button>
                                )}
                              </div>
                            </div>

                            {cl.sample_limitation && (
                              <div className={`mt-2 text-[11px] italic ${dark ? 'text-amber-300/90' : 'text-amber-800'}`}>
                                ℹ {cl.sample_limitation}
                              </div>
                            )}

                            {/* Expandable Traceable Evidence */}
                            {isExpanded && (
                              <div className={`mt-4 pt-3 border-t space-y-3 ${dark ? 'border-slate-700/50' : 'border-slate-200'}`}>
                                {cl.evidence_a.length > 0 && (
                                  <div>
                                    <div className={`text-[11px] font-bold uppercase ${dark ? 'text-slate-400' : 'text-slate-600'}`}>
                                      Version {versionResult.version_a.version} Customer Quotes:
                                    </div>
                                    <div className="mt-1.5 space-y-1.5">
                                      {cl.evidence_a.map((ev, i) => (
                                        <div key={i} className={`rounded-lg border p-2.5 text-xs ${dark ? 'border-slate-800 bg-slate-950/70' : 'border-slate-200 bg-slate-50'}`}>
                                          <div className={`flex justify-between text-[10px] font-mono mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600 font-semibold'}`}>
                                            <span>Review ID: {ev.reference_id}</span>
                                            <span>Sentiment: {typeof ev.sentiment_score === 'number' ? ev.sentiment_score.toFixed(2) : 'N/A'}</span>
                                          </div>
                                          <p className="italic">"{ev.excerpt_or_observation}"</p>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {cl.evidence_b.length > 0 && (
                                  <div>
                                    <div className={`text-[11px] font-bold uppercase ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                                      Version {versionResult.version_b.version} Customer Quotes:
                                    </div>
                                    <div className="mt-1.5 space-y-1.5">
                                      {cl.evidence_b.map((ev, i) => (
                                        <div key={i} className={`rounded-lg border p-2.5 text-xs ${dark ? 'border-slate-800 bg-slate-950/70' : 'border-slate-200 bg-slate-50'}`}>
                                          <div className={`flex justify-between text-[10px] font-mono mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600 font-semibold'}`}>
                                            <span>Review ID: {ev.reference_id}</span>
                                            <span>Sentiment: {typeof ev.sentiment_score === 'number' ? ev.sentiment_score.toFixed(2) : 'N/A'}</span>
                                          </div>
                                          <p className="italic">"{ev.excerpt_or_observation}"</p>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </section>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: COMPARE DISTINCT PRODUCTS */}
          {activeTab === 'products' && (
            <div className="space-y-6">
              {/* Product Controls Box */}
              <section className={`rounded-2xl border p-5 ${surface}`}>
                <div className={`flex flex-wrap items-center justify-between gap-2 border-b pb-3 mb-4 ${dark ? 'border-slate-700/50' : 'border-slate-200'}`}>
                  <div className="font-bold text-sm flex items-center gap-2">
                    <Scale size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Competitive Product Selection
                  </div>
                  <span className={`text-xs ${muted}`}>
                    Dataset contains {availableProducts.length} identified product(s)
                  </span>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  {/* Product A */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Product A
                    </label>
                    <select
                      value={productA}
                      onChange={(e) => setProductA(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="">Select Product A...</option>
                      {availableProducts.map((p) => (
                        <option key={p.name} value={p.name}>
                          {p.name} ({p.count} reviews)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Product B */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Product B
                    </label>
                    <select
                      value={productB}
                      onChange={(e) => setProductB(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="">Select Product B...</option>
                      {availableProducts.map((p) => (
                        <option key={p.name} value={p.name}>
                          {p.name} ({p.count} reviews)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Focus Aspect */}
                  <div>
                    <label className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                      Tailored Priority Focus (Optional)
                    </label>
                    <select
                      value={focusAspect}
                      onChange={(e) => setFocusAspect(e.target.value)}
                      className={`w-full rounded-lg border px-3 py-2 text-xs font-medium ${control}`}
                    >
                      <option value="All">All Aspects (Balanced)</option>
                      <option value="battery">Battery & Charging</option>
                      <option value="quality">Build Quality & Durability</option>
                      <option value="performance">Performance & Speed</option>
                      <option value="sound">Audio & Acoustics</option>
                      <option value="camera">Camera & Optics</option>
                      <option value="pricing">Price & Value</option>
                      <option value="delivery">Shipping & Delivery</option>
                      <option value="packaging">Packaging Integrity</option>
                    </select>
                  </div>
                </div>

                {productError && (
                  <div className={`mt-4 rounded-xl border p-3 text-xs ${dark ? 'border-rose-500/30 bg-rose-500/10 text-rose-400' : 'border-rose-200 bg-rose-50 text-rose-700'}`}>
                    {productError}
                  </div>
                )}

                <div className={`mt-4 flex items-center justify-between pt-3 border-t ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                  <span className={`text-[11px] ${muted}`}>
                    * Compares aspect strengths, common complaints, and user trade-offs without declaring false winners.
                  </span>
                  <button
                    onClick={handleRunProductCompare}
                    disabled={loadingProduct || !productA || !productB || productA === productB}
                    className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-bold text-white shadow hover:bg-indigo-500 disabled:opacity-50"
                  >
                    {loadingProduct ? <RefreshCw size={14} className="animate-spin" /> : <Scale size={14} />}
                    Run Product Comparison
                  </button>
                </div>
              </section>

              {/* Product Comparison Results */}
              {productResult && (
                <div className="space-y-6">
                  {/* Tradeoff Summary Banner */}
                  <div className={`rounded-2xl border p-5 space-y-2 ${dark ? 'border-indigo-500/30 bg-indigo-500/10' : 'border-indigo-200 bg-indigo-50/70'}`}>
                    <div className={`flex items-center gap-2 text-sm font-bold ${dark ? 'text-indigo-300' : 'text-indigo-800'}`}>
                      <Scale size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} /> Balanced Product Trade-Off Analysis
                    </div>
                    <p className={`text-xs font-medium leading-relaxed ${dark ? 'text-slate-100' : 'text-slate-900'}`}>
                      {productResult.tradeoff_summary}
                    </p>
                    <div className={`rounded-xl border p-3 text-xs ${dark ? 'border-indigo-500/20 bg-indigo-500/10 text-indigo-200' : 'border-indigo-200 bg-indigo-50/70 text-indigo-900'}`}>
                      <strong>Targeted Recommendation:</strong> {productResult.recommended_fit}
                    </div>
                    <p className={`text-[11px] italic ${dark ? 'text-indigo-300/80' : 'text-indigo-700'}`}>
                      {productResult.disclaimer}
                    </p>
                  </div>

                  {/* Warnings Banner */}
                  {productResult.warnings.length > 0 && (
                    <div className={`rounded-xl border p-3.5 text-xs space-y-1 ${dark ? 'border-amber-500/30 bg-amber-500/10 text-amber-300' : 'border-amber-200 bg-amber-50/70 text-amber-900'}`}>
                      {productResult.warnings.map((w, i) => (
                        <div key={i} className="flex items-start gap-1.5">
                          <AlertTriangle size={15} className={`shrink-0 mt-0.5 ${dark ? 'text-amber-400' : 'text-amber-700'}`} />
                          <span>{w}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Prominent Safety Alerts */}
                  {productResult.prominent_safety_alerts.length > 0 && (
                    <div className={`rounded-2xl border p-5 ${dark ? 'border-rose-500/30 bg-rose-500/10' : 'border-rose-200 bg-rose-50/70'}`}>
                      <div className={`font-bold text-sm flex items-center gap-2 mb-3 ${dark ? 'text-rose-400' : 'text-rose-700'}`}>
                        <ShieldAlert size={18} /> Prominent Safety Alerts for Compared Products
                      </div>
                      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                        {productResult.prominent_safety_alerts.map((al, idx) => (
                          <div key={idx} className={`rounded-xl border p-3 text-xs ${dark ? 'border-rose-500/30 bg-slate-950/70' : 'border-rose-200 bg-white'}`}>
                            <div className="flex justify-between font-bold text-[11px] mb-1">
                              <span className={dark ? 'text-rose-400' : 'text-rose-600'}>{al.product} · {al.tier.toUpperCase()} TIER</span>
                              <span className={`font-mono ${dark ? 'text-slate-400' : 'text-slate-600'}`}>Review {al.review_id}</span>
                            </div>
                            <p className={`italic ${dark ? 'text-slate-200' : 'text-slate-800'}`}>"{al.excerpt}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Side-by-Side Product Cards */}
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                    {/* Product A Card */}
                    <div className={`rounded-2xl border p-5 ${surface}`}>
                      <div className={`flex items-center justify-between border-b pb-3 mb-3 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                        <span className="rounded-md bg-indigo-600 px-2.5 py-1 text-xs font-bold text-white">
                          {productResult.product_a.product}
                        </span>
                        <span className={`text-xs ${muted}`}>
                          {productResult.product_a.review_count} Reviews
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-center my-3">
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Average Rating</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
                            {typeof productResult.product_a.average_rating === 'number'
                              ? `${productResult.product_a.average_rating.toFixed(2)} ★`
                              : 'No Ratings'}
                          </div>
                        </div>
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Safety Incidents</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-rose-400' : 'text-rose-600'}`}>
                            {productResult.product_a.safety_alert_count}
                          </div>
                        </div>
                      </div>

                      {/* Sentiment Proportions */}
                      <div className="mt-4">
                        <div className={`text-[11px] font-semibold mb-1.5 flex justify-between ${muted}`}>
                          <span>Customer Sentiment</span>
                          <span>
                            {Math.round(productResult.product_a.sentiment_proportions.pos_ratio * 100)}% Pos /{' '}
                            {Math.round(productResult.product_a.sentiment_proportions.neg_ratio * 100)}% Neg
                          </span>
                        </div>
                        <div className={`h-3 w-full overflow-hidden rounded-full flex ${dark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                          <div
                            style={{ width: `${productResult.product_a.sentiment_proportions.pos_ratio * 100}%` }}
                            className="bg-emerald-500"
                            title={`Positive: ${productResult.product_a.sentiment_proportions.pos_count}`}
                          />
                          <div
                            style={{ width: `${productResult.product_a.sentiment_proportions.neu_ratio * 100}%` }}
                            className="bg-slate-500"
                            title={`Neutral: ${productResult.product_a.sentiment_proportions.neu_count}`}
                          />
                          <div
                            style={{ width: `${productResult.product_a.sentiment_proportions.neg_ratio * 100}%` }}
                            className="bg-rose-500"
                            title={`Negative: ${productResult.product_a.sentiment_proportions.neg_count}`}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Product B Card */}
                    <div className={`rounded-2xl border p-5 ${surface}`}>
                      <div className={`flex items-center justify-between border-b pb-3 mb-3 ${dark ? 'border-slate-700/40' : 'border-slate-200'}`}>
                        <span className="rounded-md bg-blue-600 px-2.5 py-1 text-xs font-bold text-white">
                          {productResult.product_b.product}
                        </span>
                        <span className={`text-xs ${muted}`}>
                          {productResult.product_b.review_count} Reviews
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-center my-3">
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Average Rating</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-amber-400' : 'text-amber-600'}`}>
                            {typeof productResult.product_b.average_rating === 'number'
                              ? `${productResult.product_b.average_rating.toFixed(2)} ★`
                              : 'No Ratings'}
                          </div>
                        </div>
                        <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-slate-50'}`}>
                          <div className={`text-[10px] uppercase font-bold ${muted}`}>Safety Incidents</div>
                          <div className={`text-xl font-extrabold mt-1 ${dark ? 'text-rose-400' : 'text-rose-600'}`}>
                            {productResult.product_b.safety_alert_count}
                          </div>
                        </div>
                      </div>

                      {/* Sentiment Proportions */}
                      <div className="mt-4">
                        <div className={`text-[11px] font-semibold mb-1.5 flex justify-between ${muted}`}>
                          <span>Customer Sentiment</span>
                          <span>
                            {Math.round(productResult.product_b.sentiment_proportions.pos_ratio * 100)}% Pos /{' '}
                            {Math.round(productResult.product_b.sentiment_proportions.neg_ratio * 100)}% Neg
                          </span>
                        </div>
                        <div className={`h-3 w-full overflow-hidden rounded-full flex ${dark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                          <div
                            style={{ width: `${productResult.product_b.sentiment_proportions.pos_ratio * 100}%` }}
                            className="bg-emerald-500"
                            title={`Positive: ${productResult.product_b.sentiment_proportions.pos_count}`}
                          />
                          <div
                            style={{ width: `${productResult.product_b.sentiment_proportions.neu_ratio * 100}%` }}
                            className="bg-slate-500"
                            title={`Neutral: ${productResult.product_b.sentiment_proportions.neu_count}`}
                          />
                          <div
                            style={{ width: `${productResult.product_b.sentiment_proportions.neg_ratio * 100}%` }}
                            className="bg-rose-500"
                            title={`Negative: ${productResult.product_b.sentiment_proportions.neg_count}`}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Aspect Strengths & Weaknesses */}
                  <section className={`rounded-2xl border p-5 ${surface}`}>
                    <h4 className="font-bold text-sm mb-3">Aspect-by-Aspect Strengths & Weaknesses</h4>
                    <div className="space-y-3">
                      {productResult.aspect_strengths.map((item) => (
                        <div
                          key={item.aspect}
                          className={`rounded-xl border p-4 ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-white'}`}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className={`font-bold text-sm ${dark ? 'text-slate-100' : 'text-slate-900'}`}>{item.aspect_label}</span>
                              <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-extrabold uppercase ${
                                item.leader === 'product_a'
                                  ? (dark ? 'bg-indigo-500/20 text-indigo-400' : 'bg-indigo-100 text-indigo-800 border border-indigo-200')
                                  : item.leader === 'product_b'
                                    ? (dark ? 'bg-blue-500/20 text-blue-400' : 'bg-blue-100 text-blue-800 border border-blue-200')
                                    : (dark ? 'bg-slate-500/20 text-slate-400' : 'bg-slate-100 text-slate-700 border border-slate-200')
                              }`}>
                                {item.leader === 'product_a'
                                  ? `${productResult.product_a.product} Advantage`
                                  : item.leader === 'product_b'
                                    ? `${productResult.product_b.product} Advantage`
                                    : 'Evenly Matched'}
                              </span>
                            </div>
                            <div className={`text-xs ${muted}`}>
                              {productResult.product_a.product}: {item.complaints_a} complaints vs {productResult.product_b.product}: {item.complaints_b} complaints
                            </div>
                          </div>
                          <p className={`mt-2 text-xs leading-relaxed ${dark ? 'text-slate-200' : 'text-slate-800'}`}>
                            {item.strength_summary}
                          </p>
                        </div>
                      ))}
                    </div>
                  </section>

                  {/* Complaint Themes: Common vs Unique */}
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                    {/* Common Themes */}
                    <div className={`rounded-2xl border p-4 ${surface}`}>
                      <div className={`font-bold text-xs uppercase mb-2.5 flex items-center gap-1.5 ${dark ? 'text-slate-400' : 'text-slate-700'}`}>
                        <CheckCircle2 size={14} className={dark ? 'text-emerald-400' : 'text-emerald-700'} /> Shared Complaint Themes
                      </div>
                      {productResult.common_complaint_themes.length > 0 ? (
                        <ul className={`space-y-1.5 text-xs ${dark ? 'text-slate-200' : 'text-slate-800'}`}>
                          {productResult.common_complaint_themes.map((t, idx) => (
                            <li key={idx}>• {t}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className={`text-xs ${muted}`}>No common complaint themes identified across both products.</p>
                      )}
                    </div>

                    {/* Unique to Product A */}
                    <div className={`rounded-2xl border p-4 ${surface}`}>
                      <div className={`font-bold text-xs uppercase mb-2.5 flex items-center gap-1.5 ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                        <Info size={14} /> Unique to {productResult.product_a.product}
                      </div>
                      {productResult.unique_issues_a.length > 0 ? (
                        <ul className={`space-y-1.5 text-xs ${dark ? 'text-slate-200' : 'text-slate-800'}`}>
                          {productResult.unique_issues_a.map((u, idx) => (
                            <li key={idx}>• {u}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className={`text-xs ${muted}`}>No product-specific complaint themes observed.</p>
                      )}
                    </div>

                    {/* Unique to Product B */}
                    <div className={`rounded-2xl border p-4 ${surface}`}>
                      <div className={`font-bold text-xs uppercase mb-2.5 flex items-center gap-1.5 ${dark ? 'text-blue-400' : 'text-blue-700'}`}>
                        <Info size={14} /> Unique to {productResult.product_b.product}
                      </div>
                      {productResult.unique_issues_b.length > 0 ? (
                        <ul className={`space-y-1.5 text-xs ${dark ? 'text-slate-200' : 'text-slate-800'}`}>
                          {productResult.unique_issues_b.map((u, idx) => (
                            <li key={idx}>• {u}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className={`text-xs ${muted}`}>No product-specific complaint themes observed.</p>
                      )}
                    </div>
                  </div>

                  {/* Key Evidence Snippets */}
                  <section className={`rounded-2xl border p-5 ${surface}`}>
                    <h4 className="font-bold text-sm mb-3">Traceable Review Evidence Quotes</h4>
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                      <div>
                        <div className={`text-xs font-bold uppercase mb-2 ${dark ? 'text-indigo-400' : 'text-indigo-700'}`}>
                          {productResult.product_a.product} Evidence Excerpts
                        </div>
                        <div className="space-y-2">
                          {productResult.key_evidence_a.map((ev, i) => (
                            <div key={i} className={`rounded-xl border p-3 text-xs ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-white'}`}>
                              <div className={`flex justify-between text-[10px] font-mono mb-1 ${dark ? 'text-indigo-400' : 'text-indigo-600 font-semibold'}`}>
                                <span>Review ID: {ev.reference_id}</span>
                                <span>Sentiment: {typeof ev.sentiment_score === 'number' ? ev.sentiment_score.toFixed(2) : 'N/A'}</span>
                              </div>
                              <p className="italic">"{ev.excerpt_or_observation}"</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div>
                        <div className={`text-xs font-bold uppercase mb-2 ${dark ? 'text-blue-400' : 'text-blue-700'}`}>
                          {productResult.product_b.product} Evidence Excerpts
                        </div>
                        <div className="space-y-2">
                          {productResult.key_evidence_b.map((ev, i) => (
                            <div key={i} className={`rounded-xl border p-3 text-xs ${dark ? 'border-slate-800 bg-slate-900/60' : 'border-slate-200 bg-white'}`}>
                              <div className={`flex justify-between text-[10px] font-mono mb-1 ${dark ? 'text-blue-400' : 'text-blue-600 font-semibold'}`}>
                                <span>Review ID: {ev.reference_id}</span>
                                <span>Sentiment: {typeof ev.sentiment_score === 'number' ? ev.sentiment_score.toFixed(2) : 'N/A'}</span>
                              </div>
                              <p className="italic">"{ev.excerpt_or_observation}"</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </section>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </>
  )
}

function HistoryPage({
  dark,
  surface,
  muted,
  control,
  history,
  loading = false,
  onLoadRun,
  onOpenPage,
}: PageProps & {
  history: HistoryItem[]
  loading?: boolean
  onLoadRun: (item: HistoryItem) => void
}) {
  return (
    <>
      <PageHeading
        title="Analysis History"
        description="Review previously completed analyses saved securely to your private workspace."
        muted={muted}
      />

      {loading ? (
        <section className={`rounded-2xl border p-12 text-center ${surface}`}>
          <div className="size-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className={`text-xs ${muted}`}>Loading your analysis history...</p>
        </section>
      ) : history.length > 0 ? (
        <section className="space-y-3">
          {history.map((item) => (
            <div
              key={item.id}
              className={`flex flex-wrap items-center justify-between gap-4 rounded-xl border p-5 ${surface}`}
            >
              <div>
                <div className="flex items-center gap-2 font-semibold">
                  <FileSpreadsheet size={16} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
                  {item.sourceName}
                </div>
                <div className={`mt-1 text-xs ${muted}`}>
                  {item.timestamp} · {item.totalReviews} reviews · {item.safetyAlerts} safety alerts
                </div>
              </div>

              <button
                onClick={() => onLoadRun(item)}
                className={`rounded-lg border px-4 py-2 text-xs font-semibold ${control}`}
              >
                Open Analysis →
              </button>
            </div>
          ))}
        </section>
      ) : (
        <section className={`rounded-2xl border p-8 text-center md:p-12 ${surface}`}>
          <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-500">
            <History size={27} />
          </div>
          <h3 className="mt-4 text-lg font-semibold">No Saved Analyses Yet</h3>
          <p className={`mx-auto mt-2 max-w-lg text-sm leading-6 ${muted}`}>
            Run an analysis from the Analyze Reviews tab to record sessions here.
          </p>
          <button
            onClick={() => onOpenPage('Analyze Reviews')}
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500"
          >
            Analyze reviews <ArrowRight size={16} />
          </button>
        </section>
      )}
    </>
  )
}

function SettingsPage({
  dark,
  surface,
  muted,
  control,
  theme,
  setTheme,
  backendHealth,
  onRefreshHealth,
}: SharedProps & {
  theme: Theme
  setTheme: (theme: Theme) => void
  backendHealth: HealthResponse | null
  onRefreshHealth: () => void
}) {
  return (
    <>
      <PageHeading
        title="Settings & Workspace Preferences"
        description="Verify backend connectivity, processing modes, and UI appearance."
        muted={muted}
      />

      <section className={`divide-y rounded-xl border ${dark ? 'divide-slate-800' : 'divide-slate-200'} ${surface}`}>
        {/* Backend Health Diagnostic */}
        <div className="p-5 flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 font-semibold">
              <Activity size={18} className="text-indigo-500" /> Backend Engine Status
            </div>
            <p className={`mt-1 text-xs ${muted}`}>
              FastAPI NLP service connectivity on {API_BASE_URL || 'local proxy'}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                backendHealth?.status === 'ok'
                  ? (dark ? 'bg-emerald-500/10 text-emerald-400' : 'bg-emerald-100 text-emerald-800 border border-emerald-200')
                  : (dark ? 'bg-rose-500/10 text-rose-400' : 'bg-rose-100 text-rose-800 border border-rose-200')
              }`}
            >
              <span
                className={`size-2 rounded-full ${
                  backendHealth?.status === 'ok' ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
              {backendHealth?.status === 'ok'
                ? `Online (${backendHealth.mode.toUpperCase()} mode)`
                : 'Offline / Connecting'}
            </span>

            <button
              onClick={onRefreshHealth}
              className={`rounded-lg border p-2 text-xs ${control}`}
              title="Refresh connection status"
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </div>

        {/* Appearance Switcher */}
        <div className="flex flex-wrap items-center justify-between gap-4 p-5">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-indigo-500/10 p-2 text-indigo-500">
              {dark ? <Moon size={20} /> : <Sun size={20} />}
            </div>
            <div>
              <h3 className="font-semibold text-sm">Theme Appearance</h3>
              <p className={`mt-0.5 text-xs ${muted}`}>Choose a light or dark visual theme.</p>
            </div>
          </div>
          <div className={`flex rounded-lg border p-1 ${control}`}>
            <button
              onClick={() => setTheme('light')}
              className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-semibold ${
                theme === 'light' ? 'bg-indigo-600 text-white' : ''
              }`}
            >
              <Sun size={15} /> Light
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-semibold ${
                theme === 'dark' ? 'bg-indigo-600 text-white' : ''
              }`}
            >
              <Moon size={15} /> Dark
            </button>
          </div>
        </div>

        {/* API Base URL Spec */}
        <div className="p-5">
          <h3 className="font-semibold text-sm">Configured API Base URL</h3>
          <p className={`mt-1 text-xs ${muted}`}>
            Environment variable <code>VITE_API_BASE_URL</code>: {API_BASE_URL || '(empty — using local Vite proxy)'}
          </p>
        </div>
      </section>
    </>
  )
}

export default function App() {
  const { user, loading: authLoading, signOut, isSupabaseConfigured } = useAuth()

  const [showSplash, setShowSplash] = useState<boolean>(() => {
    try {
      return !sessionStorage.getItem('insightforge_splash_shown')
    } catch {
      return false
    }
  })

  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem('insightforge-theme') === 'dark' ? 'dark' : 'light'
    } catch {
      return 'dark'
    }
  })

  const [activeNav, setActiveNav] = useState<Page>('Overview')
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(null)
  const [selectedCluster, setSelectedCluster] = useState<IssueCluster | null>(null)
  const [visionContext, setVisionContext] = useState<{ cluster?: string; aspect?: string } | null>(null)
  const [visionGallery, setVisionGallery] = useState<ImageDefectAnalysis[]>([])
  const [backendHealth, setBackendHealth] = useState<HealthResponse | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState<boolean>(false)

  // Fetch user-isolated analysis history when authenticated user changes
  useEffect(() => {
    let active = true
    if (user?.id) {
      fetchUserHistory(user.id)
        .then((items) => {
          if (!active) return
          setHistory(items)
          setHistoryLoading(false)
        })
        .catch((err) => {
          if (!active) return
          console.warn('Failed to load user history:', err)
          setHistory([])
          setHistoryLoading(false)
        })
    }

    return () => {
      active = false
    }
  }, [user?.id])

  // Theme effect
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.style.colorScheme = theme
    try {
      localStorage.setItem('insightforge-theme', theme)
    } catch {
      // ignore
    }
  }, [theme])

  // Check health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  function checkHealth() {
    fetchHealth()
      .then(setBackendHealth)
      .catch((err) => {
        console.warn('Backend health check failed:', err)
        setBackendHealth(null)
      })
  }

  async function handleAnalysisComplete(res: AnalyzeResponse) {
    setAnalysisResult(res)

    const userId = user?.id || 'anonymous'
    const saved = await saveUserHistory(userId, {
      sourceName: `Batch (${res.reviews.length} reviews)`,
      totalReviews: res.reviews.length,
      avgRating: Number(
        (
          res.reviews.reduce((acc, r) => acc + (r.rating ?? r.pseudo_rating ?? 3), 0) /
          Math.max(1, res.reviews.length)
        ).toFixed(2)
      ),
      safetyAlerts: res.reviews.filter((r) => r.safety.is_alert).length,
      result: res,
    })

    setHistory((prev) => [saved, ...prev.filter((item) => item.id !== saved.id).slice(0, 19)])
    setActiveNav('Insights')
  }

  async function handleLogout() {
    // Clear user-specific in-memory state to prevent cross-account leakage
    setHistory([])
    setAnalysisResult(null)
    setSelectedCluster(null)
    setVisionContext(null)
    setVisionGallery([])
    await signOut()
    setActiveNav('Overview')
  }

  function openPage(page: Page) {
    setActiveNav(page)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const dark = theme === 'dark'

  function handleSplashDone() {
    try {
      sessionStorage.setItem('insightforge_splash_shown', 'true')
    } catch {
      // ignore storage access errors
    }
    setShowSplash(false)
  }

  // Show cursor-reactive intro splash once per browser session
  if (showSplash) {
    return <IntroSplash onDone={handleSplashDone} />
  }

  // If session is still verifying, show clean loader without flashing unauthenticated screens
  if (authLoading) {
    return (
      <div className={`min-h-screen flex flex-col items-center justify-center p-4 ${dark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
        <div className="size-14 rounded-2xl bg-indigo-600 flex items-center justify-center text-white font-black text-2xl shadow-xl shadow-indigo-500/25 animate-pulse mb-4">
          I
        </div>
        <div className="flex items-center gap-2.5 text-xs text-indigo-400 font-semibold">
          <span className="size-3.5 rounded-full border-2 border-indigo-400/30 border-t-indigo-400 animate-spin" />
          Verifying authenticated session...
        </div>
      </div>
    )
  }

  // If Supabase is configured and visitor is unauthenticated, redirect to AuthView
  if (isSupabaseConfigured && !user) {
    return <AuthView dark={dark} />
  }

  const surface = dark
    ? 'bg-slate-900 border-slate-800 text-slate-100'
    : 'bg-white border-slate-200 text-slate-900'
  const muted = dark ? 'text-slate-400' : 'text-slate-600'
  const canvas = dark ? 'bg-slate-950' : 'bg-slate-50'
  const control = dark
    ? 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700'
    : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50'

  const sharedPageProps: PageProps = {
    dark,
    surface,
    muted,
    control,
    onOpenPage: openPage,
  }

  return (
    <div className={`min-h-screen ${canvas} ${dark ? 'dark' : ''}`}>
      <div className="flex min-h-screen">
        {/* Sidebar Navigation */}
        <aside className={`hidden w-64 shrink-0 flex-col border-r p-4 md:flex ${surface}`}>
          <button
            onClick={() => openPage('Overview')}
            className="mb-8 flex items-center gap-3 px-2 text-left"
          >
            <div className="flex size-10 items-center justify-center rounded-xl bg-indigo-600 text-xl font-bold text-white shadow-md shadow-indigo-500/25">
              I
            </div>
            <div>
              <div className="font-bold tracking-tight text-base">InsightForge</div>
              <div className={`text-xs ${muted}`}>Review Intelligence</div>
            </div>
          </button>

          <div className={`mb-3 px-3 text-[11px] font-bold uppercase tracking-wider ${muted}`}>
            Workspace
          </div>

          <nav className="space-y-1">
            {navItems.map(({ name, icon: Icon }) => (
              <button
                key={name}
                onClick={() => openPage(name)}
                aria-current={activeNav === name ? 'page' : undefined}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition ${
                  activeNav === name
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : `${muted} hover:bg-indigo-500/10 ${dark ? 'hover:text-indigo-400' : 'hover:text-indigo-600'}`
                }`}
              >
                <Icon size={17} /> {name}
              </button>
            ))}
          </nav>

          {/* Sidebar Status Pill */}
          <div
            className={`mt-auto rounded-xl border p-3.5 text-xs ${
              dark ? 'border-slate-800 bg-slate-950' : 'border-slate-200 bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-between font-semibold">
              <span className="flex items-center gap-2">
                <Activity size={14} className="text-indigo-500" /> Engine
              </span>
              <span
                className={`size-2 rounded-full ${
                  backendHealth?.status === 'ok' ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
            </div>
            <div className={`mt-1.5 leading-5 ${muted}`}>
              Mode: {backendHealth?.mode?.toUpperCase() || 'LITE'}
              <br />
              Status: {backendHealth?.status === 'ok' ? 'Online' : 'Disconnected'}
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="min-w-0 flex-1">
          {/* Header */}
          <header className={`sticky top-0 z-10 flex items-center justify-between gap-3 border-b px-4 py-3 md:px-8 ${surface}`}>
            <div className="min-w-0">
              <div className={`text-xs ${muted}`}>Workspace / {activeNav}</div>
              <h1 className="mt-0.5 text-lg font-bold">{activeNav}</h1>
            </div>

            <div className="flex items-center gap-2.5">
              {backendHealth?.status === 'ok' && (
                <span className={`hidden rounded-lg border px-3 py-1.5 text-xs sm:inline-flex items-center gap-2 ${control}`}>
                  <span className="size-2 rounded-full bg-emerald-500" />
                  Backend Online
                </span>
              )}

              {user && (
                <div className={`flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs ${control}`}>
                  <span className="flex size-5 items-center justify-center rounded-full bg-indigo-600 text-[10px] font-bold text-white shrink-0">
                    {user.email ? user.email[0].toUpperCase() : 'U'}
                  </span>
                  <span className="hidden font-medium sm:inline max-w-[140px] truncate" title={user.email || ''}>
                    {user.email || 'User'}
                  </span>
                  <button
                    onClick={handleLogout}
                    className={`ml-1 transition ${dark ? 'text-slate-400 hover:text-rose-400' : 'text-slate-600 hover:text-rose-600'}`}
                    title="Sign Out"
                    aria-label="Sign out"
                  >
                    <LogOut size={14} />
                  </button>
                </div>
              )}

              <button
                onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
                aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
                className={`rounded-lg border p-2 ${control}`}
              >
                {theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}
              </button>
            </div>
          </header>

          <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
            {activeNav === 'Overview' && (
              <OverviewPage
                {...sharedPageProps}
                analysis={analysisResult}
                onShowEvidence={(cl) => setSelectedCluster(cl)}
              />
            )}

            {activeNav === 'Analyze Reviews' && (
              <AnalyzePage
                {...sharedPageProps}
                analysis={analysisResult}
                onAnalyzeComplete={handleAnalysisComplete}
              />
            )}

            {activeNav === 'Insights' && (
              <InsightsPage
                {...sharedPageProps}
                analysis={analysisResult}
                onShowEvidence={(cl) => setSelectedCluster(cl)}
              />
            )}

            {activeNav === 'Emerging Issues' && (
              <EmergingIssuesPage
                {...sharedPageProps}
                analysis={analysisResult}
                onShowEvidence={(cl) => setSelectedCluster(cl)}
                onInspectVision={(label, aspect) => {
                  setVisionContext({ cluster: label, aspect })
                  openPage('Defect Vision')
                }}
              />
            )}

            {activeNav === 'Defect Vision' && (
              <DefectVisionPage
                key={visionContext ? `${visionContext.cluster}-${visionContext.aspect}` : 'default-vision'}
                {...sharedPageProps}
                analysis={analysisResult}
                initialContext={visionContext}
                gallery={visionGallery}
                onNewFinding={(item) =>
                  setVisionGallery((prev) => [item, ...prev.filter((g) => g.image_id !== item.image_id)])
                }
                onShowEvidence={(cl) => setSelectedCluster(cl)}
              />
            )}

            {activeNav === 'Compare' && (
              <ComparePage
                {...sharedPageProps}
                analysis={analysisResult}
                onAnalysisComplete={handleAnalysisComplete}
                onShowEvidence={(cl) => setSelectedCluster(cl)}
              />
            )}

            {activeNav === 'Model Evaluation' && (
              <ModelEvaluationPage {...sharedPageProps} />
            )}

            {activeNav === 'Analysis History' && (
              <HistoryPage
                {...sharedPageProps}
                history={history}
                loading={historyLoading}
                onLoadRun={(item) => {
                  setAnalysisResult(item.result)
                  setActiveNav('Insights')
                }}
              />
            )}

            {activeNav === 'Settings' && (
              <SettingsPage
                {...sharedPageProps}
                theme={theme}
                setTheme={setTheme}
                backendHealth={backendHealth}
                onRefreshHealth={checkHealth}
              />
            )}

            <footer className={`flex flex-wrap items-center justify-between gap-3 border-t pt-5 text-xs ${muted}`}>
              <span>InsightForge · AI Product Review Intelligence</span>
              <span className="flex items-center gap-1.5 text-emerald-500 font-medium">
                <CheckCircle2 size={14} /> Full End-to-End Analysis Pipeline
              </span>
            </footer>
          </div>
        </main>
      </div>

      {/* Traceable Evidence Modal */}
      {selectedCluster && (
        <EvidenceModal
          dark={dark}
          surface={surface}
          muted={muted}
          control={control}
          cluster={selectedCluster}
          analysis={analysisResult}
          imageFindings={visionGallery}
          onClose={() => setSelectedCluster(null)}
          onInspectVision={(cl) => {
            setVisionContext({ cluster: cl.label, aspect: cl.aspect })
            openPage('Defect Vision')
          }}
        />
      )}

    </div>
  )
}
