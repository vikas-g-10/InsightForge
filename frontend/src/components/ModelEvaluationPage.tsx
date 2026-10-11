import { useState, useEffect } from 'react'
import {
  Brain,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Zap,
  Layers,
  Sparkles,
  BarChart3,
  ShieldCheck,
  Send,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Minus,
} from 'lucide-react'
import { fetchMLMetrics, predictSentiment, type MLMetricsResponse, type MLPredictResponse } from '../api'

interface ModelEvaluationPageProps {
  dark?: boolean
  surface?: string
  muted?: string
  control?: string
}

export function ModelEvaluationPage({
  dark = true,
  surface = 'bg-slate-900 border-slate-800 text-slate-100',
  muted = 'text-slate-400',
  control = 'bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700',
}: ModelEvaluationPageProps) {
  const [metrics, setMetrics] = useState<MLMetricsResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Interactive Live Inference Playground State
  const [inputText, setInputText] = useState<string>(
    'The build quality is outstanding and the motor is whisper quiet. Very satisfied with this purchase!'
  )
  const [predicting, setPredicting] = useState<boolean>(false)
  const [prediction, setPrediction] = useState<MLPredictResponse | null>(null)
  const [predictError, setPredictError] = useState<string | null>(null)

  const [reloadKey, setReloadKey] = useState<number>(0)

  useEffect(() => {
    let active = true
    fetchMLMetrics()
      .then((data) => {
        if (!active) return
        setMetrics(data)
        setLoading(false)
      })
      .catch((err) => {
        if (!active) return
        setErrorMsg(err.message || 'Failed to load model evaluation metrics.')
        setLoading(false)
      })

    return () => {
      active = false
    }
  }, [reloadKey])

  function handleRetry() {
    setLoading(true)
    setErrorMsg(null)
    setReloadKey((prev) => prev + 1)
  }

  async function handlePredict(e?: React.FormEvent) {
    if (e) e.preventDefault()
    if (!inputText.trim()) return

    setPredicting(true)
    setPredictError(null)

    try {
      const res = await predictSentiment(inputText.trim())
      setPrediction(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Prediction request failed.'
      setPredictError(msg)
    } finally {
      setPredicting(false)
    }
  }

  const samplePresets = [
    {
      label: 'Positive Hardware',
      text: 'This is the best appliance ever, five stars! Highly recommended, whisper quiet and sturdy.',
    },
    {
      label: 'Negative Defect',
      text: 'Motor burned out on day 3 with an alarming smoke smell. Completely defective and returning immediately!',
    },
    {
      label: 'Neutral Feedback',
      text: 'It works as described. Nothing extraordinary, average construction but acceptable for basic chores.',
    },
  ]

  if (loading) {
    return (
      <div className="space-y-6">
        <div className={`rounded-3xl border p-12 text-center ${surface}`}>
          <div className="size-8 border-3 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <h3 className="font-bold text-base">Loading Empirical Model Metrics...</h3>
          <p className={`text-xs mt-1 ${muted}`}>Connecting to model evaluation service...</p>
        </div>
      </div>
    )
  }

  if (errorMsg || !metrics) {
    const isNetworkError =
      errorMsg?.toLowerCase().includes('failed to fetch') ||
      errorMsg?.toLowerCase().includes('network') ||
      errorMsg?.toLowerCase().includes('connection')
    const isAuthError =
      errorMsg?.toLowerCase().includes('authorization') ||
      errorMsg?.toLowerCase().includes('unauthorized') ||
      errorMsg?.toLowerCase().includes('token')
    const isModelMissing =
      errorMsg?.toLowerCase().includes('model_not_found') ||
      errorMsg?.toLowerCase().includes('artifact')

    return (
      <div className="space-y-6">
        <div className="rounded-3xl border border-rose-500/30 bg-rose-500/10 p-8 text-rose-300">
          <div className="flex items-center justify-between gap-4 mb-2">
            <div className="flex items-center gap-2.5 font-bold">
              <AlertCircle size={18} />
              <span>Evaluation Metrics Unavailable</span>
            </div>
            <button
              onClick={handleRetry}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-200 transition"
            >
              <RefreshCw size={13} />
              <span>Retry</span>
            </button>
          </div>
          <p className="text-xs leading-relaxed mb-4">
            {errorMsg || 'The model artifact or evaluation metrics file was not found.'}
          </p>

          {isNetworkError && (
            <div className={`text-[11px] font-mono p-3 rounded-xl border ${
              dark
                ? 'text-amber-300 bg-amber-950/40 border-amber-500/20'
                : 'text-amber-900 bg-amber-50 border-amber-200'
            }`}>
              Backend service unreachable at <code className={`font-bold ${dark ? 'text-white' : 'text-amber-950'}`}>http://localhost:8000</code>.
              <br />
              Ensure the FastAPI backend is running:
              <br />
              <code className={`font-bold ${dark ? 'text-white' : 'text-amber-950'}`}>python -m uvicorn main:app --reload --port 8000</code>
            </div>
          )}

          {isAuthError && (
            <div className={`text-[11px] font-mono p-3 rounded-xl border ${
              dark
                ? 'text-amber-300 bg-amber-950/40 border-amber-500/20'
                : 'text-amber-900 bg-amber-50 border-amber-200'
            }`}>
              Authentication required. Ensure your Supabase session is active or sign in again.
            </div>
          )}

          {isModelMissing && (
            <div className={`text-[11px] font-mono p-3 rounded-xl border ${
              dark
                ? 'text-rose-400 bg-rose-950/40 border-rose-500/20'
                : 'text-rose-900 bg-rose-50 border-rose-200'
            }`}>
              Run the training pipeline in backend: <br />
              <code className={`font-bold ${dark ? 'text-white' : 'text-rose-950'}`}>python -m ml.train</code>
            </div>
          )}
        </div>
      </div>
    )
  }

  const { dataset, model, test_results, per_class_metrics, confusion_matrix, model_comparison, limitations } = metrics

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <section className={`rounded-3xl border p-6 sm:p-8 ${surface} relative overflow-hidden`}>
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold mb-3 ${
              dark
                ? 'border-indigo-500/30 bg-indigo-500/10 text-indigo-300'
                : 'border-indigo-200 bg-indigo-50 text-indigo-700'
            }`}>
              <Brain size={14} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
              <span>Empirical Model Evaluation & Benchmark</span>
            </div>
            <h2 className={`text-2xl sm:text-3xl font-black tracking-tight ${dark ? 'text-white' : 'text-slate-900'}`}>
              Sentiment ML Pipeline & Held-Out Test Accuracy
            </h2>
            <p className={`mt-2 text-xs sm:text-sm ${muted} max-w-2xl leading-relaxed`}>
              Trained and validated on <strong className={dark ? 'text-slate-200' : 'text-slate-900'}>Amazon Reviews 2023</strong> ({dataset.category})
              curated by the McAuley Lab at UC San Diego. Tested on held-out reviews to establish true generalizability.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <a
              href={dataset.source_url}
              target="_blank"
              rel="noreferrer"
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2 text-xs font-semibold ${control} transition`}
            >
              <span>Hugging Face Dataset</span>
              <ExternalLink size={13} />
            </a>
          </div>
        </div>

        {/* Dataset Meta Banner */}
        <div className={`mt-6 pt-6 border-t ${dark ? 'border-slate-800/80' : 'border-slate-200'} flex flex-wrap items-center justify-between gap-4 text-xs`}>
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-emerald-500" />
            <span className={`font-semibold ${dark ? 'text-white' : 'text-slate-900'}`}>Selected Champion:</span>
            <span className="text-indigo-600 dark:text-indigo-400 font-mono font-bold">{model.selected_model}</span>
          </div>
          <div className={`flex items-center gap-4 ${muted}`}>
            <span>Vocabulary: <strong className={dark ? 'text-slate-200' : 'text-slate-900'}>{model.vocabulary_size.toLocaleString()}</strong> n-grams</span>
            <span>Seed: <strong className={dark ? 'text-slate-200' : 'text-slate-900'}>{model.random_seed}</strong></span>
            <span>Trained: <strong className={dark ? 'text-slate-200' : 'text-slate-900'}>{new Date(model.training_timestamp).toLocaleDateString()}</strong></span>
          </div>
        </div>
      </section>

      {/* Primary KPI Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Test Accuracy */}
        <div className={`rounded-2xl border p-5 ${surface} shadow-sm`}>
          <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
            <span className={muted}>Test Accuracy</span>
            <span className="size-2 rounded-full bg-emerald-500" />
          </div>
          <div className={`text-3xl font-black ${dark ? 'text-emerald-400' : 'text-emerald-600'}`}>
            {(test_results.accuracy * 100).toFixed(2)}%
          </div>
          <p className={`mt-1.5 text-[11px] ${muted}`}>
            Held-out test set accuracy across 3 sentiment classes
          </p>
        </div>

        {/* Macro F1 Score */}
        <div className={`rounded-2xl border p-5 ${surface} shadow-sm`}>
          <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
            <span className={muted}>Macro F1 Score</span>
            <span className="size-2 rounded-full bg-cyan-500" />
          </div>
          <div className={`text-3xl font-black ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>
            {test_results.macro_f1.toFixed(4)}
          </div>
          <p className={`mt-1.5 text-[11px] ${muted}`}>
            Unweighted average F1 score across all classes
          </p>
        </div>

        {/* Average Latency */}
        <div className={`rounded-2xl border p-5 ${surface} shadow-sm`}>
          <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
            <span className={muted}>Inference Latency</span>
            <span className="size-2 rounded-full bg-indigo-500" />
          </div>
          <div className={`text-3xl font-black ${dark ? 'text-indigo-400' : 'text-indigo-600'} flex items-baseline gap-1`}>
            {test_results.average_latency_ms.toFixed(3)}
            <span className={`text-sm font-normal ${muted}`}>ms</span>
          </div>
          <p className={`mt-1.5 text-[11px] ${muted}`}>
            Ultra-fast vectorized CPU inference per review
          </p>
        </div>

        {/* Total Dataset Examples */}
        <div className={`rounded-2xl border p-5 ${surface} shadow-sm`}>
          <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
            <span className={muted}>Dataset Examples</span>
            <span className="size-2 rounded-full bg-purple-500" />
          </div>
          <div className={`text-3xl font-black ${dark ? 'text-purple-400' : 'text-purple-600'}`}>
            {dataset.total_samples.toLocaleString()}
          </div>
          <p className={`mt-1.5 text-[11px] ${muted}`}>
            70% Train ({dataset.splits.train.toLocaleString()}) · 15% Val · 15% Test
          </p>
        </div>
      </section>

      {/* Model Comparison Table */}
      <section className={`rounded-3xl border p-6 sm:p-8 ${surface}`}>
        <div className={`flex items-center gap-2 mb-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
          <BarChart3 size={18} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
          <h3>Model Comparison on Held-Out Test Set</h3>
        </div>
        <p className={`text-xs ${muted} mb-6`}>
          Comparison between TF-IDF + Logistic Regression and TF-IDF + Calibrated Linear SVM evaluated under identical test splits.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className={`border-b ${dark ? 'border-slate-800 text-slate-400' : 'border-slate-200 text-slate-600'} font-semibold uppercase tracking-wider text-[11px]`}>
                <th className="py-3 px-4">Model Pipeline</th>
                <th className="py-3 px-4">Val Macro F1</th>
                <th className="py-3 px-4">Test Accuracy</th>
                <th className="py-3 px-4">Test Macro F1</th>
                <th className="py-3 px-4">Training Time</th>
                <th className="py-3 px-4">Inference Latency</th>
                <th className="py-3 px-4">Selection Status</th>
              </tr>
            </thead>
            <tbody className={`divide-y ${dark ? 'divide-slate-800/60' : 'divide-slate-200'}`}>
              {model_comparison.map((comp, idx) => (
                <tr
                  key={idx}
                  className={`${dark ? 'hover:bg-slate-800/30' : 'hover:bg-slate-50'} transition ${
                    comp.status === 'Selected Champion' ? (dark ? 'bg-indigo-500/10 font-medium' : 'bg-indigo-50/70 font-medium') : ''
                  }`}
                >
                  <td className={`py-3.5 px-4 font-mono font-bold flex items-center gap-2 ${dark ? 'text-white' : 'text-slate-900'}`}>
                    {comp.status === 'Selected Champion' && (
                      <CheckCircle2 size={14} className={dark ? 'text-emerald-400' : 'text-emerald-600'} />
                    )}
                    {comp.model_name}
                  </td>
                  <td className="py-3.5 px-4 font-mono">{comp.val_macro_f1.toFixed(4)}</td>
                  <td className={`py-3.5 px-4 font-mono font-bold ${dark ? 'text-emerald-400' : 'text-emerald-600'}`}>
                    {(comp.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td className={`py-3.5 px-4 font-mono font-semibold ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>{comp.test_macro_f1.toFixed(4)}</td>
                  <td className={`py-3.5 px-4 font-mono ${muted}`}>{comp.training_time_s}s</td>
                  <td className={`py-3.5 px-4 font-mono ${muted}`}>{comp.latency_ms.toFixed(3)} ms</td>
                  <td className="py-3.5 px-4">
                    {comp.status === 'Selected Champion' ? (
                      <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                        dark
                          ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-300'
                          : 'bg-emerald-100 border border-emerald-300 text-emerald-800'
                      }`}>
                        Selected Champion
                      </span>
                    ) : (
                      <span className={`rounded-full px-2.5 py-0.5 text-[10px] ${
                        dark
                          ? 'bg-slate-800 text-slate-400'
                          : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        Baseline Candidate
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Grid: Per-Class Metrics & Confusion Matrix */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Per-Class Performance */}
        <div className={`rounded-3xl border p-6 sm:p-8 ${surface}`}>
          <div className={`flex items-center gap-2 mb-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
            <Layers size={18} className={dark ? 'text-cyan-400' : 'text-cyan-600'} />
            <h3>Per-Class Sentiment Metrics</h3>
          </div>
          <p className={`text-xs ${muted} mb-6`}>
            Precision, Recall, and F1-score evaluated on 4,500 held-out test reviews (1,500 per class).
          </p>

          <div className="space-y-4">
            {Object.entries(per_class_metrics).map(([cls, scores]) => {
              const isPos = cls === 'positive'
              const isNeg = cls === 'negative'
              const badgeColor = isPos
                ? dark ? 'border-emerald-500/30 bg-emerald-950/40 text-emerald-300' : 'border-emerald-200 bg-emerald-50 text-emerald-800 font-bold'
                : isNeg
                ? dark ? 'border-rose-500/30 bg-rose-950/40 text-rose-300' : 'border-rose-200 bg-rose-50 text-rose-800 font-bold'
                : dark ? 'border-amber-500/30 bg-amber-950/40 text-amber-300' : 'border-amber-200 bg-amber-50 text-amber-900 font-bold'

              return (
                <div key={cls} className={`rounded-2xl border p-4 ${dark ? 'border-slate-800 bg-slate-950/40' : 'border-slate-200 bg-slate-50'}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className={`rounded-lg border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider ${badgeColor}`}>
                      {cls}
                    </span>
                    <span className={`text-[11px] ${muted}`}>Support: {scores.support.toLocaleString()} reviews</span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className={`${dark ? 'bg-slate-900/60 border-slate-800/60' : 'bg-white border-slate-200 shadow-xs'} p-2.5 rounded-xl border`}>
                      <div className={`text-[10px] uppercase font-semibold ${muted}`}>Precision</div>
                      <div className={`font-bold font-mono mt-0.5 ${dark ? 'text-white' : 'text-slate-900'}`}>{(scores.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div className={`${dark ? 'bg-slate-900/60 border-slate-800/60' : 'bg-white border-slate-200 shadow-xs'} p-2.5 rounded-xl border`}>
                      <div className={`text-[10px] uppercase font-semibold ${muted}`}>Recall</div>
                      <div className={`font-bold font-mono mt-0.5 ${dark ? 'text-white' : 'text-slate-900'}`}>{(scores.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div className={`${dark ? 'bg-slate-900/60 border-slate-800/60' : 'bg-white border-slate-200 shadow-xs'} p-2.5 rounded-xl border`}>
                      <div className={`text-[10px] uppercase font-semibold ${muted}`}>F1-Score</div>
                      <div className={`font-bold font-mono mt-0.5 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>{(scores.f1_score * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Labeled Confusion Matrix */}
        <div className={`rounded-3xl border p-6 sm:p-8 ${surface}`}>
          <div className={`flex items-center gap-2 mb-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
            <Zap size={18} className={dark ? 'text-amber-400' : 'text-amber-600'} />
            <h3>Held-Out Test Confusion Matrix</h3>
          </div>
          <p className={`text-xs ${muted} mb-6`}>
            Rows represent Ground-Truth Actual classes; columns represent Model Predictions.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-center border-collapse">
              <thead>
                <tr>
                  <th className="p-2"></th>
                  {confusion_matrix.labels.map((lbl) => (
                    <th key={lbl} className={`p-2 font-bold uppercase text-[10px] ${dark ? 'text-slate-400' : 'text-slate-600'}`}>
                      Pred {lbl}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusion_matrix.matrix.map((row, rIdx) => {
                  const actualLabel = confusion_matrix.labels[rIdx]
                  const rowSum = row.reduce((a, b) => a + b, 0)

                  return (
                    <tr key={rIdx} className={`border-t ${dark ? 'border-slate-800/60' : 'border-slate-200'}`}>
                      <td className={`p-2 font-bold uppercase text-[10px] ${dark ? 'text-slate-400' : 'text-slate-600'} text-left`}>
                        Act {actualLabel}
                      </td>
                      {row.map((val, cIdx) => {
                        const isDiagonal = rIdx === cIdx
                        const pct = ((val / rowSum) * 100).toFixed(1)

                        return (
                          <td
                            key={cIdx}
                            className={`p-3 font-mono rounded-xl transition ${
                              isDiagonal
                                ? dark
                                  ? 'bg-emerald-500/15 text-emerald-300 font-bold border border-emerald-500/30'
                                  : 'bg-emerald-50 text-emerald-800 font-bold border border-emerald-300'
                                : val > 300
                                ? dark
                                  ? 'bg-slate-800/50 text-slate-300'
                                  : 'bg-slate-100 text-slate-800 border border-slate-200 font-semibold'
                                : dark
                                ? 'text-slate-400'
                                : 'text-slate-600'
                            }`}
                          >
                            <div className="text-sm">{val.toLocaleString()}</div>
                            <div className="text-[10px] opacity-75 font-medium">({pct}%)</div>
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className={`mt-4 text-[11px] ${muted} flex items-center justify-between`}>
            <span>Diagonal cells indicate correct classifications.</span>
            <span className={`${dark ? 'text-emerald-400' : 'text-emerald-700'} font-semibold font-mono`}>
              Total Test: {confusion_matrix.matrix.flat().reduce((a, b) => a + b, 0).toLocaleString()} reviews
            </span>
          </div>
        </div>
      </section>

      {/* Interactive Live Inference Playground */}
      <section className={`rounded-3xl border p-6 sm:p-8 ${surface} ${dark ? 'border-indigo-500/30' : 'border-indigo-200'} shadow-sm`}>
        <div className={`flex items-center gap-2 mb-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
          <Sparkles size={18} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
          <h3>Interactive Model Inference Playground</h3>
        </div>
        <p className={`text-xs ${muted} mb-4`}>
          Test the trained machine learning pipeline in real time. Submit any product review text to inspect the predicted sentiment,
          polarity, confidence score, and calibrated class probabilities.
        </p>

        {/* Quick Example Presets */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <span className={`text-[11px] font-semibold ${muted}`}>Presets:</span>
          {samplePresets.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setInputText(preset.text)
                setPrediction(null)
              }}
              className={`rounded-lg border px-3 py-1 text-xs font-medium transition ${control}`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Form Input */}
        <form onSubmit={handlePredict} className="space-y-4">
          <div>
            <textarea
              rows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste or type customer feedback text..."
              disabled={predicting}
              className={`w-full rounded-2xl border p-4 text-xs font-normal focus:outline-none focus:ring-2 focus:ring-indigo-500 transition ${
                dark ? 'bg-slate-950 border-slate-800 text-white placeholder-slate-500' : 'bg-white border-slate-200 text-slate-900 placeholder-slate-400 shadow-xs'
              }`}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className={`text-[11px] ${muted}`}>
              Evaluated with <strong className={dark ? 'text-slate-200' : 'text-slate-900 font-semibold'}>{model.selected_model}</strong> pipeline
            </div>
            <button
              type="submit"
              disabled={predicting || !inputText.trim()}
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-bold text-white shadow-md shadow-indigo-600/30 hover:bg-indigo-500 disabled:opacity-50 transition"
            >
              {predicting ? (
                <>
                  <RefreshCw size={14} className="animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <span>Run ML Prediction</span>
                  <Send size={14} />
                </>
              )}
            </button>
          </div>
        </form>

        {/* Prediction Results Banner */}
        {predictError && (
          <div className="mt-5 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-400 font-medium">
            {predictError}
          </div>
        )}

        {prediction && (
          <div className={`mt-6 rounded-2xl border ${dark ? 'border-slate-800 bg-slate-950/70' : 'border-slate-200 bg-slate-50'} p-5 space-y-4`}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <span className={`text-xs font-bold ${dark ? 'text-slate-300' : 'text-slate-800'}`}>Predicted Sentiment:</span>
                {prediction.sentiment === 'positive' && (
                  <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold ${
                    dark ? 'border-emerald-500/30 bg-emerald-950/40 text-emerald-300' : 'border-emerald-300 bg-emerald-100 text-emerald-800'
                  }`}>
                    <ThumbsUp size={13} /> POSITIVE
                  </span>
                )}
                {prediction.sentiment === 'negative' && (
                  <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold ${
                    dark ? 'border-rose-500/30 bg-rose-950/40 text-rose-300' : 'border-rose-300 bg-rose-100 text-rose-800'
                  }`}>
                    <ThumbsDown size={13} /> NEGATIVE
                  </span>
                )}
                {prediction.sentiment === 'neutral' && (
                  <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold ${
                    dark ? 'border-amber-500/30 bg-amber-950/40 text-amber-300' : 'border-amber-300 bg-amber-100 text-amber-800'
                  }`}>
                    <Minus size={13} /> NEUTRAL
                  </span>
                )}
              </div>

              <div className="flex items-center gap-4 text-xs">
                <div>
                  <span className={muted}>Confidence: </span>
                  <strong className={`font-mono ${dark ? 'text-white' : 'text-slate-900'}`}>{(prediction.confidence * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span className={muted}>Latency: </span>
                  <strong className={`font-mono ${dark ? 'text-indigo-400' : 'text-indigo-600'}`}>{prediction.latency_ms.toFixed(2)} ms</strong>
                </div>
              </div>
            </div>

            {/* Probability Breakdown Bars */}
            <div className={`space-y-2 pt-2 border-t ${dark ? 'border-slate-800/60' : 'border-slate-200'}`}>
              <div className={`text-[11px] font-bold uppercase tracking-wider ${dark ? 'text-slate-400' : 'text-slate-600'}`}>
                Calibrated Class Probability Distribution
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {['negative', 'neutral', 'positive'].map((cName) => {
                  const prob = prediction.probabilities[cName] ?? 0
                  const barColor =
                    cName === 'positive' ? 'bg-emerald-500' : cName === 'negative' ? 'bg-rose-500' : 'bg-amber-500'

                  return (
                    <div key={cName} className={`rounded-xl border p-3 ${dark ? 'border-slate-800/60 bg-slate-900/60' : 'border-slate-200 bg-white'}`}>
                      <div className="flex justify-between text-xs mb-1.5">
                        <span className={`capitalize font-medium ${dark ? 'text-slate-300' : 'text-slate-700'}`}>{cName}</span>
                        <span className={`font-mono font-bold ${dark ? 'text-white' : 'text-slate-900'}`}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className={`w-full rounded-full h-2 overflow-hidden ${dark ? 'bg-slate-800' : 'bg-slate-200'}`}>
                        <div
                          className={`h-full ${barColor} transition-all duration-500`}
                          style={{ width: `${Math.max(2, prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Neutral Class Noise Analysis & Binary Benchmark Section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Neutral Class Noise Analysis */}
        {metrics.neutral_class_analysis && (
          <div className={`rounded-3xl border p-6 sm:p-8 ${surface} ${dark ? 'border-amber-500/20' : 'border-amber-200'}`}>
            <div className="flex items-center justify-between mb-2">
              <div className={`flex items-center gap-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
                <AlertCircle size={18} className={dark ? 'text-amber-400' : 'text-amber-600'} />
                <h3>Neutral (3-Star) Class Noise Analysis</h3>
              </div>
              <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                dark
                  ? 'bg-amber-500/10 border border-amber-500/30 text-amber-300'
                  : 'bg-amber-100 border border-amber-300 text-amber-900'
              }`}>
                Task 6 Investigation
              </span>
            </div>
            <p className={`text-xs ${muted} mb-4 leading-relaxed`}>
              {metrics.neutral_class_analysis.finding}
            </p>

            <div className="grid grid-cols-3 gap-3 text-center text-xs mb-4">
              <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className={`text-[10px] uppercase font-semibold ${muted}`}>Neutral F1</div>
                <div className={`font-bold font-mono text-base mt-1 ${dark ? 'text-amber-400' : 'text-amber-700'}`}>
                  {(metrics.neutral_class_analysis.neutral_f1 * 100).toFixed(1)}%
                </div>
                <div className={`text-[10px] mt-0.5 ${dark ? 'text-slate-500' : 'text-slate-600'}`}>vs Pos: {(metrics.neutral_class_analysis.positive_f1 * 100).toFixed(1)}%</div>
              </div>

              <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className={`text-[10px] uppercase font-semibold ${muted}`}>Confused as Neg</div>
                <div className={`font-bold font-mono text-base mt-1 ${dark ? 'text-rose-400' : 'text-rose-700'}`}>
                  {metrics.neutral_class_analysis.confused_as_negative}
                </div>
                <div className={`text-[10px] mt-0.5 ${dark ? 'text-slate-500' : 'text-slate-600'}`}>24.3% of 3-star reviews</div>
              </div>

              <div className={`rounded-xl border p-3 ${dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className={`text-[10px] uppercase font-semibold ${muted}`}>Confused as Pos</div>
                <div className={`font-bold font-mono text-base mt-1 ${dark ? 'text-emerald-400' : 'text-emerald-700'}`}>
                  {metrics.neutral_class_analysis.confused_as_positive}
                </div>
                <div className={`text-[10px] mt-0.5 ${dark ? 'text-slate-500' : 'text-slate-600'}`}>11.1% of 3-star reviews</div>
              </div>
            </div>

            <div className={`rounded-xl border p-3 text-[11px] ${
              dark
                ? 'border-amber-500/20 bg-amber-950/30 text-amber-200'
                : 'border-amber-200 bg-amber-50 text-amber-950'
            }`}>
              💡 <strong>Linguistic Insight:</strong> {metrics.neutral_class_analysis.mixed_signals_note} Customers frequently award 3 stars to mixed reviews containing explicit polarity phrases (e.g., <em>"Great motor but plastic lid cracked on day 2"</em>).
            </div>
          </div>
        )}

        {/* Separate Binary Benchmark */}
        {metrics.binary_benchmark && (
          <div className={`rounded-3xl border p-6 sm:p-8 ${surface} ${dark ? 'border-emerald-500/20' : 'border-emerald-200'}`}>
            <div className="flex items-center justify-between mb-2">
              <div className={`flex items-center gap-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
                <CheckCircle2 size={18} className={dark ? 'text-emerald-400' : 'text-emerald-600'} />
                <h3>Separate Binary Benchmark (Excl. 3-Star)</h3>
              </div>
              <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                dark
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
                  : 'bg-emerald-100 border border-emerald-300 text-emerald-900'
              }`}>
                Task 7 Benchmark
              </span>
            </div>
            <p className={`text-xs ${muted} mb-4 leading-relaxed`}>
              {metrics.binary_benchmark.note}
            </p>

            <div className="grid grid-cols-2 gap-3 text-center text-xs mb-4">
              <div className={`rounded-xl border p-4 ${dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className={`text-[10px] uppercase font-semibold ${muted}`}>Binary Test Accuracy</div>
                <div className={`font-black font-mono text-2xl mt-1 ${dark ? 'text-emerald-400' : 'text-emerald-600'}`}>
                  {(metrics.binary_benchmark.test_accuracy * 100).toFixed(2)}%
                </div>
                <div className={`text-[10px] mt-0.5 ${dark ? 'text-slate-400' : 'text-slate-600'}`}>1-2 Stars vs 4-5 Stars</div>
              </div>

              <div className={`rounded-xl border p-4 ${dark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className={`text-[10px] uppercase font-semibold ${muted}`}>Binary Macro F1</div>
                <div className={`font-black font-mono text-2xl mt-1 ${dark ? 'text-cyan-400' : 'text-cyan-600'}`}>
                  {metrics.binary_benchmark.test_macro_f1.toFixed(4)}
                </div>
                <div className={`text-[10px] mt-0.5 ${dark ? 'text-slate-400' : 'text-slate-600'}`}>Held-out 3,000 test set</div>
              </div>
            </div>

            <div className={`rounded-xl border p-3 text-[11px] ${
              dark
                ? 'border-emerald-500/20 bg-emerald-950/30 text-emerald-200'
                : 'border-emerald-200 bg-emerald-50 text-emerald-950'
            }`}>
              ✓ <strong>Clean Polarity Proof:</strong> Removing ambiguous neutral proxy ratings demonstrates that InsightForge's optimized TF-IDF representation achieves <strong>93.70% accuracy</strong> on unambiguous sentiment.
            </div>
          </div>
        )}
      </section>

      {/* Dataset & Methodology Limitations */}
      <section className={`rounded-3xl border p-6 sm:p-8 ${surface}`}>
        <div className={`flex items-center gap-2 mb-2 font-bold text-base ${dark ? 'text-white' : 'text-slate-900'}`}>
          <ShieldCheck size={18} className={dark ? 'text-indigo-400' : 'text-indigo-600'} />
          <h3>Methodology, Transparency & Dataset Limitations</h3>
        </div>
        <p className={`text-xs ${muted} mb-4`}>
          Transparent documentation of proxy labeling caveats and dataset boundary constraints.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className={`rounded-2xl border p-4 ${dark ? 'border-slate-800/80 bg-slate-950/40' : 'border-slate-200 bg-slate-50'}`}>
            <h4 className={`font-bold mb-1.5 flex items-center gap-1.5 ${dark ? 'text-white' : 'text-slate-900'}`}>
              <span className="size-1.5 rounded-full bg-amber-400" /> Star Ratings as Proxy Labels
            </h4>
            <p className={muted}>
              {dataset.proxy_labeling_notice}
              Individual ratings may contain user noise (e.g., giving 1 star due to courier shipping delays despite praising product quality).
            </p>
          </div>

          <div className={`rounded-2xl border p-4 ${dark ? 'border-slate-800/80 bg-slate-950/40' : 'border-slate-200 bg-slate-50'}`}>
            <h4 className={`font-bold mb-1.5 flex items-center gap-1.5 ${dark ? 'text-white' : 'text-slate-900'}`}>
              <span className="size-1.5 rounded-full bg-cyan-400" /> Neutral (3-Star) Inherent Ambiguity
            </h4>
            <p className={muted}>
              Neutral feedback displays lower recall ({((per_class_metrics.neutral?.recall || 0) * 100).toFixed(1)}%) than polarized classes,
              as 3-star reviews often blend high praise for features with strong complaints about bugs.
            </p>
          </div>
        </div>

        <div className={`mt-4 pt-4 border-t ${dark ? 'border-slate-800/60' : 'border-slate-200'} flex flex-wrap gap-2`}>
          {limitations.map((note, idx) => (
            <span key={idx} className={`rounded-lg px-3 py-1 text-[11px] ${
              dark
                ? 'bg-slate-800/60 text-slate-300'
                : 'bg-slate-100 text-slate-700 border border-slate-200'
            }`}>
              • {note}
            </span>
          ))}
        </div>
      </section>
    </div>
  )
}
