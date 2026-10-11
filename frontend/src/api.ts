/**
 * InsightForge Frontend API Client
 * Connects to FastAPI backend endpoints:
 * - GET /health
 * - POST /api/analyze
 * - POST /api/analyze/file
 * - POST /api/issues
 * - POST /api/simulate
 */

import { supabase, isSupabaseConfigured } from './supabase'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || ''

export interface HealthResponse {
  status: 'ok'
  mode: 'lite' | 'full'
  db_connected: boolean
}

export interface ReviewInputItem {
  id?: string | number
  review_id?: string | number
  text: string
  rating?: number | string | null
  date?: string | null
  version?: string | null
  product?: string | null
}

export interface ClauseAnalysisResult {
  clause_id: string | number
  text: string
  aspect?: string | null
  aspects: string[]
  score: number
  polarity: 'pos' | 'neu' | 'neg'
  label: string
  confidence: number
  sarcasm_suspect: boolean
  rating_conflict: boolean
  language: string
  start: number
  end: number
  cluster_id?: string | null
}

export interface SafetyMatchDetail {
  category: string
  tier: 'none' | 'medium' | 'high' | 'critical'
  term: string
  phrase: string
  excerpt: string
  suggested_action?: string | null
}

export interface SafetyAnalysisResult {
  tier: 'none' | 'medium' | 'high' | 'critical'
  is_alert: boolean
  matches: SafetyMatchDetail[]
  suggested_action?: string | null
}

export interface ReviewAnalysisResult {
  review_id: string
  original_text: string
  normalized_text: string
  rating?: number | null
  rating_source?: string | null
  pseudo_rating?: number | null
  date?: string | null
  version?: string | null
  product?: string | null
  language: string
  language_badge: string
  language_confidence: number
  is_duplicate: boolean
  duplicate_of?: string | null
  truncated: boolean
  sentiment_score: number
  polarity: 'pos' | 'neu' | 'neg'
  label: string
  sarcasm_suspect: boolean
  rating_conflict: boolean
  aspects: string[]
  safety: SafetyAnalysisResult
  clauses: ClauseAnalysisResult[]
  status: 'analyzed' | 'empty' | 'error'
}

export interface AnalysisSummary {
  total_reviews: number
  analyzed_reviews: number
  unique_reviews: number
  duplicate_count: number
  total_clauses: number
  languages: Record<string, number>
  sentiment_breakdown: Record<string, number>
  safety_breakdown: Record<string, number>
  sarcasm_count: number
  rating_conflict_count: number
  aspect_mentions: Record<string, number>
  warnings: string[]
}

export interface AnalysisMetadata {
  mode: 'lite' | 'full'
  requested_mode?: string | null
  processing_time_ms: number
  summary: AnalysisSummary
}

export interface IssueClusterClauseEvidence {
  review_id: string
  clause_id: string | number
  text: string
  score: number
  polarity: 'pos' | 'neu' | 'neg'
  safety_tier: string
}

export interface IssueCluster {
  cluster_id: string
  aspect: string
  polarity: 'pos' | 'neu' | 'neg'
  label: string
  example_clause_id: string | number
  example_quote: string
  size: number
  review_count: number
  clause_ids: (string | number)[]
  review_ids: string[]
  evidence: IssueClusterClauseEvidence[]
  safety_tier: string
  is_safety_alert: boolean
  average_rating?: number | null
  average_sentiment: number
  team?: string | null
  priority?: string | null
}

export interface IssuePriorityInsight {
  aspect: string
  polarity: 'pos' | 'neu' | 'neg'
  mentions: number
  complaint_count: number
  pct_negative: number
  beta: number
  ci_low: number
  ci_high: number
  damage: number
  priority_score: number
  priority_rank: number
  confidence_label: 'ok' | 'low'
  single_cause_unhappy: number
  team?: string | null
  priority_code?: string | null
  safety_tier: string
  is_sentiment_estimate: boolean
}

export interface AnalyzeResponse {
  reviews: ReviewAnalysisResult[]
  meta: AnalysisMetadata
  clusters: IssueCluster[]
  insights: IssuePriorityInsight[]
}

export interface IssuesResponse {
  clusters: IssueCluster[]
  insights: IssuePriorityInsight[]
  total_clusters: number
  top_aspects: string[]
  warnings: string[]
}

export interface AspectSimulateImpact {
  aspect: string
  fraction_fixed: number
  uplift: number
  unhappy_resolved: number
  complaints_resolved: number
}

export interface SimulateRequest {
  fixes: Record<string, number>
  reviews: ReviewInputItem[]
}

export interface SimulateResponse {
  current_rating: number
  projected_rating: number
  delta: number
  unhappy_reviews_resolved: number
  per_aspect: AspectSimulateImpact[]
  is_sentiment_estimate: boolean
  confidence_label: 'ok' | 'low'
  notes: string[]
}

export interface ApiError {
  code: string
  message: string
  columns?: string[]
}

function resolveUrl(path: string): string {
  if (API_BASE_URL) {
    return `${API_BASE_URL.replace(/\/+$/, '')}${path}`
  }
  return path
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errBody: ApiError
    try {
      const json = await res.json()
      errBody = json.error || { code: `http_${res.status}`, message: res.statusText }
    } catch {
      errBody = { code: `http_${res.status}`, message: res.statusText || 'Unknown error' }
    }
    throw new Error(errBody.message || errBody.code)
  }
  return (await res.json()) as T
}

export async function authenticatedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const url = resolveUrl(path)
  const headers = new Headers(init.headers || {})

  if (isSupabaseConfigured) {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (session) {
        const now = Math.floor(Date.now() / 1000)
        let token = session.access_token
        if (session.expires_at && session.expires_at - now < 30) {
          const { data: refreshed } = await supabase.auth.refreshSession()
          if (refreshed.session) {
            token = refreshed.session.access_token
          }
        }
        headers.set('Authorization', `Bearer ${token}`)
      }
    } catch (err) {
      console.warn('Error reading auth session for API request:', err)
    }
  }

  let res = await fetch(url, { ...init, headers })

  if (res.status === 401 && isSupabaseConfigured) {
    try {
      const { data: refreshed, error } = await supabase.auth.refreshSession()
      if (!error && refreshed.session?.access_token) {
        headers.set('Authorization', `Bearer ${refreshed.session.access_token}`)
        res = await fetch(url, { ...init, headers })
      }
    } catch {
      // ignore
    }
  }

  return res
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await authenticatedFetch('/health')
  return handleResponse<HealthResponse>(res)
}

export async function analyzeReviews(
  reviews: ReviewInputItem[],
  mode: string = 'lite',
): Promise<AnalyzeResponse> {
  const res = await authenticatedFetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviews, mode }),
  })
  return handleResponse<AnalyzeResponse>(res)
}

export async function analyzeFile(
  file: File,
  textColumn?: string,
  mode: string = 'lite',
): Promise<AnalyzeResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (textColumn) {
    formData.append('text_column', textColumn)
  }
  formData.append('mode', mode)

  const res = await authenticatedFetch('/api/analyze/file', {
    method: 'POST',
    body: formData,
  })
  return handleResponse<AnalyzeResponse>(res)
}

export async function fetchIssues(
  reviews: ReviewInputItem[],
  mode: string = 'lite',
): Promise<IssuesResponse> {
  const res = await authenticatedFetch('/api/issues', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviews, mode }),
  })
  return handleResponse<IssuesResponse>(res)
}

export async function simulateFixes(
  fixes: Record<string, number>,
  reviews: ReviewInputItem[],
): Promise<SimulateResponse> {
  const res = await authenticatedFetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fixes, reviews }),
  })
  return handleResponse<SimulateResponse>(res)
}

export interface EmergingIssueEvidence {
  review_id: string
  clause_id: string | number
  text: string
  rating?: number | null
  date?: string | null
  safety_tier: string
}

export interface EmergingIssueFinding {
  issue_id: string
  label: string
  category: string
  aspect: string
  polarity: 'neg' | 'neu' | 'pos'
  first_observed_date?: string | null
  recent_count: number
  baseline_count: number
  recent_share: number
  baseline_share: number
  change_count: number
  change_share: number
  velocity_ratio?: number | null
  severity: 'critical' | 'high' | 'medium' | 'low'
  priority: number
  status: 'new' | 'accelerating' | 'steady' | 'insufficient_data'
  confidence: 'high' | 'medium' | 'low'
  limitations: string[]
  is_safety_escalated: boolean
  evidence: EmergingIssueEvidence[]
}

export interface TimeWindowInfo {
  recent_start?: string | null
  recent_end?: string | null
  baseline_start?: string | null
  baseline_end?: string | null
  total_recent_reviews: number
  total_baseline_reviews: number
  undated_reviews: number
  has_sufficient_timeline: boolean
}

export interface EmergingIssuesResponse {
  total_emerging_issues: number
  new_issues_count: number
  accelerating_issues_count: number
  safety_escalations_count: number
  insufficient_data_count: number
  time_windows: TimeWindowInfo
  findings: EmergingIssueFinding[]
  warnings: string[]
}

export async function fetchEmergingIssues(
  reviews: ReviewInputItem[],
  recent_days: number = 14,
  baseline_days: number = 30,
  split_date?: string,
  min_recent_count: number = 2,
  mode: string = 'lite',
): Promise<EmergingIssuesResponse> {
  const res = await authenticatedFetch('/api/emerging-issues', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reviews,
      recent_days,
      baseline_days,
      split_date,
      min_recent_count,
      mode,
    }),
  })
  return handleResponse<EmergingIssuesResponse>(res)
}

export interface VisionHealthResponse {
  status: 'ready' | 'unavailable' | 'model_not_installed' | 'vision_unsupported'
  provider: string
  base_url: string
  model: string
  vision_supported: boolean
  message: string
  available_models: string[]
}

export interface ImageDefectAnalysis {
  image_id: string
  filename: string
  mime_type: string
  file_size_bytes: number
  visual_description: string
  potential_defect_category: string
  visual_observations: string[]
  severity: 'none' | 'low' | 'medium' | 'high' | 'critical'
  confidence: number
  relevant_aspect?: string | null
  relevant_issue_cluster?: string | null
  limitations: string[]
  suggested_manual_checks: string[]
  is_defect_observed: boolean
  thumbnail_data_url?: string | null
  raw_model_response?: string | null
}

export async function fetchVisionHealth(): Promise<VisionHealthResponse> {
  const res = await authenticatedFetch('/api/vision/health')
  return handleResponse<VisionHealthResponse>(res)
}

export async function analyzeProductImage(
  file: File,
  clusterContext?: string,
  aspectContext?: string,
): Promise<ImageDefectAnalysis> {
  const formData = new FormData()
  formData.append('file', file)
  if (clusterContext) formData.append('cluster_context', clusterContext)
  if (aspectContext) formData.append('aspect_context', aspectContext)

  const res = await authenticatedFetch('/api/vision/analyze', {
    method: 'POST',
    body: formData,
  })
  return handleResponse<ImageDefectAnalysis>(res)
}

export interface EvidenceReference {
  source_type: 'review' | 'image' | 'temporal'
  reference_id: string
  excerpt_or_observation: string
  sentiment_score?: number | null
  metadata?: Record<string, string | number | boolean | null>
}

export interface HypothesisItem {
  hypothesis_id: string
  rank: number
  proposed_cause: string
  category: string
  confidence_score: number
  confidence_level: 'high' | 'medium' | 'low'
  explanation: string
  supporting_evidence: EvidenceReference[]
  contradicting_evidence: EvidenceReference[]
  missing_information: string[]
  verification_steps: string[]
  falsification_criteria: string
  is_hypothesis_only: boolean
}

export interface RootCauseAnalysisResult {
  cluster_id: string
  issue_label: string
  aspect: string
  status: 'hypotheses_generated' | 'insufficient_evidence'
  has_sufficient_evidence: boolean
  summary: string
  hypotheses: HypothesisItem[]
  insufficient_evidence_reason?: string | null
  total_supporting_reviews: number
  total_contradicting_reviews: number
  associated_images_count: number
  temporal_pattern_summary?: string | null
  warnings: string[]
  disclaimer: string
}

export interface RootCauseBatchResponse {
  results: RootCauseAnalysisResult[]
  total_clusters_analyzed: number
  clusters_with_hypotheses: number
  clusters_with_insufficient_evidence: number
  warnings: string[]
}

export async function fetchRootCauseHypotheses(
  reviews: ReviewInputItem[],
  imageFindings: ImageDefectAnalysis[] = [],
  clusterId?: string,
  mode: string = 'lite',
): Promise<RootCauseBatchResponse> {
  const res = await authenticatedFetch('/api/root-cause/hypotheses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reviews,
      image_findings: imageFindings,
      cluster_id: clusterId,
      mode,
    }),
  })
  return handleResponse<RootCauseBatchResponse>(res)
}

// --- Comparison Types (Feature 1 & Feature 2) -----------------------------

export interface SentimentProportions {
  pos_count: number
  neu_count: number
  neg_count: number
  pos_ratio: number
  neu_ratio: number
  neg_ratio: number
}

export interface AspectComparisonItem {
  aspect: string
  aspect_label: string
  count_a: number
  complaint_count_a: number
  negative_rate_a: number
  avg_sentiment_a: number
  count_b: number
  complaint_count_b: number
  negative_rate_b: number
  avg_sentiment_b: number
  complaint_diff: number
  negative_rate_diff: number
  status: 'improved' | 'worsened' | 'newly_observed' | 'not_observed' | 'resolved' | 'unchanged'
  sample_note?: string | null
}

export interface IssueClusterComparisonItem {
  cluster_label: string
  aspect: string
  count_a: number
  share_a: number
  priority_a?: string | null
  safety_tier_a: string
  count_b: number
  share_b: number
  priority_b?: string | null
  safety_tier_b: string
  status: 'improved' | 'worsened' | 'newly_observed' | 'not_observed' | 'resolved' | 'unchanged'
  sample_limitation?: string | null
  evidence_a: EvidenceReference[]
  evidence_b: EvidenceReference[]
}

export interface VersionMetricsSummary {
  version: string
  review_count: number
  rating_count: number
  average_rating?: number | null
  sentiment_proportions: SentimentProportions
  average_sentiment: number
  safety_alert_count: number
  safety_tiers: string[]
}

export interface VersionCompareResponse {
  product?: string | null
  version_a: VersionMetricsSummary
  version_b: VersionMetricsSummary
  aspect_comparisons: AspectComparisonItem[]
  cluster_comparisons: IssueClusterComparisonItem[]
  safety_changes: string[]
  summary_of_changes: string
  disclaimer: string
  is_sparse_sample: boolean
  warnings: string[]
}

export interface ProductMetricsSummary {
  product: string
  review_count: number
  rating_count: number
  average_rating?: number | null
  sentiment_proportions: SentimentProportions
  average_sentiment: number
  safety_alert_count: number
  safety_tiers: string[]
}

export interface ProductAspectStrengthItem {
  aspect: string
  aspect_label: string
  leader: 'product_a' | 'product_b' | 'tie'
  sentiment_a: number
  negative_rate_a: number
  complaints_a: number
  sentiment_b: number
  negative_rate_b: number
  complaints_b: number
  strength_summary: string
}

export interface ProductCompareResponse {
  product_a: ProductMetricsSummary
  product_b: ProductMetricsSummary
  focus_aspect?: string | null
  aspect_strengths: ProductAspectStrengthItem[]
  common_complaint_themes: string[]
  unique_issues_a: string[]
  unique_issues_b: string[]
  prominent_safety_alerts: Array<{
    product: string
    review_id: string
    tier: string
    excerpt: string
  }>
  key_evidence_a: EvidenceReference[]
  key_evidence_b: EvidenceReference[]
  tradeoff_summary: string
  recommended_fit: string
  disclaimer: string
  is_sparse: boolean
  is_unbalanced: boolean
  warnings: string[]
}

export interface TargetVersionItem {
  version: string
  count: number
}

export interface TargetProductItem {
  product: string
  count: number
  versions: TargetVersionItem[]
}

export interface AvailableTargetsResponse {
  products: TargetProductItem[]
  all_versions: TargetVersionItem[]
  unlabeled_product_count: number
  unlabeled_version_count: number
}

export async function fetchCompareTargets(
  reviews: ReviewInputItem[],
): Promise<AvailableTargetsResponse> {
  const res = await authenticatedFetch('/api/compare/targets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviews }),
  })
  return handleResponse<AvailableTargetsResponse>(res)
}

export async function compareVersions(payload: {
  reviews?: ReviewInputItem[]
  reviews_a?: ReviewInputItem[]
  reviews_b?: ReviewInputItem[]
  version_a: string
  version_b: string
  product?: string
  mode?: string
}): Promise<VersionCompareResponse> {
  const res = await authenticatedFetch('/api/compare/versions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse<VersionCompareResponse>(res)
}

export async function compareProducts(payload: {
  reviews?: ReviewInputItem[]
  reviews_a?: ReviewInputItem[]
  reviews_b?: ReviewInputItem[]
  product_a: string
  product_b: string
  focus_aspect?: string
  mode?: string
}): Promise<ProductCompareResponse> {
  const res = await authenticatedFetch('/api/compare/products', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse<ProductCompareResponse>(res)
}

export interface MLMetricsResponse {
  dataset: {
    name: string
    category: string
    source_url: string
    total_samples: number
    splits: { train: number; validation: number; test: number }
    class_distribution: Record<string, number>
    proxy_labeling_notice: string
  }
  model: {
    selected_model: string
    vectorizer: string
    random_seed: number
    training_timestamp: string
    vocabulary_size: number
  }
  test_results: {
    accuracy: number
    macro_f1: number
    weighted_f1: number
    average_latency_ms: number
  }
  per_class_metrics: Record<string, { precision: number; recall: number; f1_score: number; support: number }>
  confusion_matrix: {
    labels: string[]
    matrix: number[][]
  }
  model_comparison: Array<{
    model_name: string
    val_accuracy: number
    val_macro_f1: number
    test_accuracy: number
    test_macro_f1: number
    training_time_s: number
    latency_ms: number
    status: string
  }>
  neutral_class_analysis?: {
    finding: string
    total_neutral_test_samples: number
    correctly_classified_neutral: number
    confused_as_negative: number
    confused_as_positive: number
    neutral_f1: number
    negative_f1: number
    positive_f1: number
    mixed_signals_note: string
  }
  binary_benchmark?: {
    task: string
    dataset_samples: number
    splits: { train: number; validation: number; test: number }
    test_accuracy: number
    test_macro_f1: number
    classes: string[]
    note: string
  }
  limitations: string[]
}

export interface MLPredictResponse {
  text: string
  sentiment: 'negative' | 'neutral' | 'positive'
  polarity: 'neg' | 'neu' | 'pos'
  confidence: number
  probabilities: Record<string, number>
  model_name: string
  latency_ms: number
}

export async function fetchMLMetrics(): Promise<MLMetricsResponse> {
  const res = await authenticatedFetch('/api/ml/metrics', {
    method: 'GET',
  })
  return handleResponse<MLMetricsResponse>(res)
}

export async function predictSentiment(text: string): Promise<MLPredictResponse> {
  const res = await authenticatedFetch('/api/ml/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  return handleResponse<MLPredictResponse>(res)
}



