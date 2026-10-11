"""Pydantic schemas shared across API responses."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["lite", "full"]
    # True only when a database connection has actually been checked and
    # confirmed. No database check exists yet, so this is always False.
    db_connected: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    # Candidate column names for a column picker. Omitted when not applicable.
    columns: list[str] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class IngestMetadata(BaseModel):
    """Summary of a validated upload. Never contains review text."""

    source_name: str
    file_type: Literal["csv", "xlsx"]
    encoding: str | None  # CSV only
    original_columns: list[str]
    text_column: str
    rating_column: str | None
    date_column: str | None
    version_column: str | None
    rows_read: int  # non-blank data rows, before empty-text exclusion
    excluded_empty: int  # rows dropped because the review text was empty
    usable_rows: int
    warnings: list[str] = Field(default_factory=list)


class ReviewInputItem(BaseModel):
    id: str | int | None = None
    review_id: str | int | None = None
    text: str
    rating: float | int | str | None = None
    date: str | None = None
    version: str | None = None
    product: str | None = None


class AnalyzeRequest(BaseModel):
    reviews: list[ReviewInputItem]
    mode: str = "lite"


class ClauseAnalysisResult(BaseModel):
    clause_id: str | int
    text: str
    aspect: str | None = None
    aspects: list[str] = Field(default_factory=list)
    score: float
    polarity: Literal["pos", "neu", "neg"]
    label: str
    confidence: float
    sarcasm_suspect: bool = False
    rating_conflict: bool = False
    language: str = "en"
    start: int  # 0-based offset into normalized_text
    end: int  # 0-based exclusive end offset into normalized_text
    cluster_id: str | None = None


class SafetyMatchDetail(BaseModel):
    category: str
    tier: Literal["none", "medium", "high", "critical"]
    term: str
    phrase: str
    excerpt: str  # Redacted contextual excerpt
    suggested_action: str | None = None


class SafetyAnalysisResult(BaseModel):
    tier: Literal["none", "medium", "high", "critical"]
    is_alert: bool
    matches: list[SafetyMatchDetail] = Field(default_factory=list)
    suggested_action: str | None = None


class ReviewAnalysisResult(BaseModel):
    review_id: str
    original_text: str  # Redacted review text (PII email/phone masked)
    normalized_text: str  # Normalized text (offsets index into this)
    rating: float | None = None  # Observed rating (1.0 - 5.0) or null
    rating_source: str | None = None  # "given", "pseudo", or null
    pseudo_rating: float | None = None  # Estimated from sentiment when rating is null
    date: str | None = None
    version: str | None = None
    product: str | None = None
    language: str
    language_badge: str
    language_confidence: float = 0.0
    is_duplicate: bool = False
    duplicate_of: str | None = None  # review_id of the canonical first review
    truncated: bool = False
    sentiment_score: float
    polarity: Literal["pos", "neu", "neg"]
    label: str
    sarcasm_suspect: bool = False
    rating_conflict: bool = False
    aspects: list[str] = Field(default_factory=list)
    safety: SafetyAnalysisResult
    clauses: list[ClauseAnalysisResult] = Field(default_factory=list)
    status: Literal["analyzed", "empty", "error"] = "analyzed"


class AnalysisSummary(BaseModel):
    total_reviews: int
    analyzed_reviews: int
    unique_reviews: int
    duplicate_count: int = 0
    total_clauses: int
    languages: dict[str, int] = Field(default_factory=dict)
    sentiment_breakdown: dict[str, int] = Field(default_factory=dict)
    safety_breakdown: dict[str, int] = Field(default_factory=dict)
    sarcasm_count: int = 0
    rating_conflict_count: int = 0
    aspect_mentions: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AnalysisMetadata(BaseModel):
    mode: Literal["lite", "full"]
    requested_mode: str | None = None
    processing_time_ms: float
    summary: AnalysisSummary


class IssueClusterClauseEvidence(BaseModel):
    review_id: str
    clause_id: str | int
    text: str
    score: float
    polarity: Literal["pos", "neu", "neg"]
    safety_tier: str = "none"


class IssueCluster(BaseModel):
    """PRD Section 5 Table 'clusters' with traceability evidence."""
    cluster_id: str
    aspect: str
    polarity: Literal["pos", "neu", "neg"]
    label: str
    example_clause_id: str | int
    example_quote: str
    size: int
    review_count: int
    clause_ids: list[str | int] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)
    evidence: list[IssueClusterClauseEvidence] = Field(default_factory=list)
    safety_tier: str = "none"
    is_safety_alert: bool = False
    average_rating: float | None = None
    average_sentiment: float = 0.0
    team: str | None = None
    priority: str | None = None


class IssuePriorityInsight(BaseModel):
    """PRD Section 5 Table 'insights' with transparent priority scoring."""
    aspect: str
    polarity: Literal["pos", "neu", "neg"] = "neg"
    mentions: int
    complaint_count: int
    pct_negative: float
    beta: float  # rating points lost per negative complaint
    ci_low: float
    ci_high: float
    damage: float  # rating damage = complaint_count * |beta|
    priority_score: float
    priority_rank: int
    confidence_label: Literal["ok", "low"]
    single_cause_unhappy: int
    team: str | None = None
    priority_code: str | None = None
    safety_tier: str = "none"
    is_sentiment_estimate: bool = False


class IssuesResponse(BaseModel):
    clusters: list[IssueCluster] = Field(default_factory=list)
    insights: list[IssuePriorityInsight] = Field(default_factory=list)
    total_clusters: int = 0
    top_aspects: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AspectSimulateImpact(BaseModel):
    aspect: str
    fraction_fixed: float
    uplift: float
    unhappy_resolved: int
    complaints_resolved: int


class SimulateRequest(BaseModel):
    fixes: dict[str, float] = Field(default_factory=dict)
    reviews: list[ReviewInputItem] = Field(default_factory=list)


class SimulateResponse(BaseModel):
    current_rating: float
    projected_rating: float
    delta: float
    unhappy_reviews_resolved: int
    per_aspect: list[AspectSimulateImpact] = Field(default_factory=list)
    is_sentiment_estimate: bool = False
    confidence_label: Literal["ok", "low"] = "ok"
    notes: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    reviews: list[ReviewAnalysisResult]
    meta: AnalysisMetadata
    clusters: list[IssueCluster] = Field(default_factory=list)
    insights: list[IssuePriorityInsight] = Field(default_factory=list)

    def to_reviews_frame(self):
        """Export to PRD Section 5 Table 'reviews' DataFrame."""
        import pandas as pd

        rows = []
        for r in self.reviews:
            rows.append({
                "review_id": r.review_id,
                "text": r.original_text,
                "rating": r.rating,
                "date": r.date,
                "version": r.version,
                "lang": r.language,
                "is_duplicate": r.is_duplicate,
                "rating_source": r.rating_source,
            })
        return pd.DataFrame(rows)

    def to_clauses_frame(self):
        """Export to PRD Section 5 Table 'clauses' DataFrame."""
        import pandas as pd

        rows = []
        for r in self.reviews:
            for c in r.clauses:
                rows.append({
                    "review_id": r.review_id,
                    "clause_id": c.clause_id,
                    "clause_text": c.text,
                    "aspect": c.aspect,
                    "score": c.score,
                    "polarity": c.polarity,
                    "confidence": c.confidence,
                    "sarcasm_suspect": c.sarcasm_suspect,
                    "rating_conflict": c.rating_conflict,
                    "safety_tier": r.safety.tier,
                    "cluster_id": c.cluster_id,
                })
        return pd.DataFrame(rows)

    def to_clusters_frame(self):
        """Export to PRD Section 5 Table 'clusters' DataFrame."""
        import pandas as pd

        rows = []
        for cl in self.clusters:
            rows.append({
                "cluster_id": cl.cluster_id,
                "aspect": cl.aspect,
                "polarity": cl.polarity,
                "label": cl.label,
                "example_clause_id": cl.example_clause_id,
                "size": cl.size,
            })
        return pd.DataFrame(rows)

    def to_insights_frame(self):
        """Export to PRD Section 5 Table 'insights' DataFrame."""
        import pandas as pd

        rows = []
        for ins in self.insights:
            rows.append({
                "aspect": ins.aspect,
                "polarity": ins.polarity,
                "mentions": ins.mentions,
                "pct_negative": ins.pct_negative,
                "beta": ins.beta,
                "ci_low": ins.ci_low,
                "ci_high": ins.ci_high,
                "damage": ins.damage,
                "priority_rank": ins.priority_rank,
                "confidence_label": ins.confidence_label,
                "single_cause_unhappy": ins.single_cause_unhappy,
            })
        return pd.DataFrame(rows)


class EmergingIssueEvidence(BaseModel):
    review_id: str
    clause_id: str | int
    text: str
    rating: float | None = None
    date: str | None = None
    safety_tier: str = "none"


class EmergingIssueFinding(BaseModel):
    issue_id: str
    label: str
    category: str
    aspect: str
    polarity: Literal["neg", "neu", "pos"]
    first_observed_date: str | None = None
    recent_count: int
    baseline_count: int
    recent_share: float
    baseline_share: float
    change_count: int
    change_share: float
    velocity_ratio: float | None = None
    severity: Literal["critical", "high", "medium", "low"]
    priority: int
    status: Literal["new", "accelerating", "steady", "insufficient_data"]
    confidence: Literal["high", "medium", "low"]
    limitations: list[str] = Field(default_factory=list)
    is_safety_escalated: bool = False
    evidence: list[EmergingIssueEvidence] = Field(default_factory=list)


class TimeWindowInfo(BaseModel):
    recent_start: str | None = None
    recent_end: str | None = None
    baseline_start: str | None = None
    baseline_end: str | None = None
    total_recent_reviews: int = 0
    total_baseline_reviews: int = 0
    undated_reviews: int = 0
    has_sufficient_timeline: bool = False


class EmergingIssuesRequest(BaseModel):
    reviews: list[ReviewInputItem]
    mode: str = "lite"
    recent_days: int = 14
    baseline_days: int = 30
    split_date: str | None = None
    min_recent_count: int = 2


class EmergingIssuesResponse(BaseModel):
    total_emerging_issues: int
    new_issues_count: int
    accelerating_issues_count: int
    safety_escalations_count: int
    insufficient_data_count: int
    time_windows: TimeWindowInfo
    findings: list[EmergingIssueFinding]
    warnings: list[str] = Field(default_factory=list)


class VisionHealthResponse(BaseModel):
    status: Literal["ready", "unavailable", "model_not_installed", "vision_unsupported"]
    provider: str
    base_url: str
    model: str
    vision_supported: bool
    message: str
    available_models: list[str] = Field(default_factory=list)


class ImageDefectAnalysis(BaseModel):
    image_id: str
    filename: str
    mime_type: str
    file_size_bytes: int
    visual_description: str
    potential_defect_category: str = "none_detected"
    visual_observations: list[str] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high", "critical"] = "none"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    relevant_aspect: str | None = None
    relevant_issue_cluster: str | None = None
    limitations: list[str] = Field(default_factory=list)
    suggested_manual_checks: list[str] = Field(default_factory=list)
    is_defect_observed: bool = False
    thumbnail_data_url: str | None = None
    raw_model_response: str | None = None


class EvidenceReference(BaseModel):
    source_type: Literal["review", "image", "temporal"]
    reference_id: str
    excerpt_or_observation: str
    sentiment_score: float | None = None
    metadata: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class HypothesisItem(BaseModel):
    hypothesis_id: str
    rank: int
    proposed_cause: str
    category: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_level: Literal["high", "medium", "low"]
    explanation: str
    supporting_evidence: list[EvidenceReference] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    falsification_criteria: str
    is_hypothesis_only: bool = True


class RootCauseAnalysisResult(BaseModel):
    cluster_id: str
    issue_label: str
    aspect: str
    status: Literal["hypotheses_generated", "insufficient_evidence"]
    has_sufficient_evidence: bool
    summary: str
    hypotheses: list[HypothesisItem] = Field(default_factory=list)
    insufficient_evidence_reason: str | None = None
    total_supporting_reviews: int
    total_contradicting_reviews: int
    associated_images_count: int
    temporal_pattern_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "HYPOTHESIS ONLY: Results represent plausible failure mechanisms deduced from "
        "customer review symptoms and available photo observations. They are not established "
        "root causes and require engineering bench verification."
    )


class RootCauseRequest(BaseModel):
    cluster_id: str | None = None
    reviews: list[ReviewInputItem] = Field(default_factory=list)
    image_findings: list[ImageDefectAnalysis] = Field(default_factory=list)
    mode: str = "lite"


class RootCauseBatchResponse(BaseModel):
    results: list[RootCauseAnalysisResult]
    total_clusters_analyzed: int
    clusters_with_hypotheses: int
    clusters_with_insufficient_evidence: int
    warnings: list[str] = Field(default_factory=list)


# --- Comparison Schemas (Feature 1 & Feature 2) -----------------------------

class SentimentProportions(BaseModel):
    pos_count: int
    neu_count: int
    neg_count: int
    pos_ratio: float
    neu_ratio: float
    neg_ratio: float


class AspectComparisonItem(BaseModel):
    aspect: str
    aspect_label: str
    count_a: int
    complaint_count_a: int
    negative_rate_a: float
    avg_sentiment_a: float
    count_b: int
    complaint_count_b: int
    negative_rate_b: float
    avg_sentiment_b: float
    complaint_diff: int
    negative_rate_diff: float
    status: Literal["improved", "worsened", "newly_observed", "not_observed", "resolved", "unchanged"]
    sample_note: str | None = None


class IssueClusterComparisonItem(BaseModel):
    cluster_label: str
    aspect: str
    count_a: int
    share_a: float
    priority_a: str | None = None
    safety_tier_a: str = "none"
    count_b: int
    share_b: float
    priority_b: str | None = None
    safety_tier_b: str = "none"
    status: Literal["improved", "worsened", "newly_observed", "not_observed", "resolved", "unchanged"]
    sample_limitation: str | None = None
    evidence_a: list[EvidenceReference] = Field(default_factory=list)
    evidence_b: list[EvidenceReference] = Field(default_factory=list)


class VersionMetricsSummary(BaseModel):
    version: str
    review_count: int
    rating_count: int
    average_rating: float | None = None
    sentiment_proportions: SentimentProportions
    average_sentiment: float
    safety_alert_count: int
    safety_tiers: list[str] = Field(default_factory=list)


class VersionCompareResponse(BaseModel):
    product: str | None = None
    version_a: VersionMetricsSummary
    version_b: VersionMetricsSummary
    aspect_comparisons: list[AspectComparisonItem]
    cluster_comparisons: list[IssueClusterComparisonItem]
    safety_changes: list[str] = Field(default_factory=list)
    summary_of_changes: str
    disclaimer: str
    is_sparse_sample: bool = False
    warnings: list[str] = Field(default_factory=list)


class VersionCompareRequest(BaseModel):
    reviews: list[ReviewInputItem] | None = None
    reviews_a: list[ReviewInputItem] | None = None
    reviews_b: list[ReviewInputItem] | None = None
    product: str | None = None
    version_a: str
    version_b: str
    mode: str = "lite"


class ProductMetricsSummary(BaseModel):
    product: str
    review_count: int
    rating_count: int
    average_rating: float | None = None
    sentiment_proportions: SentimentProportions
    average_sentiment: float
    safety_alert_count: int
    safety_tiers: list[str] = Field(default_factory=list)


class ProductAspectStrengthItem(BaseModel):
    aspect: str
    aspect_label: str
    leader: Literal["product_a", "product_b", "tie"]
    sentiment_a: float
    negative_rate_a: float
    complaints_a: int
    sentiment_b: float
    negative_rate_b: float
    complaints_b: int
    strength_summary: str


class ProductCompareResponse(BaseModel):
    product_a: ProductMetricsSummary
    product_b: ProductMetricsSummary
    focus_aspect: str | None = None
    aspect_strengths: list[ProductAspectStrengthItem]
    common_complaint_themes: list[str]
    unique_issues_a: list[str]
    unique_issues_b: list[str]
    prominent_safety_alerts: list[dict[str, Any]] = Field(default_factory=list)
    key_evidence_a: list[EvidenceReference] = Field(default_factory=list)
    key_evidence_b: list[EvidenceReference] = Field(default_factory=list)
    tradeoff_summary: str
    recommended_fit: str
    disclaimer: str
    is_sparse: bool = False
    is_unbalanced: bool = False
    warnings: list[str] = Field(default_factory=list)


class ProductCompareRequest(BaseModel):
    reviews: list[ReviewInputItem] | None = None
    reviews_a: list[ReviewInputItem] | None = None
    reviews_b: list[ReviewInputItem] | None = None
    product_a: str
    product_b: str
    focus_aspect: str | None = None
    mode: str = "lite"


class TargetVersionItem(BaseModel):
    version: str
    count: int


class TargetProductItem(BaseModel):
    product: str
    count: int
    versions: list[TargetVersionItem] = Field(default_factory=list)


class AvailableTargetsResponse(BaseModel):
    products: list[TargetProductItem]
    all_versions: list[TargetVersionItem]
    unlabeled_product_count: int
    unlabeled_version_count: int


class AvailableTargetsRequest(BaseModel):
    reviews: list[ReviewInputItem]


class MLPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Customer review text to analyze")


class MLPredictResponse(BaseModel):
    text: str
    sentiment: str
    polarity: str
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_name: str
    latency_ms: float


class MLMetricsResponse(BaseModel):
    dataset: dict[str, Any]
    model: dict[str, Any]
    test_results: dict[str, Any]
    per_class_metrics: dict[str, Any]
    confusion_matrix: dict[str, Any]
    model_comparison: list[dict[str, Any]] = Field(default_factory=list)
    neutral_class_analysis: dict[str, Any] | None = None
    binary_benchmark: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)




