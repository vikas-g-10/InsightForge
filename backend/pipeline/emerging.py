"""Emerging-issue detection pipeline for InsightForge.

Analyzes review clusters across configurable temporal windows:
- Compares recent vs. baseline time windows.
- Detects new issue clusters (0 in baseline, >= min_recent_count in recent).
- Detects accelerating issues (statistically meaningful increase in count or frequency share).
- Accounts for sample size: avoids labeling single complaints as confirmed trends.
- Prioritizes safety hazards independently of frequency or sentiment (safety escalation).
- Identifies sparse data and missing timestamps honestly without fabricating historical trends.
"""

from __future__ import annotations

import datetime
from typing import Any, Sequence

import pandas as pd

from pipeline.clustering import cluster_clauses
from schemas import (
    EmergingIssueEvidence,
    EmergingIssueFinding,
    EmergingIssuesResponse,
    IssueCluster,
    ReviewAnalysisResult,
    TimeWindowInfo,
)


def parse_date(val: Any) -> datetime.date | None:
    """Robustly parse date from string, datetime, or pandas timestamp."""
    if val is None:
        return None
    if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
        return val
    if isinstance(val, datetime.datetime):
        return val.date()

    raw = str(val).strip()
    if not raw or raw.lower() in ("none", "null", "nan", "nat", ""):
        return None

    # Handle ISO string with T
    date_part = raw.split("T")[0].split(" ")[0].strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(date_part, fmt).date()
        except Exception:
            continue

    try:
        dt = pd.to_datetime(raw, errors="coerce")
        if pd.notna(dt):
            return dt.date()
    except Exception:
        pass
    return None


def detect_emerging_issues(
    reviews: Sequence[ReviewAnalysisResult],
    clusters: list[IssueCluster] | None = None,
    *,
    recent_days: int = 14,
    baseline_days: int = 30,
    split_date: str | None = None,
    min_recent_count: int = 2,
    min_baseline_reviews: int = 5,
    distance_threshold: float = 0.85,
) -> EmergingIssuesResponse:
    """Detect emerging, accelerating, new, and safety-escalated issues across time windows.

    Parameters
    ----------
    reviews : Sequence[ReviewAnalysisResult]
        Analyzed review objects.
    clusters : list[IssueCluster] | None
        Pre-computed clusters. If None, clusters will be computed automatically.
    recent_days : int
        Duration (in days) of the recent window when split_date is not provided.
    baseline_days : int
        Duration (in days) of the baseline historical window.
    split_date : str | None
        Optional explicit date splitting the baseline and recent windows (YYYY-MM-DD).
    min_recent_count : int
        Minimum recent occurrences required to confirm a trend (avoids 1-off complaints as trends).
    min_baseline_reviews : int
        Minimum total reviews in the baseline window before flagging low-confidence baseline.
    distance_threshold : float
        Clustering distance threshold for grouping clauses into issue topics (default: 0.85).
    """
    if clusters is None:
        clusters = cluster_clauses(reviews, distance_threshold=distance_threshold)

    review_by_id: dict[str, ReviewAnalysisResult] = {r.review_id: r for r in reviews}

    # Map review dates
    rev_dates: dict[str, datetime.date | None] = {
        r.review_id: parse_date(r.date) for r in reviews
    }
    valid_dates: list[datetime.date] = [d for d in rev_dates.values() if d is not None]
    undated_count = sum(1 for d in rev_dates.values() if d is None)

    warnings: list[str] = []

    # Check temporal sufficiency
    distinct_dates = set(valid_dates)
    has_sufficient_timeline = len(distinct_dates) >= 2

    recent_start_d: datetime.date | None = None
    recent_end_d: datetime.date | None = None
    baseline_start_d: datetime.date | None = None
    baseline_end_d: datetime.date | None = None

    if split_date:
        parsed_split = parse_date(split_date)
        if parsed_split:
            recent_start_d = parsed_split
            max_d = max(valid_dates) if valid_dates else parsed_split
            recent_end_d = max(parsed_split, max_d)
            baseline_end_d = parsed_split
            baseline_start_d = parsed_split - datetime.timedelta(days=baseline_days)
            has_sufficient_timeline = True
        else:
            warnings.append(f"Invalid split_date '{split_date}'. Falling back to automatic date windows.")

    if not recent_start_d and has_sufficient_timeline:
        max_d = max(valid_dates)
        min_d = min(valid_dates)
        total_span_days = (max_d - min_d).days

        if total_span_days < 2:
            has_sufficient_timeline = False
            warnings.append("Review date span is less than 2 days; insufficient timeline for historical baseline.")
        else:
            # Bound recent_days to not swallow the entire dataset
            eff_recent_days = min(recent_days, max(1, total_span_days // 2))
            recent_start_d = max_d - datetime.timedelta(days=eff_recent_days)
            recent_end_d = max_d
            baseline_end_d = recent_start_d
            baseline_start_d = max(min_d, recent_start_d - datetime.timedelta(days=baseline_days))

    # Partition reviews into windows
    recent_rev_ids: set[str] = set()
    baseline_rev_ids: set[str] = set()

    if has_sufficient_timeline and recent_start_d and recent_end_d and baseline_start_d and baseline_end_d:
        for r_id, d in rev_dates.items():
            if d is None:
                continue
            if recent_start_d <= d <= recent_end_d:
                recent_rev_ids.add(r_id)
            elif baseline_start_d <= d < baseline_end_d:
                baseline_rev_ids.add(r_id)
    else:
        warnings.append("Timestamps are missing or insufficient to establish historical baseline vs. recent windows.")

    total_recent = len(recent_rev_ids)
    total_baseline = len(baseline_rev_ids)

    time_windows = TimeWindowInfo(
        recent_start=recent_start_d.isoformat() if recent_start_d else None,
        recent_end=recent_end_d.isoformat() if recent_end_d else None,
        baseline_start=baseline_start_d.isoformat() if baseline_start_d else None,
        baseline_end=baseline_end_d.isoformat() if baseline_end_d else None,
        total_recent_reviews=total_recent,
        total_baseline_reviews=total_baseline,
        undated_reviews=undated_count,
        has_sufficient_timeline=has_sufficient_timeline,
    )

    findings: list[EmergingIssueFinding] = []

    for cluster in clusters:
        cluster_rev_ids = set(cluster.review_ids)
        # Determine counts in each window
        if has_sufficient_timeline:
            c_recent_revs = cluster_rev_ids.intersection(recent_rev_ids)
            c_baseline_revs = cluster_rev_ids.intersection(baseline_rev_ids)
            recent_count = len(c_recent_revs)
            baseline_count = len(c_baseline_revs)
        else:
            c_recent_revs = set()
            c_baseline_revs = set()
            recent_count = len(cluster_rev_ids)
            baseline_count = 0

        # Calculate shares
        recent_share = round(recent_count / max(1, total_recent), 4) if total_recent > 0 else 0.0
        baseline_share = round(baseline_count / max(1, total_baseline), 4) if total_baseline > 0 else 0.0
        change_count = recent_count - baseline_count
        change_share = round(recent_share - baseline_share, 4)

        if baseline_share > 0:
            velocity_ratio = round(recent_share / baseline_share, 2)
        else:
            velocity_ratio = None

        # First observed date
        cluster_dates = [rev_dates.get(r_id) for r_id in cluster_rev_ids if rev_dates.get(r_id) is not None]
        first_observed_d = min(cluster_dates) if cluster_dates else None
        first_observed_str = first_observed_d.isoformat() if first_observed_d else None

        limitations: list[str] = []
        is_safety_escalated = cluster.safety_tier in ("critical", "high") or cluster.is_safety_alert

        # Safety escalation takes precedence independently of frequency or sentiment
        if is_safety_escalated:
            severity = "critical" if cluster.safety_tier == "critical" else "high"
            priority = 1
            if not has_sufficient_timeline:
                status = "insufficient_data"
                confidence = "medium"
                limitations.append("Timestamp trend unavailable; safety issue escalated immediately based on severity.")
            elif baseline_count == 0 and recent_count >= min_recent_count:
                status = "new"
                confidence = "high"
            elif baseline_count > 0 and recent_count >= min_recent_count and recent_share >= baseline_share:
                status = "accelerating"
                confidence = "high"
            elif recent_count == 1:
                status = "new" if baseline_count == 0 else "steady"
                confidence = "medium"
                limitations.append("Single report recorded; escalated immediately due to safety hazard.")
            else:
                status = "steady"
                confidence = "medium"
        elif not has_sufficient_timeline:
            # Sparse or missing timestamps: never fabricate trend
            status = "insufficient_data"
            severity = "medium" if cluster.polarity == "neg" else "low"
            priority = 4
            confidence = "low"
            limitations.append("Timestamps missing or inadequate to establish temporal velocity.")
        else:
            # Baseline & recent timelines exist
            if total_baseline < min_baseline_reviews:
                limitations.append(f"Baseline window has small sample size ({total_baseline} reviews); trend confidence is limited.")

            if recent_count == 0 and baseline_count > 0:
                # Issue did not recur in recent window
                status = "steady"
                severity = "low"
                priority = 5
                confidence = "medium"
                limitations.append("No occurrences detected in recent window (potential recovery).")
            elif recent_count == 1 and baseline_count == 0:
                # Single complaint: requirement states explicitly: do not label a single complaint as a confirmed trend
                status = "insufficient_data"
                severity = "low"
                priority = 4
                confidence = "low"
                limitations.append("Single complaint in recent window; insufficient sample size to confirm an emerging trend.")
            elif baseline_count == 0 and recent_count >= min_recent_count:
                # Brand new issue cluster appearing in recent window
                status = "new"
                severity = "high" if recent_count >= 4 or (cluster.polarity == "neg" and cluster.average_sentiment < -0.4) else "medium"
                priority = 2 if severity == "high" else 3
                confidence = "high" if recent_count >= 3 and total_baseline >= min_baseline_reviews else "medium"
                if total_baseline < min_baseline_reviews:
                    confidence = "low"
            elif baseline_count > 0 and recent_count >= min_recent_count:
                # Check for significant velocity/share growth
                is_growing = (recent_share >= 1.4 * baseline_share and change_count >= 1) or (recent_count >= 2 * baseline_count and change_count >= 2)
                if is_growing:
                    status = "accelerating"
                    severity = "high" if (velocity_ratio and velocity_ratio >= 2.0) or recent_count >= 5 else "medium"
                    priority = 2 if severity == "high" else 3
                    confidence = "high" if recent_count >= 3 and total_recent >= 5 and total_baseline >= min_baseline_reviews else "medium"
                    if total_baseline < min_baseline_reviews:
                        confidence = "low"
                else:
                    status = "steady"
                    severity = "low"
                    priority = 4
                    confidence = "medium"
            else:
                status = "steady"
                severity = "low"
                priority = 5
                confidence = "low"
                limitations.append("Low overall occurrence count in both baseline and recent windows.")

        # Representative evidence
        evidence_items: list[EmergingIssueEvidence] = []
        for ev in cluster.evidence[:5]:
            rev = review_by_id.get(ev.review_id)
            evidence_items.append(
                EmergingIssueEvidence(
                    review_id=ev.review_id,
                    clause_id=ev.clause_id,
                    text=ev.text,
                    rating=rev.rating if rev else None,
                    date=rev.date if rev else None,
                    safety_tier=ev.safety_tier,
                )
            )

        findings.append(
            EmergingIssueFinding(
                issue_id=cluster.cluster_id,
                label=cluster.label,
                category=cluster.aspect,
                aspect=cluster.aspect,
                polarity=cluster.polarity,
                first_observed_date=first_observed_str,
                recent_count=recent_count,
                baseline_count=baseline_count,
                recent_share=recent_share,
                baseline_share=baseline_share,
                change_count=change_count,
                change_share=change_share,
                velocity_ratio=velocity_ratio,
                severity=severity,
                priority=priority,
                status=status,
                confidence=confidence,
                limitations=limitations,
                is_safety_escalated=is_safety_escalated,
                evidence=evidence_items,
            )
        )

    # Sort findings: Safety escalations first, then priority ascending, then recent_count descending
    status_order = {"new": 0, "accelerating": 1, "insufficient_data": 2, "steady": 3}
    findings.sort(
        key=lambda f: (
            0 if f.is_safety_escalated else 1,
            f.priority,
            status_order.get(f.status, 9),
            -f.recent_count,
        )
    )

    new_count = sum(1 for f in findings if f.status == "new")
    accel_count = sum(1 for f in findings if f.status == "accelerating")
    safety_count = sum(1 for f in findings if f.is_safety_escalated)
    insufficient_count = sum(1 for f in findings if f.status == "insufficient_data")
    total_emerging = sum(1 for f in findings if f.status in ("new", "accelerating") or f.is_safety_escalated)

    return EmergingIssuesResponse(
        total_emerging_issues=total_emerging,
        new_issues_count=new_count,
        accelerating_issues_count=accel_count,
        safety_escalations_count=safety_count,
        insufficient_data_count=insufficient_count,
        time_windows=time_windows,
        findings=findings,
        warnings=warnings,
    )
