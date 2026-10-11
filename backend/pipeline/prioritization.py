"""Issue prioritization and Fix-This-First simulator pipeline.

PRD Section 5 Table 'insights' & Section 7.1 Fix-This-First Simulator:
- Transparent impact / priority scoring grounded in available evidence.
- Rating damage = negative mentions * |beta|.
- Priority score ranks safety (P0) first, then by rating damage.
- Simulator models projected rating = current rating + sum(f * (neg_reviews/total) * |beta|), capped at 5.0.
- Tracks unhappy reviews resolved (reviews rated <= 2 stars with single negative cause).
- Honest guardrails: low confidence if < 30 mentions; labeled if sentiment pseudo-ratings used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pipeline._common import load_yaml
from schemas import (
    AspectSimulateImpact,
    IssueCluster,
    IssuePriorityInsight,
    ReviewAnalysisResult,
    SimulateResponse,
)


def _load_routing_config() -> dict[str, dict[str, str]]:
    """Load team and priority mapping from routing.yaml."""
    try:
        data = load_yaml(None, "routing.yaml")
        return data.get("routing", {})
    except Exception:
        return {
            "battery": {"team": "Product/Engineering", "priority": "P1"},
            "performance": {"team": "Product/Engineering", "priority": "P1"},
            "quality": {"team": "Product/Engineering", "priority": "P1"},
            "delivery": {"team": "Logistics", "priority": "P1"},
            "safety": {"team": "QA/Compliance", "priority": "P0"},
        }


def prioritize_issues(
    reviews: list[ReviewAnalysisResult],
    clusters: list[IssueCluster] | None = None,
) -> list[IssuePriorityInsight]:
    """Calculate transparent impact and priority scoring across aspects.

    Follows PRD Section 5 Table 'insights' and Section 7.1.
    """
    if not reviews:
        return []

    routing_map = _load_routing_config()

    # Active non-duplicate reviews
    active_reviews = [r for r in reviews if r.status == "analyzed" and not r.is_duplicate]
    if not active_reviews:
        active_reviews = [r for r in reviews if r.status == "analyzed"]

    total_rev_count = max(1, len(active_reviews))

    # Collect all unique aspects mentioned
    all_aspects: set[str] = set()
    for r in active_reviews:
        for a in r.aspects:
            all_aspects.add(a)
    if clusters:
        for cl in clusters:
            all_aspects.add(cl.aspect)

    if not all_aspects:
        # Check if clauses have aspects
        for r in active_reviews:
            for c in r.clauses:
                if c.aspect:
                    all_aspects.add(c.aspect)

    if not all_aspects:
        return []

    # Map aspect -> ratings of reviews with negative aspect, and ratings without
    insights_out: list[IssuePriorityInsight] = []

    # Check if we are relying on pseudo-ratings
    any_pseudo_rating = any(r.rating is None and r.pseudo_rating is not None for r in active_reviews)

    # For single-cause unhappy: count reviews rated <= 2.0 whose only negative aspect is this
    single_cause_counts: dict[str, int] = {a: 0 for a in all_aspects}
    for r in active_reviews:
        eff_rating = r.rating if r.rating is not None else r.pseudo_rating
        if eff_rating is not None and eff_rating <= 2.0:
            # Find all negative aspects in this review
            neg_aspects = set()
            for c in r.clauses:
                if c.polarity == "neg" and c.aspect:
                    neg_aspects.add(c.aspect)
            if len(neg_aspects) == 1:
                single_asp = list(neg_aspects)[0]
                if single_asp in single_cause_counts:
                    single_cause_counts[single_asp] += 1

    tier_rank = {"none": 0, "medium": 1, "high": 2, "critical": 3}

    for aspect in sorted(list(all_aspects)):
        mentions = 0
        complaints = 0
        with_ratings: list[float] = []
        without_ratings: list[float] = []
        highest_safety = "none"

        for r in active_reviews:
            has_aspect = aspect in r.aspects or any(c.aspect == aspect for c in r.clauses)
            eff_rating = r.rating if r.rating is not None else r.pseudo_rating

            if has_aspect:
                mentions += 1
                # Check if negative mention
                is_neg = (
                    r.polarity == "neg"
                    or (eff_rating is not None and eff_rating <= 2.0)
                    or any(c.aspect == aspect and (c.polarity == "neg" or c.score < 0) for c in r.clauses)
                )
                if is_neg:
                    complaints += 1
                    if eff_rating is not None:
                        with_ratings.append(eff_rating)
                else:
                    if eff_rating is not None:
                        without_ratings.append(eff_rating)

                if tier_rank.get(r.safety.tier, 0) > tier_rank.get(highest_safety, 0):
                    highest_safety = r.safety.tier
            else:
                if eff_rating is not None:
                    without_ratings.append(eff_rating)

        pct_negative = round((complaints / mentions * 100.0), 1) if mentions > 0 else 0.0

        # Calculate beta (rating points lost per complaint)
        # Difference-in-means / OLS association (PRD Section 7.1)
        if with_ratings and without_ratings:
            mean_with = sum(with_ratings) / len(with_ratings)
            mean_without = sum(without_ratings) / len(without_ratings)
            raw_diff = mean_without - mean_with
            beta = round(max(0.1, raw_diff), 2)
            # Standard error estimation
            var_with = np.var(with_ratings) if len(with_ratings) > 1 else 0.5
            var_without = np.var(without_ratings) if len(without_ratings) > 1 else 0.5
            se = math.sqrt((var_with / len(with_ratings)) + (var_without / len(without_ratings)))
            ci_low = round(max(0.0, beta - 1.96 * se), 2)
            ci_high = round(beta + 1.96 * se, 2)
        elif with_ratings:
            # All mentions are negative
            beta = 0.8
            ci_low = 0.4
            ci_high = 1.2
        else:
            # Pure positive or neutral mentions
            beta = 0.3
            ci_low = 0.1
            ci_high = 0.5

        # Damage = complaint count * |beta|
        damage = round(complaints * abs(beta), 2)

        # Priority scoring: P0 boost for safety alerts, otherwise ranked by rating damage
        is_safety_hazard = highest_safety in {"critical", "high"}
        if is_safety_hazard:
            priority_score = round(100.0 + damage + complaints, 2)
        else:
            priority_score = damage

        confidence_label = "ok" if mentions >= 30 else "low"
        route = routing_map.get(aspect.lower(), {})
        team = "QA/Compliance" if is_safety_hazard else route.get("team", "Product/Engineering")
        priority_code = "P0" if is_safety_hazard else route.get("priority", "P1")

        insights_out.append(
            IssuePriorityInsight(
                aspect=aspect,
                polarity="neg",
                mentions=mentions,
                complaint_count=complaints,
                pct_negative=pct_negative,
                beta=beta,
                ci_low=ci_low,
                ci_high=ci_high,
                damage=damage,
                priority_score=priority_score,
                priority_rank=1,  # will be assigned below
                confidence_label=confidence_label,
                single_cause_unhappy=single_cause_counts.get(aspect, 0),
                team=team,
                priority_code=priority_code,
                safety_tier=highest_safety,
                is_sentiment_estimate=any_pseudo_rating,
            )
        )

    # Sort descending by priority_score, then complaints
    insights_out.sort(
        key=lambda ins: (
            tier_rank.get(ins.safety_tier, 0),
            ins.priority_score,
            ins.complaint_count,
        ),
        reverse=True,
    )

    # Assign priority_rank 1, 2, 3...
    for idx, ins in enumerate(insights_out):
        ins.priority_rank = idx + 1

    return insights_out


def simulate_fixes(
    reviews: list[ReviewAnalysisResult],
    insights: list[IssuePriorityInsight],
    fixes: dict[str, float],
) -> SimulateResponse:
    """PRD Section 7.1: Fix-This-First Simulator.

    Models projected average rating when a fraction f of each aspect's complaints are fixed:
    projected rating = current average + sum(f * (neg_reviews / total_reviews) * |beta|), capped at 5.0.
    """
    active_reviews = [r for r in reviews if r.status == "analyzed" and not r.is_duplicate]
    if not active_reviews:
        active_reviews = [r for r in reviews if r.status == "analyzed"]

    total_count = max(1, len(active_reviews))

    # Calculate baseline average rating
    ratings = [r.rating for r in active_reviews if r.rating is not None]
    is_sentiment = False
    if not ratings:
        ratings = [r.pseudo_rating for r in active_reviews if r.pseudo_rating is not None]
        is_sentiment = True

    current_avg = round(sum(ratings) / len(ratings), 2) if ratings else 3.0

    insight_by_aspect = {ins.aspect.lower(): ins for ins in insights}

    total_uplift = 0.0
    total_unhappy_resolved = 0
    per_aspect: list[AspectSimulateImpact] = []
    has_low_confidence = False
    notes: list[str] = []

    for aspect_raw, fraction_val in fixes.items():
        asp_key = aspect_raw.strip().lower()
        f = min(1.0, max(0.0, float(fraction_val)))
        if f <= 0.0:
            continue

        ins = insight_by_aspect.get(asp_key)
        if not ins:
            continue

        if ins.confidence_label == "low":
            has_low_confidence = True

        # Formula: f * (negative reviews / total reviews) * |beta|
        aspect_uplift = f * (ins.complaint_count / total_count) * abs(ins.beta)
        unhappy_res = int(round(f * ins.single_cause_unhappy))
        complaints_res = int(round(f * ins.complaint_count))

        total_uplift += aspect_uplift
        total_unhappy_resolved += unhappy_res

        per_aspect.append(
            AspectSimulateImpact(
                aspect=ins.aspect,
                fraction_fixed=f,
                uplift=round(aspect_uplift, 3),
                unhappy_resolved=unhappy_res,
                complaints_resolved=complaints_res,
            )
        )

    projected = min(5.0, round(current_avg + total_uplift, 2))
    delta = round(projected - current_avg, 2)

    if is_sentiment:
        notes.append("Simulation is based on sentiment-derived pseudo-ratings because observed ratings were absent.")
    if has_low_confidence:
        notes.append("One or more simulated aspects have fewer than 30 mentions; estimate reflects association rather than proven causality.")

    return SimulateResponse(
        current_rating=current_avg,
        projected_rating=projected,
        delta=delta,
        unhappy_reviews_resolved=total_unhappy_resolved,
        per_aspect=per_aspect,
        is_sentiment_estimate=is_sentiment,
        confidence_label="low" if has_low_confidence else "ok",
        notes=notes,
    )
