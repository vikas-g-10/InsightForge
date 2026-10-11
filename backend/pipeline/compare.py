"""Product and Version Comparison Engine (Feature 1 & Feature 2).

Provides dedicated comparison pipelines:
1. Version Comparison:
   - Compares two versions of the same product.
   - Validates version existence, guards against mixed products.
   - Computes rating, sentiment proportions, aspect-level complaint deltas.
   - Distinguishes "not observed" from "resolved" in sparse samples.
   - Retains traceable evidence references and independent safety alerts.
   - Generates observational, non-causal summaries.

2. Product Comparison:
   - Compares two distinct products using explicit product identifiers.
   - Evaluates sample balance and flags sparse/unbalanced datasets.
   - Computes aspect-by-aspect strengths and weaknesses.
   - Identifies common complaint themes vs product-specific issues.
   - Provides balanced trade-off summaries and optional tailored focus aspects.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from ingest import IngestError
from pipeline.analyze import analyze
from pipeline.clustering import cluster_clauses
from schemas import (
    AspectComparisonItem,
    AvailableTargetsResponse,
    EvidenceReference,
    IssueClusterComparisonItem,
    ProductAspectStrengthItem,
    ProductCompareResponse,
    ProductMetricsSummary,
    ReviewAnalysisResult,
    ReviewInputItem,
    SentimentProportions,
    TargetProductItem,
    TargetVersionItem,
    VersionCompareResponse,
    VersionMetricsSummary,
)

logger = logging.getLogger(__name__)

ASPECT_LABELS = {
    "camera": "Camera & Optics",
    "battery": "Battery & Charging",
    "performance": "Performance & Speed",
    "quality": "Build Quality & Durability",
    "design": "Design & Ergonomics",
    "sound": "Audio & Acoustics",
    "delivery": "Shipping & Delivery",
    "packaging": "Packaging & Box Integrity",
    "customer_support": "Customer Support",
    "pricing": "Price & Value",
    "stability": "Software & System Stability",
    "ui_ux": "User Interface & Experience",
    "payments": "Billing & Payments",
    "general": "General Experience",
}


def _get_aspect_label(aspect: str) -> str:
    return ASPECT_LABELS.get(aspect, aspect.replace("_", " ").title())


def get_available_targets(
    reviews: list[ReviewInputItem] | list[ReviewAnalysisResult] | list[dict[str, Any]],
) -> AvailableTargetsResponse:
    """Extract distinct products and versions available in a review batch."""
    product_version_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_totals: Counter[str] = Counter()
    version_totals: Counter[str] = Counter()
    unlabeled_product_count = 0
    unlabeled_version_count = 0

    for r in reviews:
        prod = (
            getattr(r, "product", None)
            or (r.get("product") if isinstance(r, dict) else None)
        )
        ver = (
            getattr(r, "version", None)
            or (r.get("version") if isinstance(r, dict) else None)
        )

        p_clean = str(prod).strip() if prod and str(prod).strip() else None
        v_clean = str(ver).strip() if ver and str(ver).strip() else None

        if p_clean:
            product_totals[p_clean] += 1
            if v_clean:
                product_version_counts[p_clean][v_clean] += 1
        else:
            unlabeled_product_count += 1

        if v_clean:
            version_totals[v_clean] += 1
        else:
            unlabeled_version_count += 1

    products_out: list[TargetProductItem] = []
    for prod_name, total in sorted(product_totals.items(), key=lambda x: -x[1]):
        v_items = [
            TargetVersionItem(version=v, count=cnt)
            for v, cnt in sorted(product_version_counts[prod_name].items(), key=lambda x: -x[1])
        ]
        products_out.append(TargetProductItem(product=prod_name, count=total, versions=v_items))

    all_versions_out = [
        TargetVersionItem(version=v, count=cnt)
        for v, cnt in sorted(version_totals.items(), key=lambda x: -x[1])
    ]

    return AvailableTargetsResponse(
        products=products_out,
        all_versions=all_versions_out,
        unlabeled_product_count=unlabeled_product_count,
        unlabeled_version_count=unlabeled_version_count,
    )


def _compute_sentiment_proportions(reviews: list[ReviewAnalysisResult]) -> SentimentProportions:
    pos = sum(1 for r in reviews if r.polarity == "pos")
    neu = sum(1 for r in reviews if r.polarity == "neu")
    neg = sum(1 for r in reviews if r.polarity == "neg")
    total = len(reviews) or 1
    return SentimentProportions(
        pos_count=pos,
        neu_count=neu,
        neg_count=neg,
        pos_ratio=round(pos / total, 4),
        neu_ratio=round(neu / total, 4),
        neg_ratio=round(neg / total, 4),
    )


def _compute_version_metrics(version: str, reviews: list[ReviewAnalysisResult]) -> VersionMetricsSummary:
    rated = [r.rating for r in reviews if r.rating is not None]
    avg_rating = round(sum(rated) / len(rated), 2) if rated else None
    avg_sent = round(sum(r.sentiment_score for r in reviews) / len(reviews), 4) if reviews else 0.0

    safety_alerts = [r for r in reviews if r.safety.is_alert]
    safety_tiers = sorted(list({r.safety.tier for r in safety_alerts if r.safety.tier != "none"}))

    return VersionMetricsSummary(
        version=version,
        review_count=len(reviews),
        rating_count=len(rated),
        average_rating=avg_rating,
        sentiment_proportions=_compute_sentiment_proportions(reviews),
        average_sentiment=avg_sent,
        safety_alert_count=len(safety_alerts),
        safety_tiers=safety_tiers,
    )


def _compute_product_metrics(product: str, reviews: list[ReviewAnalysisResult]) -> ProductMetricsSummary:
    rated = [r.rating for r in reviews if r.rating is not None]
    avg_rating = round(sum(rated) / len(rated), 2) if rated else None
    avg_sent = round(sum(r.sentiment_score for r in reviews) / len(reviews), 4) if reviews else 0.0

    safety_alerts = [r for r in reviews if r.safety.is_alert]
    safety_tiers = sorted(list({r.safety.tier for r in safety_alerts if r.safety.tier != "none"}))

    return ProductMetricsSummary(
        product=product,
        review_count=len(reviews),
        rating_count=len(rated),
        average_rating=avg_rating,
        sentiment_proportions=_compute_sentiment_proportions(reviews),
        average_sentiment=avg_sent,
        safety_alert_count=len(safety_alerts),
        safety_tiers=safety_tiers,
    )


def compare_versions(
    all_reviews: list[ReviewAnalysisResult] | list[ReviewInputItem],
    version_a: str,
    version_b: str,
    *,
    product: str | None = None,
    mode: str = "lite",
) -> VersionCompareResponse:
    """Compare two versions of the same product with grounded metrics and evidence."""
    v_a = version_a.strip()
    v_b = version_b.strip()

    if not v_a or not v_b:
        raise IngestError("missing_version", "Both Version A and Version B must be specified.", 400)

    if v_a.lower() == v_b.lower():
        raise IngestError("same_version", "Cannot compare a version with itself. Select two different versions.", 400)

    # 1. Ensure reviews are analyzed
    if not all_reviews:
        raise IngestError("empty_batch", "The review dataset is empty.", 400)

    if isinstance(all_reviews[0], ReviewInputItem):
        analysis_res = analyze(all_reviews, mode=mode)
        analyzed_reviews = analysis_res.reviews
    else:
        analyzed_reviews = all_reviews  # type: ignore[assignment]

    # 2. Filter reviews for Version A and Version B
    reviews_a = [r for r in analyzed_reviews if (r.version or "").strip().lower() == v_a.lower()]
    reviews_b = [r for r in analyzed_reviews if (r.version or "").strip().lower() == v_b.lower()]

    if not reviews_a:
        raise IngestError(
            "version_not_found",
            f"No reviews found for Version A ('{v_a}'). Please verify the version value in the dataset.",
            404,
        )

    if not reviews_b:
        raise IngestError(
            "version_not_found",
            f"No reviews found for Version B ('{v_b}'). Please verify the version value in the dataset.",
            404,
        )

    # 3. Product validation: require both versions to belong to the same product
    products_a = {r.product.strip() for r in reviews_a if r.product and r.product.strip()}
    products_b = {r.product.strip() for r in reviews_b if r.product and r.product.strip()}

    target_product = product.strip() if product and product.strip() else None

    if target_product:
        reviews_a = [r for r in reviews_a if (r.product or "").strip().lower() == target_product.lower()]
        reviews_b = [r for r in reviews_b if (r.product or "").strip().lower() == target_product.lower()]
        if not reviews_a or not reviews_b:
            raise IngestError(
                "version_product_mismatch",
                f"Selected versions do not both exist for product '{target_product}'.",
                400,
            )
    else:
        # Check if products in A and B conflict
        if products_a and products_b and not (products_a & products_b):
            p_a_str = ", ".join(sorted(products_a))
            p_b_str = ", ".join(sorted(products_b))
            raise IngestError(
                "mixed_products",
                f"Version '{v_a}' belongs to product ({p_a_str}) while Version '{v_b}' belongs to product ({p_b_str}). "
                "Version comparison requires both versions to belong to the same product.",
                400,
            )
        common_products = products_a & products_b
        if common_products:
            target_product = sorted(common_products)[0]

    warnings: list[str] = []
    is_sparse = False
    if len(reviews_a) < 5 or len(reviews_b) < 5:
        is_sparse = True
        warnings.append(
            f"Sparse sample warning: Version A has {len(reviews_a)} reviews and Version B has {len(reviews_b)} reviews. "
            "Findings represent preliminary observations and should not be treated as confirmed statistical trends."
        )

    # 4. Compute Version Metrics
    metrics_a = _compute_version_metrics(v_a, reviews_a)
    metrics_b = _compute_version_metrics(v_b, reviews_b)

    # 5. Aspect-level comparison
    aspect_clauses_a: dict[str, list[Any]] = defaultdict(list)
    aspect_clauses_b: dict[str, list[Any]] = defaultdict(list)

    for r in reviews_a:
        for c in r.clauses:
            aspect_clauses_a[c.aspect or "general"].append(c)

    for r in reviews_b:
        for c in r.clauses:
            aspect_clauses_b[c.aspect or "general"].append(c)

    all_aspects = sorted(list(set(aspect_clauses_a.keys()) | set(aspect_clauses_b.keys())))
    aspect_comparisons: list[AspectComparisonItem] = []

    for asp in all_aspects:
        cl_a = aspect_clauses_a.get(asp, [])
        cl_b = aspect_clauses_b.get(asp, [])

        count_a = len(cl_a)
        complaints_a = sum(1 for c in cl_a if c.polarity == "neg")
        neg_rate_a = round(complaints_a / count_a, 4) if count_a else 0.0
        avg_sent_a = round(sum(c.score for c in cl_a) / count_a, 4) if count_a else 0.0

        count_b = len(cl_b)
        complaints_b = sum(1 for c in cl_b if c.polarity == "neg")
        neg_rate_b = round(complaints_b / count_b, 4) if count_b else 0.0
        avg_sent_b = round(sum(c.score for c in cl_b) / count_b, 4) if count_b else 0.0

        complaint_diff = complaints_b - complaints_a
        neg_rate_diff = round(neg_rate_b - neg_rate_a, 4)

        sample_note: str | None = None
        # Determine status
        if count_a == 0 and count_b > 0:
            status = "newly_observed"
        elif count_a > 0 and count_b == 0:
            # Distinguish "not observed" from "resolved" based on sample size
            if len(reviews_b) < 20:
                status = "not_observed"
                sample_note = (
                    f"Not observed in {v_b} sample ({len(reviews_b)} reviews), "
                    "but sample size is insufficient to declare the issue resolved."
                )
            else:
                status = "resolved"
                sample_note = f"Zero complaints observed across {len(reviews_b)} reviews in {v_b}."
        elif neg_rate_diff <= -0.15 or (complaints_a > 0 and complaints_b < complaints_a * 0.7):
            status = "improved"
        elif neg_rate_diff >= 0.15 or (complaints_a > 0 and complaints_b > complaints_a * 1.3):
            status = "worsened"
        else:
            status = "unchanged"

        aspect_comparisons.append(
            AspectComparisonItem(
                aspect=asp,
                aspect_label=_get_aspect_label(asp),
                count_a=count_a,
                complaint_count_a=complaints_a,
                negative_rate_a=neg_rate_a,
                avg_sentiment_a=avg_sent_a,
                count_b=count_b,
                complaint_count_b=complaints_b,
                negative_rate_b=neg_rate_b,
                avg_sentiment_b=avg_sent_b,
                complaint_diff=complaint_diff,
                negative_rate_diff=neg_rate_diff,
                status=status,
                sample_note=sample_note,
            )
        )

    # Sort aspects by total complaint volume descending
    aspect_comparisons.sort(key=lambda x: -(x.complaint_count_a + x.complaint_count_b))

    # 6. Issue-Cluster Comparison
    clusters_a = cluster_clauses(reviews_a)
    clusters_b = cluster_clauses(reviews_b)

    # Match clusters across versions by aspect and label
    cluster_comparisons: list[IssueClusterComparisonItem] = []
    seen_labels_b: set[str] = set()

    for cl_a in clusters_a:
        match_b = next(
            (cb for cb in clusters_b if cb.aspect == cl_a.aspect and cb.label.lower() == cl_a.label.lower()),
            None,
        )

        cnt_a = cl_a.review_count
        share_a = round(cnt_a / max(len(reviews_a), 1), 4)

        evidence_a = [
            EvidenceReference(
                source_type="review",
                reference_id=ev.review_id,
                excerpt_or_observation=ev.text,
                sentiment_score=ev.score,
            )
            for ev in cl_a.evidence[:3]
        ]

        if match_b:
            seen_labels_b.add(match_b.label.lower())
            cnt_b = match_b.review_count
            share_b = round(cnt_b / max(len(reviews_b), 1), 4)
            evidence_b = [
                EvidenceReference(
                    source_type="review",
                    reference_id=ev.review_id,
                    excerpt_or_observation=ev.text,
                    sentiment_score=ev.score,
                )
                for ev in match_b.evidence[:3]
            ]

            if share_b < share_a * 0.6:
                cl_status = "improved"
            elif share_b > share_a * 1.4:
                cl_status = "worsened"
            else:
                cl_status = "unchanged"

            cluster_comparisons.append(
                IssueClusterComparisonItem(
                    cluster_label=cl_a.label,
                    aspect=cl_a.aspect,
                    count_a=cnt_a,
                    share_a=share_a,
                    priority_a=cl_a.priority or "P2",
                    safety_tier_a=cl_a.safety_tier,
                    count_b=cnt_b,
                    share_b=share_b,
                    priority_b=match_b.priority or "P2",
                    safety_tier_b=match_b.safety_tier,
                    status=cl_status,
                    evidence_a=evidence_a,
                    evidence_b=evidence_b,
                )
            )
        else:
            # Cluster was in A but absent in B
            limitation = None
            if len(reviews_b) < 15:
                cl_status = "not_observed"
                limitation = f"Issue not observed in {v_b} sample ({len(reviews_b)} reviews); sample too small to confirm resolution."
            else:
                cl_status = "resolved"
                limitation = f"Issue no longer reported in {len(reviews_b)} reviews for {v_b}."

            cluster_comparisons.append(
                IssueClusterComparisonItem(
                    cluster_label=cl_a.label,
                    aspect=cl_a.aspect,
                    count_a=cnt_a,
                    share_a=share_a,
                    priority_a=cl_a.priority or "P2",
                    safety_tier_a=cl_a.safety_tier,
                    count_b=0,
                    share_b=0.0,
                    priority_b=None,
                    safety_tier_b="none",
                    status=cl_status,
                    sample_limitation=limitation,
                    evidence_a=evidence_a,
                    evidence_b=[],
                )
            )

    # Newly observed clusters in B
    for cl_b in clusters_b:
        if cl_b.label.lower() in seen_labels_b:
            continue
        cnt_b = cl_b.review_count
        share_b = round(cnt_b / max(len(reviews_b), 1), 4)
        evidence_b = [
            EvidenceReference(
                source_type="review",
                reference_id=ev.review_id,
                excerpt_or_observation=ev.text,
                sentiment_score=ev.score,
            )
            for ev in cl_b.evidence[:3]
        ]
        cluster_comparisons.append(
            IssueClusterComparisonItem(
                cluster_label=cl_b.label,
                aspect=cl_b.aspect,
                count_a=0,
                share_a=0.0,
                priority_a=None,
                safety_tier_a="none",
                count_b=cnt_b,
                share_b=share_b,
                priority_b=cl_b.priority or "P2",
                safety_tier_b=cl_b.safety_tier,
                status="newly_observed",
                sample_limitation=None,
                evidence_a=[],
                evidence_b=evidence_b,
            )
        )

    # Sort clusters by priority and volume
    prio_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, None: 4}
    cluster_comparisons.sort(
        key=lambda x: (
            prio_order.get(x.priority_b or x.priority_a, 4),
            -(x.count_a + x.count_b),
        )
    )

    # 7. Safety changes tracking
    safety_changes: list[str] = []
    if metrics_b.safety_alert_count > metrics_a.safety_alert_count:
        safety_changes.append(
            f"Safety alert volume increased from {metrics_a.safety_alert_count} in {v_a} to {metrics_b.safety_alert_count} in {v_b}."
        )
    elif metrics_b.safety_alert_count < metrics_a.safety_alert_count:
        safety_changes.append(
            f"Safety alert volume decreased from {metrics_a.safety_alert_count} in {v_a} to {metrics_b.safety_alert_count} in {v_b}."
        )
    new_tiers = set(metrics_b.safety_tiers) - set(metrics_a.safety_tiers)
    if new_tiers:
        safety_changes.append(f"Newly surfaced safety tiers in {v_b}: {', '.join(sorted(new_tiers))}.")

    # 8. Grounded summary of key changes
    improvements = [a.aspect_label for a in aspect_comparisons if a.status == "improved"]
    regressions = [a.aspect_label for a in aspect_comparisons if a.status == "worsened"]
    new_issues = [a.aspect_label for a in aspect_comparisons if a.status == "newly_observed"]

    summary_parts: list[str] = []
    if metrics_a.average_rating is not None and metrics_b.average_rating is not None:
        delta = round(metrics_b.average_rating - metrics_a.average_rating, 2)
        sign = "+" if delta >= 0 else ""
        summary_parts.append(f"Average rating shifted from {metrics_a.average_rating}★ to {metrics_b.average_rating}★ ({sign}{delta}★).")

    if improvements:
        summary_parts.append(f"Aspects with reduced complaint rates: {', '.join(improvements[:3])}.")
    if regressions:
        summary_parts.append(f"Aspects with increased complaint rates: {', '.join(regressions[:3])}.")
    if new_issues:
        summary_parts.append(f"Newly reported complaint areas in {v_b}: {', '.join(new_issues[:3])}.")

    if not summary_parts:
        summary_parts.append(f"Complaint frequencies and sentiment distributions remain largely stable between {v_a} and {v_b}.")

    summary_text = " ".join(summary_parts)

    disclaimer = (
        "OBSERVATIONAL COMPARISON ONLY: Differences between versions reflect customer feedback in the analyzed samples "
        "and do not establish direct causality. In sparse samples, absence of evidence is not evidence of absence."
    )

    return VersionCompareResponse(
        product=target_product,
        version_a=metrics_a,
        version_b=metrics_b,
        aspect_comparisons=aspect_comparisons,
        cluster_comparisons=cluster_comparisons,
        safety_changes=safety_changes,
        summary_of_changes=summary_text,
        disclaimer=disclaimer,
        is_sparse_sample=is_sparse,
        warnings=warnings,
    )


def compare_products(
    all_reviews: list[ReviewAnalysisResult] | list[ReviewInputItem],
    product_a: str,
    product_b: str,
    *,
    focus_aspect: str | None = None,
    mode: str = "lite",
) -> ProductCompareResponse:
    """Compare two distinct products with balanced trade-off analysis."""
    p_a = product_a.strip()
    p_b = product_b.strip()

    if not p_a or not p_b:
        raise IngestError("missing_product", "Both Product A and Product B must be specified.", 400)

    if p_a.lower() == p_b.lower():
        raise IngestError("same_product", "Cannot compare a product with itself. Select two distinct products.", 400)

    # 1. Ensure reviews are analyzed
    if not all_reviews:
        raise IngestError("empty_batch", "The review dataset is empty.", 400)

    if isinstance(all_reviews[0], ReviewInputItem):
        analysis_res = analyze(all_reviews, mode=mode)
        analyzed_reviews = analysis_res.reviews
    else:
        analyzed_reviews = all_reviews  # type: ignore[assignment]

    # 2. Filter reviews for Product A and Product B
    reviews_a = [r for r in analyzed_reviews if (r.product or "").strip().lower() == p_a.lower()]
    reviews_b = [r for r in analyzed_reviews if (r.product or "").strip().lower() == p_b.lower()]

    if not reviews_a:
        raise IngestError(
            "product_not_found",
            f"No reviews found for Product A ('{p_a}'). Please check the product identifier in the dataset.",
            404,
        )

    if not reviews_b:
        raise IngestError(
            "product_not_found",
            f"No reviews found for Product B ('{p_b}'). Please check the product identifier in the dataset.",
            404,
        )

    warnings: list[str] = []
    is_sparse = False
    if len(reviews_a) < 5 or len(reviews_b) < 5:
        is_sparse = True
        warnings.append(
            f"Sparse dataset warning: Product A has {len(reviews_a)} reviews and Product B has {len(reviews_b)} reviews. "
            "Results represent preliminary sample observations rather than conclusive market comparisons."
        )

    ratio = max(len(reviews_a), len(reviews_b)) / max(min(len(reviews_a), len(reviews_b)), 1)
    is_unbalanced = False
    if ratio > 4.0:
        is_unbalanced = True
        warnings.append(
            f"Sample size imbalance warning ({len(reviews_a)} vs {len(reviews_b)} reviews). "
            "Complaint counts should be evaluated using percentages rather than absolute totals."
        )

    # 3. Compute Product Metrics
    metrics_a = _compute_product_metrics(p_a, reviews_a)
    metrics_b = _compute_product_metrics(p_b, reviews_b)

    # 4. Aspect-by-Aspect Strengths & Weaknesses
    aspect_clauses_a: dict[str, list[Any]] = defaultdict(list)
    aspect_clauses_b: dict[str, list[Any]] = defaultdict(list)

    for r in reviews_a:
        for c in r.clauses:
            aspect_clauses_a[c.aspect or "general"].append(c)

    for r in reviews_b:
        for c in r.clauses:
            aspect_clauses_b[c.aspect or "general"].append(c)

    all_aspects = sorted(list(set(aspect_clauses_a.keys()) | set(aspect_clauses_b.keys())))
    aspect_strengths: list[ProductAspectStrengthItem] = []

    strengths_a: list[str] = []
    strengths_b: list[str] = []

    for asp in all_aspects:
        cl_a = aspect_clauses_a.get(asp, [])
        cl_b = aspect_clauses_b.get(asp, [])

        count_a = len(cl_a)
        complaints_a = sum(1 for c in cl_a if c.polarity == "neg")
        neg_rate_a = round(complaints_a / count_a, 4) if count_a else 0.0
        avg_sent_a = round(sum(c.score for c in cl_a) / count_a, 4) if count_a else 0.0

        count_b = len(cl_b)
        complaints_b = sum(1 for c in cl_b if c.polarity == "neg")
        neg_rate_b = round(complaints_b / count_b, 4) if count_b else 0.0
        avg_sent_b = round(sum(c.score for c in cl_b) / count_b, 4) if count_b else 0.0

        # Leader determination
        asp_label = _get_aspect_label(asp)
        sent_diff = avg_sent_a - avg_sent_b
        rate_diff = neg_rate_a - neg_rate_b

        if sent_diff >= 0.15 and rate_diff <= -0.10:
            leader = "product_a"
            strength_summary = f"{p_a} leads with higher sentiment ({avg_sent_a:+.2f} vs {avg_sent_b:+.2f}) and lower complaint rate ({round(neg_rate_a*100)}% vs {round(neg_rate_b*100)}%)."
            strengths_a.append(asp_label)
        elif sent_diff <= -0.15 and rate_diff >= 0.10:
            leader = "product_b"
            strength_summary = f"{p_b} leads with higher sentiment ({avg_sent_b:+.2f} vs {avg_sent_a:+.2f}) and lower complaint rate ({round(neg_rate_b*100)}% vs {round(neg_rate_a*100)}%)."
            strengths_b.append(asp_label)
        elif sent_diff > 0.08:
            leader = "product_a"
            strength_summary = f"{p_a} shows moderate advantage in customer satisfaction."
            strengths_a.append(asp_label)
        elif sent_diff < -0.08:
            leader = "product_b"
            strength_summary = f"{p_b} shows moderate advantage in customer satisfaction."
            strengths_b.append(asp_label)
        else:
            leader = "tie"
            strength_summary = "Comparable performance and complaint rates across both products."

        aspect_strengths.append(
            ProductAspectStrengthItem(
                aspect=asp,
                aspect_label=asp_label,
                leader=leader,
                sentiment_a=avg_sent_a,
                negative_rate_a=neg_rate_a,
                complaints_a=complaints_a,
                sentiment_b=avg_sent_b,
                negative_rate_b=neg_rate_b,
                complaints_b=complaints_b,
                strength_summary=strength_summary,
            )
        )

    # Sort aspects so focus_aspect is first if requested, followed by complaint volume
    clean_focus = focus_aspect.strip().lower() if focus_aspect else None
    aspect_strengths.sort(
        key=lambda x: (
            0 if clean_focus and x.aspect.lower() == clean_focus else 1,
            -(x.complaints_a + x.complaints_b),
        )
    )

    # 5. Common Complaint Themes vs Product-Specific Issues
    common_themes: list[str] = []
    unique_a: list[str] = []
    unique_b: list[str] = []

    for item in aspect_strengths:
        if item.complaints_a > 0 and item.complaints_b > 0:
            common_themes.append(f"{item.aspect_label} ({item.complaints_a} in {p_a}, {item.complaints_b} in {p_b})")
        elif item.complaints_a > 0 and item.complaints_b == 0:
            unique_a.append(f"{item.aspect_label} ({item.complaints_a} complaints)")
        elif item.complaints_b > 0 and item.complaints_a == 0:
            unique_b.append(f"{item.aspect_label} ({item.complaints_b} complaints)")

    # 6. Prominent Safety Alerts
    safety_alerts: list[dict[str, Any]] = []
    for r in reviews_a:
        if r.safety.is_alert:
            safety_alerts.append({
                "product": p_a,
                "review_id": r.review_id,
                "tier": r.safety.tier,
                "excerpt": r.safety.matches[0].excerpt if r.safety.matches else r.original_text[:120],
            })
    for r in reviews_b:
        if r.safety.is_alert:
            safety_alerts.append({
                "product": p_b,
                "review_id": r.review_id,
                "tier": r.safety.tier,
                "excerpt": r.safety.matches[0].excerpt if r.safety.matches else r.original_text[:120],
            })

    # 7. Key Evidence references
    evidence_a: list[EvidenceReference] = []
    for r in reviews_a:
        for c in r.clauses:
            if (c.polarity == "neg" or (r.rating is not None and r.rating <= 2.0)) and len(evidence_a) < 4:
                evidence_a.append(
                    EvidenceReference(
                        source_type="review",
                        reference_id=r.review_id,
                        excerpt_or_observation=c.text,
                        sentiment_score=c.score,
                    )
                )

    evidence_b: list[EvidenceReference] = []
    for r in reviews_b:
        for c in r.clauses:
            if (c.polarity == "neg" or (r.rating is not None and r.rating <= 2.0)) and len(evidence_b) < 4:
                evidence_b.append(
                    EvidenceReference(
                        source_type="review",
                        reference_id=r.review_id,
                        excerpt_or_observation=c.text,
                        sentiment_score=c.score,
                    )
                )

    # 8. Balanced Trade-Off Summary (No universal winner claim)
    tradeoff_parts: list[str] = []
    if strengths_a:
        tradeoff_parts.append(f"{p_a} demonstrates comparative advantages in {', '.join(strengths_a[:3])}.")
    if strengths_b:
        tradeoff_parts.append(f"{p_b} demonstrates comparative advantages in {', '.join(strengths_b[:3])}.")
    if not tradeoff_parts:
        tradeoff_parts.append(f"Both {p_a} and {p_b} exhibit closely balanced customer feedback profiles.")

    tradeoff_summary = " ".join(tradeoff_parts)

    # Tailored recommendation based on focus aspect or general use-case
    if clean_focus:
        match_focus = next((it for it in aspect_strengths if it.aspect.lower() == clean_focus), None)
        if match_focus:
            if match_focus.leader == "product_a":
                recommended_fit = f"Prioritizing {match_focus.aspect_label}: {p_a} is favored with higher satisfaction and fewer complaints."
            elif match_focus.leader == "product_b":
                recommended_fit = f"Prioritizing {match_focus.aspect_label}: {p_b} is favored with higher satisfaction and fewer complaints."
            else:
                recommended_fit = f"Prioritizing {match_focus.aspect_label}: Both products show comparable ratings and complaint rates."
        else:
            recommended_fit = f"Focus aspect '{focus_aspect}' was not directly observed in the customer review clauses."
    else:
        recommended_fit = (
            f"Selection depends on priority: choose {p_a} for {', '.join(strengths_a[:2]) if strengths_a else 'its balanced profile'}, "
            f"or choose {p_b} for {', '.join(strengths_b[:2]) if strengths_b else 'its overall feedback'}."
        )

    disclaimer = (
        "COMPARATIVE FEEDBACK ONLY: This report presents relative customer sentiment across the analyzed review samples. "
        "It does not constitute an absolute hardware test or universal endorsement."
    )

    return ProductCompareResponse(
        product_a=metrics_a,
        product_b=metrics_b,
        focus_aspect=focus_aspect,
        aspect_strengths=aspect_strengths,
        common_complaint_themes=common_themes,
        unique_issues_a=unique_a,
        unique_issues_b=unique_b,
        prominent_safety_alerts=safety_alerts,
        key_evidence_a=evidence_a,
        key_evidence_b=evidence_b,
        tradeoff_summary=tradeoff_summary,
        recommended_fit=recommended_fit,
        disclaimer=disclaimer,
        is_sparse=is_sparse,
        is_unbalanced=is_unbalanced,
        warnings=warnings,
    )
