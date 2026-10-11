"""Core review analysis orchestration pipeline (PRD v4 Sections 4, 5, 7, 11).

Coordinates existing NLP components into a single testable pipeline:
  1. Input ingestion and PII redaction (email and phone masks).
  2. Text normalization (with 1,000-char truncation). Offsets index into normalized text.
  3. Rating coercion, validation, and separate pseudo-rating estimation when rating is null.
  4. Language detection across English, Hindi, Hinglish, Kanglish, Tanglish, Tenglish.
  5. Safety screening with negation and context false-positive guards.
  6. Clause splitting with contrast conjunction and punctuation boundaries.
  7. Aspect detection (strong/weak terms, routing).
  8. Sentiment scoring with Rule 1 sarcasm and Rule 2 rating-conflict heuristics.
  9. Duplicate handling: traceable canonical linkage, with duplicates excluded from
     aggregate issue counts to prevent artificial inflation.
  10. Honest mode reporting: unsupported full mode rejected or flagged, lite mode verified.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import pandas as pd

from ingest import IngestError, coerce_and_validate_rating
from pipeline.aspects import detect_aspects
from pipeline.clauses import split_clauses
from pipeline.clustering import cluster_clauses
from pipeline.language import tag_language
from pipeline.normalize import NormalizedReview, normalize_review, normalize_text, to_text
from pipeline.prioritization import prioritize_issues
from pipeline.safety import highest_tier, scan_review
from pipeline.sentiment import sentiment_for_clause
from redaction import redact_pii
from schemas import (
    AnalysisMetadata,
    AnalysisSummary,
    AnalyzeRequest,
    AnalyzeResponse,
    ClauseAnalysisResult,
    ReviewAnalysisResult,
    ReviewInputItem,
    SafetyAnalysisResult,
    SafetyMatchDetail,
)

DEFAULT_MAX_BATCH_ROWS = 2000


def analyze_single_review(
    raw_text: Any,
    *,
    review_id: str | int | None = None,
    source_row: int | None = None,
    raw_rating: Any = None,
    date: str | None = None,
    version: str | None = None,
    product: str | None = None,
    is_duplicate: bool = False,
    duplicate_of: str | None = None,
    rating_source: str | None = None,
) -> tuple[ReviewAnalysisResult, list[str]]:
    """Analyze a single review through all pipeline stages.

    Redacts PII (emails and phone numbers) before normalization.
    All clause and safety match offsets index directly into ``normalized_text``.
    """
    row_warnings: list[str] = []
    r_id = str(review_id) if review_id is not None else str(source_row or 1)
    text_str = to_text(raw_text)

    # 1. PII Redaction: emails and phone numbers masked
    redacted_input = redact_pii(text_str)

    # 2. Rating coercion and validation
    r_val, r_src, out_range, inv = coerce_and_validate_rating(raw_rating)
    if rating_source is not None and r_val is not None:
        r_src = rating_source
    if out_range:
        row_warnings.append(f"Review {r_id}: rating coerced to 1-5 range.")
    if inv:
        row_warnings.append(f"Review {r_id}: dropped invalid non-numeric rating.")

    # 3. Empty or whitespace-only review handling
    if not redacted_input.strip():
        empty_safety = SafetyAnalysisResult(tier="none", is_alert=False, matches=[])
        res = ReviewAnalysisResult(
            review_id=r_id,
            original_text=redacted_input,
            normalized_text="",
            rating=r_val,
            rating_source=r_src,
            pseudo_rating=None,
            date=date,
            version=version,
            product=product,
            language="unknown",
            language_badge="Unknown",
            language_confidence=0.0,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
            truncated=False,
            sentiment_score=0.0,
            polarity="neu",
            label="neutral",
            sarcasm_suspect=False,
            rating_conflict=False,
            aspects=[],
            safety=empty_safety,
            clauses=[],
            status="empty",
        )
        return res, row_warnings

    # 4. Normalization (PRD 1,000-character cap with truncated flag)
    # Applied to redacted text so downstream clauses and offsets are aligned
    norm: NormalizedReview = normalize_review(redacted_input, source_row=source_row, max_chars=1000)

    # 5. Language detection
    lang_res = tag_language(norm.text)

    # 6. Safety scan (with negation & context guards)
    safety_matches = scan_review(norm)
    top_tier = highest_tier(safety_matches, default="none") or "none"
    match_details = [
        SafetyMatchDetail(
            category=m.category,
            tier=m.tier,  # type: ignore[arg-type]
            term=m.term,
            phrase=m.phrase,
            excerpt=m.excerpt,
            suggested_action=m.suggested_action,
        )
        for m in safety_matches
    ]
    top_action = safety_matches[0].suggested_action if safety_matches else None
    safety_res = SafetyAnalysisResult(
        tier=top_tier,  # type: ignore[arg-type]
        is_alert=(top_tier != "none"),
        matches=match_details,
        suggested_action=top_action,
    )

    # 7. Clause splitting: boundaries refer to norm.text
    clauses = split_clauses(norm)

    # 8. Aspect detection & clause sentiment (with sarcasm & rating conflicts)
    clause_results: list[ClauseAnalysisResult] = []
    unique_aspects: set[str] = set()
    clause_scores: list[float] = []
    any_sarcasm = False
    any_conflict = False

    for idx, c in enumerate(clauses):
        # Aspects for this clause
        aspect_matches = detect_aspects(c)
        aspect_names = sorted(list({m.aspect for m in aspect_matches}))
        for a in aspect_names:
            unique_aspects.add(a)
        primary_aspect = aspect_names[0] if aspect_names else None

        # Sentiment for this clause with context of next clause
        # Rating conflict only evaluates observed r_val (not pseudo)
        next_c = clauses[idx + 1] if idx + 1 < len(clauses) else None
        sent = sentiment_for_clause(c, rating=r_val, next_clause=next_c)

        if sent.sarcasm_suspect:
            any_sarcasm = True
        if sent.rating_conflict:
            any_conflict = True
        clause_scores.append(sent.score)

        clause_results.append(
            ClauseAnalysisResult(
                clause_id=f"{r_id}_{idx + 1}",
                text=c.text,
                aspect=primary_aspect,
                aspects=aspect_names,
                score=sent.score,
                polarity=sent.polarity,  # type: ignore[arg-type]
                label=sent.label,
                confidence=sent.confidence,
                sarcasm_suspect=sent.sarcasm_suspect,
                rating_conflict=sent.rating_conflict,
                language=c.language,
                start=c.start,
                end=c.end,
            )
        )

    # 9. Review-level sentiment rollup
    if clause_scores:
        avg_score = round(sum(clause_scores) / len(clause_scores), 4)
    else:
        avg_score = 0.0

    if avg_score >= 0.25:
        polarity = "pos"
        label = "positive"
    elif avg_score <= -0.25:
        polarity = "neg"
        label = "negative"
    else:
        polarity = "neu"
        label = "neutral"

    # 10. Pseudo-rating handling (PRD Sections 5 and 7.1)
    # If user provided observed rating, keep rating and set pseudo_rating=None.
    # If observed rating is missing, keep rating=None and provide labeled pseudo_rating.
    if r_val is not None:
        observed_rating = r_val
        pseudo_rating = None
        final_rating_source = r_src or "given"
    else:
        observed_rating = None
        # Linear mapping from [-1.0, 1.0] to [1.0, 5.0]
        pseudo_rating = round(min(5.0, max(1.0, 3.0 + 2.0 * avg_score)), 1)
        final_rating_source = "pseudo"

    review_res = ReviewAnalysisResult(
        review_id=r_id,
        original_text=redacted_input,
        normalized_text=norm.text,
        rating=observed_rating,
        rating_source=final_rating_source,
        pseudo_rating=pseudo_rating,
        date=date,
        version=version,
        product=product,
        language=lang_res.code,
        language_badge=lang_res.badge,
        language_confidence=lang_res.confidence,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        truncated=norm.truncated,
        sentiment_score=avg_score,
        polarity=polarity,
        label=label,
        sarcasm_suspect=any_sarcasm,
        rating_conflict=any_conflict,
        aspects=sorted(list(unique_aspects)),
        safety=safety_res,
        clauses=clause_results,
        status="analyzed",
    )
    return review_res, row_warnings


def make_duplicate_key(
    text: str,
    rating: Any = None,
    product: str | None = None,
    version: str | None = None,
) -> str | None:
    """Safe composite duplicate key generator.

    Safe duplicate rules:
    - Text: Redacted, normalized, lowercase, whitespace-collapsed.
    - Rating: Reviews with differing ratings (e.g. 5 vs 1) are never merged as duplicates.
    - Product: Reviews with differing product names are never merged as duplicates.
    - Version: Reviews with differing versions (e.g. v1.0 vs v2.0) are never merged as duplicates.
    """
    clean_text_val = to_text(text).strip()
    if not clean_text_val:
        return None
    redacted = redact_pii(clean_text_val)
    norm_text = " ".join(normalize_text(redacted, lowercase=True).split())
    if not norm_text:
        return None

    # Rating signature
    r_val, _, _, _ = coerce_and_validate_rating(rating)
    r_sig = f"r:{r_val}" if r_val is not None else "r:none"

    # Product signature
    p_clean = product.strip().lower() if product and isinstance(product, str) and product.strip() else ""
    p_sig = f"p:{p_clean}" if p_clean else "p:none"

    # Version signature
    v_clean = version.strip().lower() if version and isinstance(version, str) and version.strip() else ""
    v_sig = f"v:{v_clean}" if v_clean else "v:none"

    return f"{norm_text}|{r_sig}|{p_sig}|{v_sig}"


def analyze_reviews(
    reviews: list[ReviewInputItem] | list[dict[str, Any]],
    *,
    mode: str = "lite",
    fallback_to_lite: bool = False,
    max_rows: int = DEFAULT_MAX_BATCH_ROWS,
) -> AnalyzeResponse:
    """Analyze a batch of reviews and assemble aggregate metadata.

    Preserves exact duplicate traceability (linking to canonical review)
    while excluding duplicates from aggregate issue counts to prevent artificial inflation.
    Honest mode reporting: rejects unsupported full mode unless fallback_to_lite is True.
    """
    if not reviews:
        raise IngestError("empty_batch", "The reviews list must not be empty.", 400)
    if len(reviews) > max_rows:
        raise IngestError(
            "payload_too_large",
            f"Batch exceeds maximum limit of {max_rows} reviews.",
            413,
        )

    all_warnings: list[str] = []

    # Processing mode verification
    mode_lower = mode.strip().lower()
    if mode_lower == "full":
        if not fallback_to_lite:
            raise IngestError(
                "unsupported_mode",
                "Full mode (transformer/embeddings) is not available in this Lite environment. Use mode='lite'.",
                400,
            )
        effective_mode = "lite"
        requested_mode = "full"
        all_warnings.append(
            "Requested mode 'full' is unavailable (transformer models not installed); processed in 'lite' mode."
        )
    elif mode_lower == "lite":
        effective_mode = "lite"
        requested_mode = "lite"
    else:
        raise IngestError("invalid_mode", f"Unsupported mode '{mode}'. Use 'lite'.", 400)

    start_time = time.perf_counter()
    analyzed_list: list[ReviewAnalysisResult] = []

    # Duplicate tracking: map normalized composite key to canonical review_id
    seen_canonical: dict[str, str] = {}
    dup_count = 0

    # Aggregate metric collectors (canonical non-duplicate reviews only)
    lang_counter: Counter[str] = Counter()
    sentiment_counter: Counter[str] = Counter()
    safety_counter: Counter[str] = Counter()
    aspect_counter: Counter[str] = Counter()
    sarcasm_count = 0
    rating_conflict_count = 0
    total_clauses = 0

    for idx, item in enumerate(reviews):
        # Support either ReviewInputItem or dict
        if isinstance(item, ReviewInputItem):
            r_id = str(item.review_id or item.id or idx + 1)
            raw_text = item.text
            rating = item.rating
            date = item.date
            version = item.version
            product = item.product
        else:
            r_id = str(item.get("review_id") or item.get("id") or item.get("source_row") or idx + 1)
            raw_text = item.get("text") or item.get("text_redacted") or item.get("review_text") or ""
            rating = item.get("rating")
            date = item.get("date")
            version = item.get("version")
            product = item.get("product")

        # Duplicate key based on normalized text and compatible metadata
        dup_key = make_duplicate_key(raw_text, rating=rating, product=product, version=version)

        is_dup = False
        duplicate_of: str | None = None
        if dup_key:
            if dup_key in seen_canonical:
                is_dup = True
                duplicate_of = seen_canonical[dup_key]
                dup_count += 1
            else:
                seen_canonical[dup_key] = r_id

        review_res, row_warns = analyze_single_review(
            raw_text,
            review_id=r_id,
            source_row=idx + 1,
            raw_rating=rating,
            date=date,
            version=version,
            product=product,
            is_duplicate=is_dup,
            duplicate_of=duplicate_of,
        )
        analyzed_list.append(review_res)
        all_warnings.extend(row_warns)

        # Aggregate counts: Canonical (non-duplicate) reviews only
        # This prevents repeated spam/duplicates from inflating issue counts
        if not review_res.is_duplicate and review_res.status == "analyzed":
            lang_counter[review_res.language] += 1
            sentiment_counter[review_res.polarity] += 1
            safety_counter[review_res.safety.tier] += 1
            if review_res.sarcasm_suspect:
                sarcasm_count += 1
            if review_res.rating_conflict:
                rating_conflict_count += 1
            total_clauses += len(review_res.clauses)
            for a in review_res.aspects:
                aspect_counter[a] += 1

    if dup_count > 0:
        all_warnings.append(
            f"Flagged {dup_count} exact duplicate review(s); excluded from aggregate issue counts to prevent artificial inflation."
        )

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    analyzed_count = sum(1 for r in analyzed_list if r.status == "analyzed" and not r.is_duplicate)
    unique_reviews = len(reviews) - dup_count

    summary = AnalysisSummary(
        total_reviews=len(reviews),
        analyzed_reviews=analyzed_count,
        unique_reviews=unique_reviews,
        duplicate_count=dup_count,
        total_clauses=total_clauses,
        languages=dict(lang_counter),
        sentiment_breakdown=dict(sentiment_counter),
        safety_breakdown=dict(safety_counter),
        sarcasm_count=sarcasm_count,
        rating_conflict_count=rating_conflict_count,
        aspect_mentions=dict(aspect_counter),
        warnings=all_warnings,
    )
    meta = AnalysisMetadata(
        mode=effective_mode,  # type: ignore[arg-type]
        requested_mode=requested_mode,
        processing_time_ms=duration_ms,
        summary=summary,
    )

    # Phase 2: Issue clustering and prioritization (PRD Sections 5, 7.1, 7.5)
    clusters = cluster_clauses(analyzed_list)
    insights = prioritize_issues(analyzed_list, clusters=clusters)

    # Attach cluster_id back to clauses for PRD Table 'clauses' traceability
    clause_to_cluster: dict[str | int, str] = {}
    for cl in clusters:
        for cid in cl.clause_ids:
            clause_to_cluster[cid] = cl.cluster_id

    for rev in analyzed_list:
        for c in rev.clauses:
            if c.clause_id in clause_to_cluster:
                c.cluster_id = clause_to_cluster[c.clause_id]

    return AnalyzeResponse(
        reviews=analyzed_list,
        meta=meta,
        clusters=clusters,
        insights=insights,
    )


def analyze(
    data: list[ReviewInputItem] | list[dict[str, Any]] | pd.DataFrame | AnalyzeRequest,
    mode: str = "lite",
    *,
    fallback_to_lite: bool = False,
    max_rows: int = DEFAULT_MAX_BATCH_ROWS,
) -> AnalyzeResponse:
    """PRD Section 5 primary entry point: ``analyze(df, mode) -> Results``."""
    if isinstance(data, AnalyzeRequest):
        return analyze_reviews(
            data.reviews, mode=data.mode, fallback_to_lite=fallback_to_lite, max_rows=max_rows
        )
    if isinstance(data, pd.DataFrame):
        records: list[dict[str, Any]] = []
        for idx, row in data.iterrows():
            rec = {str(k): v for k, v in row.items()}
            if "review_id" not in rec and "id" not in rec and "source_row" in rec:
                rec["review_id"] = str(rec["source_row"])
            records.append(rec)
        return analyze_reviews(
            records, mode=mode, fallback_to_lite=fallback_to_lite, max_rows=max_rows
        )
    return analyze_reviews(
        data, mode=mode, fallback_to_lite=fallback_to_lite, max_rows=max_rows
    )
