"""Root-cause hypothesis generation and verification engine.

Uses clustered customer review symptoms, representative evidence, temporal
patterns, and available vision defect findings to generate ranked,
evidence-grounded failure hypotheses with verification steps and falsification criteria.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from schemas import (
    EvidenceReference,
    HypothesisItem,
    ImageDefectAnalysis,
    IssueCluster,
    ReviewAnalysisResult,
    RootCauseAnalysisResult,
    RootCauseBatchResponse,
)

logger = logging.getLogger("insightforge.root_cause")

# Generic non-descriptive words that do not specify engineering failure mechanisms
GENERIC_STOPWORDS = frozenset(
    {
        "bad", "terrible", "worst", "hate", "awful", "useless", "garbage", "trash",
        "horrible", "poor", "disappointed", "waste", "product", "item", "phone",
        "thing", "very", "too", "much", "buy", "bought", "money", "ever", "really",
    }
)

# Domain failure mechanism patterns keyed by aspect taxonomy and aliases
SYMPTOM_PATTERNS: dict[str, list[tuple[str, str, str, str, str]]] = {
    "battery": [
        (
            r"heat|hot|warm|burn|overheat|fire|scorching",
            "Charge controller IC switching dissipation or unthrottled thermal dissipation during fast-charge profile.",
            "thermal_power",
            "Instrument 5 return units with micro-thermocouples; log power-management IC and battery temperature under 18W/30W charging in a 25°C chamber.",
            "If PMIC surface temperature remains below 42°C during full-rate charge, reject the charge controller dissipation hypothesis.",
        ),
        (
            r"drain|fast|die|hours|overnight|sudden|deplete|depleting",
            "Unthrottled background application wake-locks or Bluetooth low-energy peripheral polling preventing deep sleep SoC states.",
            "software_firmware",
            "Capture 24-hour battery historian telemetry and kernel wake-lock logs on affected firmware versions.",
            "If average standby current draw is confirmed under 12mA, reject the software wake-lock hypothesis.",
        ),
        (
            r"charge|charging|cable|port|plug|connect|loose",
            "USB-C connector spring-contact mechanical tolerance variance or particulate-induced pin seating impedance.",
            "hardware_component",
            "Measure USB-C insertion/retention force and contact resistance across 50 cycles on returned samples.",
            "If VBUS-to-GND contact resistance measures nominal (<30mΩ), reject the connector tolerance hypothesis.",
        ),
    ],
    "quality": [
        (
            r"crack|break|snap|shatter|fractur|cracked",
            "Mechanical stress concentration at injection molding weld line or inadequate rib wall thickness under torsional flexure.",
            "assembly_fit",
            "Conduct four-point bend structural analysis and microscopic cross-section inspection of the housing seam weld line.",
            "If seam tensile yield strength exceeds 45 MPa under ASTM D638 testing, reject the weld line defect hypothesis.",
        ),
        (
            r"scratch|peel|finish|coating|rub|cosmetic",
            "Sub-optimal surface primer adhesion or insufficient UV-cured topcoat cross-linking density on outer enclosure.",
            "assembly_fit",
            "Execute cross-hatch tape adhesion test (ASTM D3359) and pencil hardness test (ASTM D3363) on pristine production samples.",
            "If coating achieves 5B adhesion rating and 3H hardness without flaking, reject the primer adhesion hypothesis.",
        ),
        (
            r"creak|loose|gap|wobble|fit|misalign|rattle",
            "Tolerance stack-up in internal snap-fit latch engagements permitting relative mechanical shift under hand pressure.",
            "assembly_fit",
            "Perform optical coordinate measuring machine (CMM) dimensional verification of snap-fit clearance tolerances.",
            "If gap variance is within design specification (±0.08mm), reject the snap-fit clearance hypothesis.",
        ),
    ],
    "sound": [
        (
            r"crackl|static|buzz|hum|distort|pop",
            "Audio DAC buffer underrun or I2S digital audio clock jitter during concurrent CPU burst processing.",
            "software_firmware",
            "Measure I2S clock line jitter with high-bandwidth oscilloscope and audit DMA audio buffer queue depths under 90% CPU load.",
            "If I2S clock jitter is under 40ps and zero buffer underruns occur during CPU spikes, reject the digital buffer hypothesis.",
        ),
        (
            r"quiet|low|muffl|faint|volume|tinny",
            "Acoustic mesh acoustic resistance variance or partial acoustic port seal occlusion during assembly adhesive dispensing.",
            "hardware_component",
            "Measure acoustic impedance and frequency response (FR) curves on an artificial ear simulator (IEC 60318-4).",
            "If acoustic SPL curve matches golden reference ±1.5dB between 100Hz and 10kHz, reject the port occlusion hypothesis.",
        ),
    ],
    "stability": [
        (
            r"crash|freeze|hang|lag|slow|bug|lock|closing|stopped",
            "Memory leak or unhandled null-pointer exception in UI rendering thread during background state synchronization.",
            "software_firmware",
            "Analyze crash stack traces from automated fuzzing and test memory footprint over 8-hour continuous UI monkey test.",
            "If heap memory remains stable and zero unhandled exceptions trigger, reject the UI thread leak hypothesis.",
        ),
    ],
    "packaging": [
        (
            r"crush|dent|open|box|transit|ship|damage",
            "Carton corrugated flute crush strength insufficiency (ECT rating) under standard pallet stacking transit loads.",
            "packaging_transit",
            "Perform ISTA-2A drop, compression, and vibration package transit simulation tests.",
            "If primary retail packaging passes 10-drop cycle at 76cm without carton deformation, reject packaging insufficiency.",
        ),
    ],
    "general": [
        (
            r"crack|fractur|break|snap",
            "Mechanical stress concentration at injection molding weld line or inadequate rib wall thickness under torsional flexure.",
            "assembly_fit",
            "Conduct four-point bend structural analysis and microscopic cross-section inspection of the seam weld line.",
            "If seam tensile yield strength exceeds 45 MPa under ASTM D638 testing, reject the weld line defect hypothesis.",
        ),
        (
            r"disconnect|drop|cut|stutter|range|signal",
            "RF antenna ground-plane detuning caused by capacitive hand-coupling or impedance mismatch in matching network.",
            "hardware_component",
            "Measure passive antenna efficiency and Over-The-Air (OTA) TRP/TIS inside an anechoic chamber.",
            "If TRP degradation under hand grip is within 2.5dB across target channels, reject the antenna detuning hypothesis.",
        ),
    ],
}

# Alias mapping for aspect lookups
ASPECT_ALIASES: dict[str, str] = {
    "build_quality": "quality",
    "hardware": "quality",
    "audio": "sound",
    "software": "stability",
    "performance": "stability",
    "connectivity": "general",
}


def _extract_symptom_tokens(text: str) -> list[str]:
    """Extract informative non-stopword tokens from review text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in GENERIC_STOPWORDS]


def generate_root_cause_hypotheses(
    cluster: IssueCluster,
    all_reviews: list[ReviewAnalysisResult],
    image_findings: list[ImageDefectAnalysis] | None = None,
) -> RootCauseAnalysisResult:
    """Generate evidence-grounded root-cause hypotheses for an issue cluster."""
    image_findings = image_findings or []

    canonical_aspect = ASPECT_ALIASES.get(cluster.aspect.lower(), cluster.aspect.lower())

    # 1. Filter associated image findings matching this cluster or aspect
    relevant_images: list[ImageDefectAnalysis] = [
        img for img in image_findings
        if (
            (img.relevant_issue_cluster and (
                img.relevant_issue_cluster.lower() in cluster.label.lower()
                or cluster.label.lower() in img.relevant_issue_cluster.lower()
            ))
            or (img.relevant_aspect and ASPECT_ALIASES.get(img.relevant_aspect.lower(), img.relevant_aspect.lower()) == canonical_aspect)
            or (canonical_aspect in ("quality", "general") and img.is_defect_observed)
        )
    ]

    # 2. Check evidence sufficiency
    has_image_evidence = any(img.is_defect_observed for img in relevant_images)
    total_reviews = cluster.review_count

    # Check if complaints have informative engineering symptoms
    combined_cluster_text = " ".join([ev.text for ev in cluster.evidence] + [cluster.example_quote])
    symptom_tokens = _extract_symptom_tokens(combined_cluster_text)

    if (total_reviews < 2 and not has_image_evidence) or len(symptom_tokens) < 2:
        reason = (
            f"Insufficient evidence: Cluster '{cluster.label}' contains only {total_reviews} customer report(s) "
            "without corroborating physical defect photos. Formulating responsible engineering hypotheses "
            "requires at least 2 independent customer corroborations or verified defect imagery."
        )
        if total_reviews >= 2 and len(symptom_tokens) < 2:
            reason = (
                f"Insufficient evidence: Reviews in cluster '{cluster.label}' lack descriptive failure symptoms "
                "(only generic dissatisfaction without specific physical or technical failure symptoms). "
                "Measurable engineering symptoms are required to deduce failure mechanisms."
            )

        return RootCauseAnalysisResult(
            cluster_id=cluster.cluster_id,
            issue_label=cluster.label,
            aspect=cluster.aspect,
            status="insufficient_evidence",
            has_sufficient_evidence=False,
            summary="Evidence insufficient to infer grounded engineering hypotheses.",
            hypotheses=[],
            insufficient_evidence_reason=reason,
            total_supporting_reviews=total_reviews,
            total_contradicting_reviews=0,
            associated_images_count=len(relevant_images),
            warnings=[reason],
        )

    # 3. Assemble Supporting Evidence References
    supporting_refs: list[EvidenceReference] = []
    seen_refs: set[str] = set()

    for ev in cluster.evidence:
        ref_id = str(ev.review_id)
        if ref_id not in seen_refs:
            seen_refs.add(ref_id)
            supporting_refs.append(
                EvidenceReference(
                    source_type="review",
                    reference_id=ref_id,
                    excerpt_or_observation=ev.text,
                    metadata={"clause_id": ev.clause_id, "score": ev.score},
                )
            )

    for img in relevant_images:
        for obs in img.visual_observations:
            supporting_refs.append(
                EvidenceReference(
                    source_type="image",
                    reference_id=img.image_id,
                    excerpt_or_observation=f"Visual Observation: {obs}",
                    metadata={"defect_category": img.potential_defect_category, "severity": img.severity},
                )
            )

    # 4. Assemble Contradicting Evidence & Missing Information
    contradicting_refs: list[EvidenceReference] = []
    for r in all_reviews:
        if r.review_id not in cluster.review_ids:
            for cl in r.clauses:
                cl_aspects = [ASPECT_ALIASES.get(a.lower(), a.lower()) for a in (cl.aspects or [])]
                cl_primary = ASPECT_ALIASES.get(cl.aspect.lower(), cl.aspect.lower()) if cl.aspect else ""
                
                aspect_matched = (canonical_aspect in cl_aspects) or (cl_primary == canonical_aspect)
                positive_signal = (cl.polarity == "pos") or (cl.score > 0.05) or (r.rating is not None and r.rating >= 4)

                if aspect_matched and positive_signal:
                    contradicting_refs.append(
                        EvidenceReference(
                            source_type="review",
                            reference_id=r.review_id,
                            excerpt_or_observation=f"Positive on same aspect: '{cl.text}'",
                            metadata={"score": cl.score, "rating": r.rating},
                        )
                    )
                    break
        if len(contradicting_refs) >= 3:
            break

    missing_info: list[str] = [
        "Production batch lot numbers and PCBA serial identifiers are not present in customer reviews.",
        "Operating ambient temperatures and customer usage environment variables are unmeasured.",
    ]
    if not relevant_images:
        missing_info.append("Physical photographic evidence not available for this issue cluster; analysis relies on textual review symptoms.")
    else:
        missing_info.append("Internal silicon and circuit layer integrity cannot be evaluated from external photo inspection.")

    # 5. Temporal Pattern Analysis
    cluster_dates = [r.date for r in all_reviews if r.review_id in cluster.review_ids and r.date]
    cluster_versions = [r.version for r in all_reviews if r.review_id in cluster.review_ids and r.version]

    temporal_summary = "Complaints distributed across the monitoring timeframe without isolated date spikes."
    if cluster_dates:
        unique_dates = sorted(list(set(cluster_dates)))
        temporal_summary = f"Symptoms observed across {len(unique_dates)} distinct dates (earliest: {unique_dates[0]}, latest: {unique_dates[-1]})."

    if cluster_versions:
        v_counts = defaultdict(int)
        for v in cluster_versions:
            v_counts[v] += 1
        most_common_v = max(v_counts, key=v_counts.get)
        if len(v_counts) > 1:
            temporal_summary += f" Concentrated primarily on version '{most_common_v}' ({v_counts[most_common_v]}/{len(cluster_versions)} versioned reports)."

    # 6. Generate Hypotheses based on Aspect & Symptoms
    aspect_key = canonical_aspect
    patterns = SYMPTOM_PATTERNS.get(aspect_key) or SYMPTOM_PATTERNS.get("general", [])

    matched_candidates: list[tuple[str, str, str, str, str]] = []
    text_lower = combined_cluster_text.lower()

    for pattern_re, cause, cat, verif, falsif in patterns:
        if re.search(pattern_re, text_lower):
            matched_candidates.append((pattern_re, cause, cat, verif, falsif))

    # Also check general patterns if nothing matched in specific aspect
    if not matched_candidates and aspect_key != "general":
        for pattern_re, cause, cat, verif, falsif in SYMPTOM_PATTERNS.get("general", []):
            if re.search(pattern_re, text_lower):
                matched_candidates.append((pattern_re, cause, cat, verif, falsif))

    # If image observations specifically point to structural defect, prioritize that
    for img in relevant_images:
        if img.potential_defect_category in ("casing_crack", "cosmetic_scratch", "misalignment") and img.is_defect_observed:
            img_cause = (
                f"Physical stress concentration or assembly seam fatigue corroborated by photo evidence ({img.potential_defect_category})."
            )
            matched_candidates.insert(
                0,
                (
                    "photo_defect",
                    img_cause,
                    "assembly_fit",
                    f"Perform microscopic dye penetrant or cross-section analysis on physical samples exhibiting {img.potential_defect_category}.",
                    f"If casing material meets specified elongation-at-break under ASTM D638, reject structural fatigue hypothesis.",
                ),
            )

    # Fallback hypothesis if no specific symptom keyword triggered
    if not matched_candidates:
        matched_candidates.append((
            "general_aspect",
            f"Component variance or interface tolerance deviation affecting {cluster.aspect.replace('_', ' ')}.",
            "hardware_component",
            f"Conduct diagnostic bench screening on 5 returned units to isolate {cluster.aspect} functional performance.",
            f"If all return units operate within engineering specification, reject hardware component deviation.",
        ))

    # Deduplicate candidate causes
    seen_causes: set[str] = set()
    deduped_candidates = []
    for cand in matched_candidates:
        if cand[1] not in seen_causes:
            seen_causes.add(cand[1])
            deduped_candidates.append(cand)

    hypotheses: list[HypothesisItem] = []
    for idx, (_, cause, cat, verif, falsif) in enumerate(deduped_candidates[:3]):
        conf_score = 0.52
        if total_reviews >= 3:
            conf_score += 0.10
        if has_image_evidence:
            conf_score += 0.18
        if cluster_versions and len(set(cluster_versions)) == 1:
            conf_score += 0.08
        if contradicting_refs:
            conf_score -= 0.08

        conf_score = max(0.25, min(0.88, conf_score - (idx * 0.10)))
        conf_level = "high" if conf_score >= 0.72 else ("medium" if conf_score >= 0.45 else "low")

        hypotheses.append(
            HypothesisItem(
                hypothesis_id=f"hyp_{cluster.cluster_id}_{idx + 1}",
                rank=idx + 1,
                proposed_cause=cause,
                category=cat,
                confidence_score=round(conf_score, 2),
                confidence_level=conf_level,
                explanation=(
                    f"Deducted from {len(supporting_refs)} supporting review/image evidence points in cluster '{cluster.label}'. "
                    f"Temporal profile indicates: {temporal_summary}"
                ),
                supporting_evidence=supporting_refs[:5],
                contradicting_evidence=contradicting_refs,
                missing_information=missing_info,
                verification_steps=[
                    verif,
                    "Review component manufacturing quality records for lots shipped during the active complaint window.",
                ],
                falsification_criteria=falsif,
                is_hypothesis_only=True,
            )
        )

    summary_text = (
        f"Generated {len(hypotheses)} plausible engineering hypotheses for '{cluster.label}' ({cluster.aspect}). "
        f"Supported by {len(supporting_refs)} evidence points; {len(contradicting_refs)} contradicting reference(s) found."
    )

    return RootCauseAnalysisResult(
        cluster_id=cluster.cluster_id,
        issue_label=cluster.label,
        aspect=cluster.aspect,
        status="hypotheses_generated",
        has_sufficient_evidence=True,
        summary=summary_text,
        hypotheses=hypotheses,
        insufficient_evidence_reason=None,
        total_supporting_reviews=total_reviews,
        total_contradicting_reviews=len(contradicting_refs),
        associated_images_count=len(relevant_images),
        temporal_pattern_summary=temporal_summary,
        warnings=[],
    )


def batch_generate_hypotheses(
    clusters: list[IssueCluster],
    all_reviews: list[ReviewAnalysisResult],
    image_findings: list[ImageDefectAnalysis] | None = None,
    target_cluster_id: str | None = None,
) -> RootCauseBatchResponse:
    """Run root cause hypothesis engine across target or all negative/unhappy clusters."""
    image_findings = image_findings or []
    target_clusters = clusters

    if target_cluster_id:
        target_clusters = [c for c in clusters if c.cluster_id == target_cluster_id]

    results: list[RootCauseAnalysisResult] = []
    generated_count = 0
    insufficient_count = 0

    for cl in target_clusters:
        # Evaluate negative clusters, safety alerts, or low-rated complaint clusters (<= 2.5 stars)
        is_negative = cl.polarity == "neg"
        is_safety = cl.is_safety_alert
        is_low_rated = cl.average_rating is not None and cl.average_rating <= 2.5

        if not (is_negative or is_safety or is_low_rated):
            continue

        res = generate_root_cause_hypotheses(
            cluster=cl,
            all_reviews=all_reviews,
            image_findings=image_findings,
        )
        results.append(res)
        if res.has_sufficient_evidence:
            generated_count += 1
        else:
            insufficient_count += 1

    return RootCauseBatchResponse(
        results=results,
        total_clusters_analyzed=len(results),
        clusters_with_hypotheses=generated_count,
        clusters_with_insufficient_evidence=insufficient_count,
        warnings=[],
    )
