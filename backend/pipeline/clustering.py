"""Issue clustering pipeline for grouping review clauses into explainable clusters.

PRD Section 5 Table 'clusters' & Section 7.5:
- Group related review clauses into explainable issue clusters.
- Preserve links to supporting review IDs and clause evidence.
- Aggregate aspect, sentiment, safety, frequency, and available rating information per cluster.
- In Lite mode: Character & word TF-IDF vectorization with agglomerative cosine clustering.
- Routing information attached from config/routing.yaml.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from pipeline._common import load_yaml
from schemas import (
    ClauseAnalysisResult,
    IssueCluster,
    IssueClusterClauseEvidence,
    ReviewAnalysisResult,
)

DEFAULT_CLUSTER_DISTANCE_THRESHOLD = 0.40

# Stopwords to filter out of cluster labeling and TF-IDF
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "we", "us", "our", "i",
        "me", "my", "you", "your", "he", "him", "his", "she", "her", "hai", "hain", "ke",
        "ki", "ko", "ka", "se", "me", "mein", "par", "bhi", "tha", "the", "thi", "aur",
        "ya", "to", "ho", "hota", "hoti", "hote", "karna", "kare", "kar", "phone", "product",
        "item", "app", "good", "bad", "great", "nice", "very", "too", "much", "romba",
        "chala", "tumba", "bahut",
    }
)

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def _tokenize_clause(text: str) -> list[str]:
    """Tokenize clause text into word unigrams, bigrams, and character n-grams."""
    raw_words = [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 1]
    filtered_words = [w for w in raw_words if w not in STOPWORDS]

    features: list[str] = list(filtered_words)

    # Word bigrams
    for i in range(len(filtered_words) - 1):
        features.append(f"{filtered_words[i]}_{filtered_words[i+1]}")

    # Character n-grams (length 3 and 4) for Hinglish spelling variants
    for word in filtered_words:
        if len(word) >= 3:
            for n in (3, 4):
                for i in range(len(word) - n + 1):
                    features.append(f"c:{word[i:i+n]}")

    return features if features else raw_words


def _compute_tfidf_matrix(tokenized_docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Compute normalized TF-IDF matrix for a list of tokenized documents."""
    num_docs = len(tokenized_docs)
    if num_docs == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # Document frequency
    df: Counter[str] = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))

    # Build vocabulary: words appearing in at least 1 document
    vocab = sorted(df.keys())
    vocab_idx = {term: idx for idx, term in enumerate(vocab)}
    num_features = len(vocab)

    if num_features == 0:
        return np.zeros((num_docs, 0), dtype=np.float32), []

    # IDF weights
    idf = np.zeros(num_features, dtype=np.float32)
    for term, idx in vocab_idx.items():
        idf[idx] = math.log((num_docs + 1.0) / (df[term] + 1.0)) + 1.0

    # TF-IDF matrix
    matrix = np.zeros((num_docs, num_features), dtype=np.float32)
    for row_idx, tokens in enumerate(tokenized_docs):
        term_counts = Counter(tokens)
        doc_len = max(1, len(tokens))
        for term, count in term_counts.items():
            if term in vocab_idx:
                col_idx = vocab_idx[term]
                tf = count / doc_len
                matrix[row_idx, col_idx] = tf * idf[col_idx]

        # L2 normalize row
        norm = np.linalg.norm(matrix[row_idx])
        if norm > 1e-8:
            matrix[row_idx] /= norm

    return matrix, vocab


def _agglomerative_cluster(
    tfidf_matrix: np.ndarray, distance_threshold: float = DEFAULT_CLUSTER_DISTANCE_THRESHOLD
) -> list[list[int]]:
    """Cluster rows of tfidf_matrix using average-linkage agglomerative clustering."""
    num_samples = tfidf_matrix.shape[0]
    if num_samples == 0:
        return []
    if num_samples == 1:
        return [[0]]

    # Cosine distance matrix: dist = 1 - dot_product (rows are already L2 normalized)
    similarity = np.clip(np.dot(tfidf_matrix, tfidf_matrix.T), 0.0, 1.0)
    dist_matrix = 1.0 - similarity

    # Initialize each sample in its own cluster
    clusters: list[list[int]] = [[i] for i in range(num_samples)]

    while len(clusters) > 1:
        min_dist = float("inf")
        merge_i, merge_j = -1, -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c_i, c_j = clusters[i], clusters[j]
                # Average distance between clusters c_i and c_j
                pair_dists = [dist_matrix[x, y] for x in c_i for y in c_j]
                avg_dist = sum(pair_dists) / len(pair_dists)

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        if min_dist <= distance_threshold and merge_i != -1:
            # Merge clusters
            merged = clusters[merge_i] + clusters[merge_j]
            clusters = [c for idx, c in enumerate(clusters) if idx not in (merge_i, merge_j)]
            clusters.append(merged)
        else:
            break

    return clusters


def _extract_cluster_label(
    aspect: str,
    polarity: str,
    member_indices: list[int],
    clauses: list[ClauseAnalysisResult],
    tfidf_matrix: np.ndarray,
    vocab: list[str],
) -> str:
    """Extract an explainable descriptive label for an issue cluster."""
    if not member_indices:
        return f"{aspect} {polarity}"

    # Sum TF-IDF scores for members of this cluster, excluding character n-grams
    if tfidf_matrix.shape[1] > 0 and vocab:
        cluster_tfidf = np.sum(tfidf_matrix[member_indices], axis=0)
        word_scores: list[tuple[float, str]] = []
        for col_idx, term in enumerate(vocab):
            # Prefer natural words over character n-gram tokens or stopwords
            if not term.startswith("c:") and "_" not in term and term not in STOPWORDS and len(term) >= 3:
                word_scores.append((float(cluster_tfidf[col_idx]), term))

        word_scores.sort(reverse=True)
        top_terms = [t for _, t in word_scores[:2]]
        if top_terms:
            terms_str = " ".join(top_terms)
            return f"{aspect}: {terms_str}"

    # Fallback to shortest representative clause snippet
    shortest_clause = min((clauses[i].text for i in member_indices), key=len)
    clean_snippet = " ".join(shortest_clause.split()[:4])
    return f"{aspect}: {clean_snippet}" if clean_snippet else f"{aspect} issue"


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


def cluster_clauses(
    reviews: list[ReviewAnalysisResult],
    *,
    distance_threshold: float = DEFAULT_CLUSTER_DISTANCE_THRESHOLD,
) -> list[IssueCluster]:
    """Group review clauses into explainable issue clusters.

    - Clusters are formed separately per aspect and polarity (PRD Section 7.5).
    - Preserves links to supporting review IDs and clause evidence.
    - Aggregates aspect, sentiment, safety, frequency, and rating data.
    """
    if not reviews:
        return []

    routing_map = _load_routing_config()

    # Create mapping from review_id to review object for metadata lookup
    review_by_id: dict[str, ReviewAnalysisResult] = {r.review_id: r for r in reviews}

    # Group all non-duplicate clauses by (aspect, polarity)
    # Exclude empty reviews
    aspect_polarity_groups: dict[tuple[str, str], list[tuple[str, ClauseAnalysisResult]]] = defaultdict(list)

    for rev in reviews:
        if rev.status == "empty":
            continue
        for c in rev.clauses:
            asp = c.aspect or "general"
            pol = c.polarity
            aspect_polarity_groups[(asp, pol)].append((rev.review_id, c))

    clusters_out: list[IssueCluster] = []
    cluster_counter = 1

    # Tier rank for highest safety lookup
    tier_rank = {"none": 0, "medium": 1, "high": 2, "critical": 3}

    for (aspect, polarity), group_items in sorted(aspect_polarity_groups.items()):
        if not group_items:
            continue

        clause_objs = [item[1] for item in group_items]
        rev_ids = [item[0] for item in group_items]

        # Vectorize clauses
        tokenized = [_tokenize_clause(c.text) for c in clause_objs]
        tfidf_mat, vocab = _compute_tfidf_matrix(tokenized)

        # Agglomerative clustering
        sub_clusters = _agglomerative_cluster(tfidf_mat, distance_threshold=distance_threshold)

        for member_idxs in sub_clusters:
            if not member_idxs:
                continue

            member_clauses = [clause_objs[i] for i in member_idxs]
            member_rev_ids = [rev_ids[i] for i in member_idxs]
            unique_rev_ids = sorted(list(set(member_rev_ids)))

            # Find medoid (clause with highest average similarity to other cluster members)
            if len(member_idxs) == 1:
                medoid_local_idx = member_idxs[0]
            else:
                sub_sim = np.dot(tfidf_mat[member_idxs], tfidf_mat[member_idxs].T)
                avg_sims = np.mean(sub_sim, axis=1)
                best_within_idx = int(np.argmax(avg_sims))
                medoid_local_idx = member_idxs[best_within_idx]

            medoid_clause = clause_objs[medoid_local_idx]
            example_clause_id = medoid_clause.clause_id
            example_quote = medoid_clause.text

            # Label
            label = _extract_cluster_label(aspect, polarity, member_idxs, clause_objs, tfidf_mat, vocab)

            # Safety tier & ratings
            highest_tier = "none"
            ratings: list[float] = []
            sentiment_scores: list[float] = []
            evidence_list: list[IssueClusterClauseEvidence] = []

            for rev_id, c in zip(member_rev_ids, member_clauses):
                rev = review_by_id.get(rev_id)
                tier = rev.safety.tier if rev else "none"
                if tier_rank.get(tier, 0) > tier_rank.get(highest_tier, 0):
                    highest_tier = tier

                if rev and rev.rating is not None:
                    ratings.append(rev.rating)
                elif rev and rev.pseudo_rating is not None:
                    ratings.append(rev.pseudo_rating)

                sentiment_scores.append(c.score)
                evidence_list.append(
                    IssueClusterClauseEvidence(
                        review_id=rev_id,
                        clause_id=c.clause_id,
                        text=c.text,
                        score=c.score,
                        polarity=c.polarity,
                        safety_tier=tier,
                    )
                )

            avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
            avg_sent = round(sum(sentiment_scores) / len(sentiment_scores), 3) if sentiment_scores else 0.0

            # Routing
            route = routing_map.get(aspect.lower(), {})
            team = "QA/Compliance" if highest_tier in {"critical", "high"} else route.get("team", "Product/Engineering")
            priority = "P0" if highest_tier in {"critical", "high"} else route.get("priority", "P1")

            c_id = f"cluster_{aspect}_{polarity}_{cluster_counter}"
            cluster_counter += 1

            clusters_out.append(
                IssueCluster(
                    cluster_id=c_id,
                    aspect=aspect,
                    polarity=polarity,
                    label=label,
                    example_clause_id=example_clause_id,
                    example_quote=example_quote,
                    size=len(member_clauses),
                    review_count=len(unique_rev_ids),
                    clause_ids=[c.clause_id for c in member_clauses],
                    review_ids=unique_rev_ids,
                    evidence=evidence_list,
                    safety_tier=highest_tier,
                    is_safety_alert=(highest_tier != "none"),
                    average_rating=avg_rating,
                    average_sentiment=avg_sent,
                    team=team,
                    priority=priority,
                )
            )

    # Sort clusters: safety alerts first, then largest clusters first
    clusters_out.sort(
        key=lambda cl: (tier_rank.get(cl.safety_tier, 0), cl.size),
        reverse=True,
    )
    return clusters_out
