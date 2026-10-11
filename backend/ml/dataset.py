"""Dataset acquisition, cleaning, and preparation pipeline for InsightForge ML.

Dataset: Amazon Reviews 2023 (McAuley-Lab/Amazon-Reviews-2023)
Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
Citation:
  Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).
  Bridging Language and Items for Retrieval and Recommendation.
  arXiv preprint arXiv:2403.03952.

License & Usage:
  The dataset is released under research and educational terms by UC San Diego
  (McAuley Lab). Star ratings (1.0 to 5.0) serve as ground-truth proxy labels:
    - 1.0 to 2.0 stars -> 'negative' (class 0)
    - 3.0 stars        -> 'neutral'  (class 1)
    - 4.0 to 5.0 stars -> 'positive' (class 2)

Important Note on Proxy Labels:
  Star ratings reflect user satisfaction at the transaction/product level.
  While strongly correlated with sentiment, star ratings are proxy labels and
  may occasionally contain label noise (e.g., sarcastic positive text, or a 1-star
  rating caused by shipping delay rather than product defect).
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd
import requests

DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "Appliances"
HF_BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"

CLASS_NEGATIVE = "negative"
CLASS_NEUTRAL = "neutral"
CLASS_POSITIVE = "positive"

CLASS_LABELS: tuple[str, ...] = (CLASS_NEGATIVE, CLASS_NEUTRAL, CLASS_POSITIVE)

LABEL_TO_INT: dict[str, int] = {
    CLASS_NEGATIVE: 0,
    CLASS_NEUTRAL: 1,
    CLASS_POSITIVE: 2,
}

INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


def rating_to_sentiment(rating: float | int | None) -> str | None:
    """Map 1.0-5.0 numeric star rating to 3-class sentiment label."""
    if rating is None:
        return None
    try:
        r = float(rating)
    except (ValueError, TypeError):
        return None

    if 1.0 <= r <= 2.0:
        return CLASS_NEGATIVE
    elif 2.5 <= r <= 3.5:
        return CLASS_NEUTRAL
    elif 4.0 <= r <= 5.0:
        return CLASS_POSITIVE
    return None


def clean_review_text(title: str | None, body: str | None) -> str:
    """Combine title and body text, normalizing whitespace and removing control chars."""
    parts = []
    if title and isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if body and isinstance(body, str) and body.strip():
        parts.append(body.strip())

    combined = " - ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    # Normalize excess whitespace
    cleaned = re.sub(r"\s+", " ", combined).strip()
    return cleaned


def stream_amazon_reviews(
    category: str = DEFAULT_CATEGORY,
    timeout: float = 30.0,
) -> Generator[dict, None, None]:
    """Stream raw review records line-by-line from Hugging Face without storing massive raw files."""
    url = f"{HF_BASE_URL}/{category}.jsonl"
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            record = json.loads(line.decode("utf-8"))
            yield record
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def acquire_dataset(
    category: str = DEFAULT_CATEGORY,
    samples_per_class: int = 12000,
    max_total: int = 36000,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Download and clean a balanced sample of Amazon Reviews 2023.

    Args:
        category: Subcategory name (e.g. 'Appliances', 'Software', 'All_Beauty').
        samples_per_class: Target quota per sentiment class to combat class imbalance.
        max_total: Hard limit on total collected records.
        output_path: Optional path to save cleaned CSV.

    Returns:
        pd.DataFrame with columns: [review_id, text, rating, sentiment, label_id]
    """
    records_by_class: dict[str, list[dict]] = {
        CLASS_NEGATIVE: [],
        CLASS_NEUTRAL: [],
        CLASS_POSITIVE: [],
    }
    seen_texts: set[str] = set()

    for raw in stream_amazon_reviews(category=category):
        rating = raw.get("rating")
        sentiment = rating_to_sentiment(rating)
        if not sentiment:
            continue

        if len(records_by_class[sentiment]) >= samples_per_class:
            # Check if all quotas reached
            if all(len(v) >= samples_per_class for v in records_by_class.values()):
                break
            continue

        text = clean_review_text(raw.get("title"), raw.get("text"))
        # Quality filter: skip reviews shorter than 12 characters or longer than 2000 chars
        if len(text) < 12 or len(text) > 2000:
            continue

        # Deduplication check
        norm_key = text.lower()[:120]
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        records_by_class[sentiment].append({
            "review_id": raw.get("parent_asin", f"rev_{len(seen_texts)}"),
            "text": text,
            "rating": float(rating),
            "sentiment": sentiment,
            "label_id": LABEL_TO_INT[sentiment],
        })

        total_collected = sum(len(v) for v in records_by_class.values())
        if total_collected >= max_total:
            break

    # Combine into single DataFrame
    combined = []
    for cls in CLASS_LABELS:
        combined.extend(records_by_class[cls])

    df = pd.DataFrame(combined)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def generate_curated_sample_fixture(output_path: Path | str | None = None) -> pd.DataFrame:
    """Generate a clean curated benchmark dataset for automated offline unit testing."""
    sample_data: list[dict[str, str | float | int]] = [
        # Negative reviews (Class 0)
        {"text": "Broke after only two uses. The plastic handle snapped completely.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Terrible quality and very noisy motor. Defective unit, returning immediately.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Battery drains in less than two hours. Completely unusable for daily work.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Stopped working within a week. Customer support refused to issue a replacement.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Poor build quality and very cheap materials. Very disappointed with this purchase.", "rating": 2.0, "sentiment": "negative", "label_id": 0},
        {"text": "Does not fit properly and constantly leaks water. Waste of money.", "rating": 2.0, "sentiment": "negative", "label_id": 0},
        {"text": "Very loud vibrations and overheats rapidly. Would not recommend to anyone.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Flimsy construction and parts arrived missing from the box.", "rating": 2.0, "sentiment": "negative", "label_id": 0},
        {"text": "The motor burned out during normal operation. Hazardous smell of smoke.", "rating": 1.0, "sentiment": "negative", "label_id": 0},
        {"text": "Software crashes constantly and Bluetooth connection drops every few minutes.", "rating": 2.0, "sentiment": "negative", "label_id": 0},

        # Neutral reviews (Class 1)
        {"text": "It works as expected, nothing extraordinary but gets the basic job done.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Average product. Build is decent for the price, but could be sturdier.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Acceptable performance. Takes a while to warm up, but operates fine once running.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Mediocre experience. It is okay for light household use, but not for heavy tasks.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Standard replacement filter. Neither impressed nor unhappy with the outcome.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "The product matches the description. Quality is okay, delivery was on time.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Fair value. Sound is average and cord length is a bit short.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Works moderately well. A bit bulky on the counter but functional.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Not great but not terrible either. Does what it claims at an entry level price.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},
        {"text": "Basic functionality is fine. Instruction manual was somewhat unclear.", "rating": 3.0, "sentiment": "neutral", "label_id": 1},

        # Positive reviews (Class 2)
        {"text": "Excellent quality and outstanding performance! Very satisfied with this machine.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Best purchase I made this year. Whisper quiet, powerful, and easy to clean.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Extremely well constructed and durable. Worth every penny spent.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Super fast shipping and works flawlessly. Exceeded my high expectations.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Incredible battery life and seamless setup. Highly recommend to everyone.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Sturdy metal construction, intuitive buttons, and reliable performance every day.", "rating": 4.0, "sentiment": "positive", "label_id": 2},
        {"text": "Very pleased with the build quality and finish. Looks fantastic in my kitchen.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Works like a charm. Compact, powerful, and very well designed.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
        {"text": "Great value for money. Operates smoothly without any hiccups or noise.", "rating": 4.0, "sentiment": "positive", "label_id": 2},
        {"text": "Top notch engineering and crystal clear display. A five star product.", "rating": 5.0, "sentiment": "positive", "label_id": 2},
    ]

    df = pd.DataFrame(sample_data)
    df["review_id"] = [f"sample_rev_{i+1:03d}" for i in range(len(df))]
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Acquire and prepare Amazon Reviews 2023 dataset.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Hugging Face category")
    parser.add_argument("--samples-per-class", type=int, default=10000, help="Target quota per sentiment class")
    parser.add_argument("--output", default="data/amazon_reviews_clean.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Streaming and preparing Amazon Reviews 2023 ({args.category})...")
    df = acquire_dataset(
        category=args.category,
        samples_per_class=args.samples_per_class,
        output_path=args.output,
    )
    print(f"Acquisition complete! Total records: {len(df)}")
    print(f"Class breakdown:\n{df['sentiment'].value_counts()}")
