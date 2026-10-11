"""Lite NLP core: normalize, language tag, clause split, aspects, sentiment, safety, analyze.

Rule-based and CPU-only. No model downloads. Review text is never logged here.
"""

from pipeline.analyze import analyze, analyze_reviews, analyze_single_review

__all__ = ["analyze", "analyze_reviews", "analyze_single_review"]
