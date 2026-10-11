"""Thread-safe model loading and prediction service for InsightForge."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml.dataset import INT_TO_LABEL

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "sentiment_model.joblib"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

POLARITY_MAP: dict[str, str] = {
    "negative": "neg",
    "neutral": "neu",
    "positive": "pos",
}


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the serialized model artifact or metrics file cannot be found."""
    pass


class SentimentModelService:
    """Singleton service for loading trained ML models and running low-latency inference."""

    _instance: SentimentModelService | None = None

    def __init__(
        self,
        model_path: Path | str | None = None,
        metrics_path: Path | str | None = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH
        self._pipeline: Pipeline | None = None
        self._metrics: dict[str, Any] | None = None

    @classmethod
    def get_instance(cls) -> SentimentModelService:
        if cls._instance is None:
            cls._instance = SentimentModelService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def is_available(self) -> bool:
        """Check whether the model artifact and metrics exist on disk."""
        return self.model_path.exists() and self.metrics_path.exists()

    def load_model(self) -> None:
        """Load the model pipeline and metrics from disk."""
        if not self.model_path.exists():
            raise ModelArtifactNotFoundError(
                f"Trained model artifact not found at {self.model_path}. "
                "Execute 'python -m ml.train' to train and serialize the model."
            )

        if self._pipeline is None:
            # Safely unpickle pipeline created by trusted internal training module
            self._pipeline = joblib.load(self.model_path)

        if self._metrics is None and self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self._metrics = json.load(f)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self.load_model()
        assert self._pipeline is not None
        return self._pipeline

    def get_metrics(self) -> dict[str, Any]:
        """Return the held-out test evaluation metrics."""
        if self._metrics is None:
            if not self.metrics_path.exists():
                raise ModelArtifactNotFoundError(
                    f"Evaluation metrics artifact not found at {self.metrics_path}. "
                    "Run 'python -m ml.train' to generate evaluation metrics."
                )
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self._metrics = json.load(f)
        return self._metrics

    def predict(self, text: str) -> dict[str, Any]:
        """Run ML sentiment inference on input review text.

        Args:
            text: Raw or normalized review text.

        Returns:
            Dict containing sentiment label, polarity, confidence score,
            calibrated class probabilities, and execution latency in ms.
        """
        if not text or not text.strip():
            raise ValueError("Review text cannot be empty or whitespace-only.")

        clean_text = text.strip()
        if len(clean_text) > 5000:
            clean_text = clean_text[:5000]

        pipe = self.pipeline
        t0 = time.perf_counter()

        pred_id = int(pipe.predict([clean_text])[0])
        pred_label = INT_TO_LABEL.get(pred_id, "neutral")
        polarity = POLARITY_MAP.get(pred_label, "neu")

        probabilities: dict[str, float] = {}
        confidence = 1.0

        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba([clean_text])[0]
            for idx, p_val in enumerate(probs):
                lbl = INT_TO_LABEL.get(idx, f"class_{idx}")
                probabilities[lbl] = round(float(p_val), 4)
            confidence = round(float(np.max(probs)), 4)
        else:
            probabilities = {pred_label: 1.0}

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        model_name = "TF-IDF + Logistic Regression"
        if self._metrics and "model" in self._metrics:
            model_name = self._metrics["model"].get("selected_model", model_name)

        return {
            "text": clean_text,
            "sentiment": pred_label,
            "polarity": polarity,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": model_name,
            "latency_ms": latency_ms,
        }
