"""Automated unit and integration tests for ML dataset, training, evaluation, and inference."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml.dataset import (
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    CLASS_POSITIVE,
    clean_review_text,
    generate_curated_sample_fixture,
    rating_to_sentiment,
)
from ml.model_service import ModelArtifactNotFoundError, SentimentModelService
from ml.train import train_and_evaluate


def test_rating_to_sentiment():
    """Verify star rating to proxy sentiment mapping."""
    assert rating_to_sentiment(1.0) == CLASS_NEGATIVE
    assert rating_to_sentiment(2.0) == CLASS_NEGATIVE
    assert rating_to_sentiment(3.0) == CLASS_NEUTRAL
    assert rating_to_sentiment(4.0) == CLASS_POSITIVE
    assert rating_to_sentiment(5.0) == CLASS_POSITIVE
    assert rating_to_sentiment(None) is None
    assert rating_to_sentiment("invalid") is None
    assert rating_to_sentiment(0.5) is None


def test_clean_review_text():
    """Verify title and body combining and whitespace cleaning."""
    assert clean_review_text("Great", "Loved it") == "Great - Loved it"
    assert clean_review_text("Great", None) == "Great"
    assert clean_review_text(None, "Loved it") == "Loved it"
    assert clean_review_text("  Lots   of    spaces ", "  here ") == "Lots of spaces - here"
    assert clean_review_text("", "") == ""


def test_generate_curated_sample_fixture():
    """Verify curated fixture generates valid balanced DataFrame."""
    df = generate_curated_sample_fixture()
    assert len(df) == 30
    assert set(df["sentiment"].unique()) == {CLASS_NEGATIVE, CLASS_NEUTRAL, CLASS_POSITIVE}
    assert (df["label_id"] >= 0).all() and (df["label_id"] <= 2).all()


def test_train_and_evaluate_pipeline():
    """Test full training pipeline with curated fixture in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fixture_df = generate_curated_sample_fixture()

        results = train_and_evaluate(
            data_source=fixture_df,
            artifacts_dir=tmp_path,
            random_state=42,
        )

        assert "dataset" in results
        assert "model" in results
        assert "test_results" in results
        assert "per_class_metrics" in results
        assert "confusion_matrix" in results
        assert "model_comparison" in results

        assert (tmp_path / "sentiment_model.joblib").exists()
        assert (tmp_path / "evaluation_metrics.json").exists()
        assert (tmp_path / "EVALUATION_REPORT.md").exists()

        assert results["test_results"]["accuracy"] > 0.0
        assert len(results["model_comparison"]) >= 2
        assert "neutral_class_analysis" in results
        assert "binary_benchmark" in results


def test_model_service_predict_and_metrics():
    """Test model service predictions and metrics retrieval."""
    service = SentimentModelService.get_instance()
    assert service.is_available() is True

    # 1. Positive review prediction
    res_pos = service.predict("This machine is fantastic and works like a dream. Highly recommended!")
    assert res_pos["sentiment"] == CLASS_POSITIVE
    assert res_pos["polarity"] == "pos"
    assert res_pos["confidence"] > 0.5
    assert CLASS_POSITIVE in res_pos["probabilities"]
    assert res_pos["latency_ms"] >= 0.0

    # 2. Negative review prediction
    res_neg = service.predict("Defective item. Broke immediately upon first use and motor burned out.")
    assert res_neg["sentiment"] == CLASS_NEGATIVE
    assert res_neg["polarity"] == "neg"
    assert res_neg["confidence"] > 0.5
    assert CLASS_NEGATIVE in res_neg["probabilities"]

    # 3. Validation: empty input rejection
    with pytest.raises(ValueError, match="cannot be empty"):
        service.predict("   ")

    # 4. Metrics retrieval
    metrics = service.get_metrics()
    assert "dataset" in metrics
    assert "test_results" in metrics
    assert metrics["test_results"]["accuracy"] > 0.5


def test_model_service_missing_artifact():
    """Verify diagnostic error when model artifact does not exist."""
    fake_service = SentimentModelService(
        model_path=Path("/nonexistent/model.joblib"),
        metrics_path=Path("/nonexistent/metrics.json"),
    )
    assert fake_service.is_available() is False
    with pytest.raises(ModelArtifactNotFoundError):
        fake_service.load_model()


def test_api_ml_endpoints_authenticated(client: TestClient):
    """Test /api/ml/metrics and /api/ml/predict with authenticated client."""
    # 1. GET /api/ml/metrics
    resp_metrics = client.get("/api/ml/metrics")
    assert resp_metrics.status_code == 200
    data_metrics = resp_metrics.json()
    assert data_metrics["dataset"]["name"] == "Amazon Reviews 2023"
    assert "test_results" in data_metrics
    assert "per_class_metrics" in data_metrics
    assert "confusion_matrix" in data_metrics
    assert len(data_metrics["model_comparison"]) >= 2

    # 2. POST /api/ml/predict
    resp_pred = client.post(
        "/api/ml/predict",
        json={"text": "This is the best appliance ever, five stars! Highly recommended and wonderful."},
    )
    assert resp_pred.status_code == 200
    data_pred = resp_pred.json()
    assert data_pred["sentiment"] == "positive"
    assert data_pred["polarity"] == "pos"
    assert "probabilities" in data_pred
    assert "latency_ms" in data_pred

    # 3. POST /api/ml/predict empty text validation error
    resp_empty = client.post("/api/ml/predict", json={"text": "   "})
    assert resp_empty.status_code in (400, 422)
