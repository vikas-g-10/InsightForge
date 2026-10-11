"""Unit and integration tests for Ollama vision inference service."""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import create_app
from services.vision import (
    OllamaVisionProvider,
    VisionService,
    detect_image_mime,
    is_vision_model,
    validate_image_bytes,
)
from settings import Settings

# Sample valid image byte fixtures
VALID_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xff\xdb\x00C\x00" + b"\x01" * 64 + b"\xff\xd9"
)
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
VALID_WEBP_BYTES = b"RIFF\x1c\x00\x00\x00WEBPVP8 \x10\x00\x00\x000\x01\x00\x9d\x01\x2a\x01\x00\x01\x00\x00"


def _mock_http_response(status: int = 200, body: dict | None = None) -> MagicMock:
    """Helper to mock urllib.request.urlopen returning JSON body."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body or {}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# =========================================================================
# 1. Byte Validation & MIME Detection Unit Tests
# =========================================================================

def test_detect_image_mime():
    assert detect_image_mime(VALID_JPEG_BYTES) == "image/jpeg"
    assert detect_image_mime(VALID_PNG_BYTES) == "image/png"
    assert detect_image_mime(VALID_WEBP_BYTES) == "image/webp"
    assert detect_image_mime(b"NOT_AN_IMAGE_AT_ALL") is None
    assert detect_image_mime(b"") is None


def test_validate_image_bytes_success():
    mime, name = validate_image_bytes(VALID_JPEG_BYTES, "test.jpg", "image/jpeg")
    assert mime == "image/jpeg"
    assert name == "test.jpg"


def test_validate_image_bytes_empty():
    with pytest.raises(Exception) as exc_info:
        validate_image_bytes(b"", "empty.jpg")
    assert "empty" in str(exc_info.value).lower()


def test_validate_image_bytes_corrupted_content():
    with pytest.raises(Exception) as exc_info:
        validate_image_bytes(b"HELLO_WORLD_TEXT_NOT_IMAGE", "fake.jpg", "image/jpeg")
    assert exc_info.value.code == "invalid_image_format"


def test_validate_image_bytes_mime_mismatch():
    with pytest.raises(Exception) as exc_info:
        validate_image_bytes(VALID_JPEG_BYTES, "test.png", "image/png")
    assert exc_info.value.code == "mime_mismatch"


def test_validate_image_bytes_exceeds_max_limit():
    with pytest.raises(Exception) as exc_info:
        validate_image_bytes(VALID_JPEG_BYTES, "big.jpg", max_bytes=10)
    assert exc_info.value.code == "payload_too_large"


# =========================================================================
# 2. Model Vision Capability Identification
# =========================================================================

def test_is_vision_model():
    assert is_vision_model("gemma3:4b") is True
    assert is_vision_model("llava:13b") is True
    assert is_vision_model("qwen2.5-vl:7b") is True
    assert is_vision_model("custom-model", families=["gemma3"]) is True
    assert is_vision_model("llama3:8b") is False
    assert is_vision_model("mistral:7b") is False
    assert is_vision_model("deepseek-r1:8b") is False


# =========================================================================
# 3. Vision Provider Health Checks
# =========================================================================

@patch("urllib.request.urlopen")
def test_provider_health_ready(mock_urlopen):
    mock_urlopen.return_value = _mock_http_response(
        200,
        {
            "models": [
                {
                    "name": "gemma3:4b",
                    "details": {"family": "gemma3", "families": ["gemma3"]},
                }
            ]
        },
    )
    provider = OllamaVisionProvider(model="gemma3:4b")
    health = provider.check_health()
    assert health.status == "ready"
    assert health.vision_supported is True
    assert "ready" in health.message.lower()


@patch("urllib.request.urlopen")
def test_provider_health_unavailable(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    provider = OllamaVisionProvider()
    health = provider.check_health()
    assert health.status == "unavailable"
    assert health.vision_supported is False
    assert "unreachable" in health.message.lower() or "cannot connect" in health.message.lower()


@patch("urllib.request.urlopen")
def test_provider_health_timeout(mock_urlopen):
    mock_urlopen.side_effect = socket.timeout("Timed out")
    provider = OllamaVisionProvider()
    health = provider.check_health()
    assert health.status == "unavailable"
    assert health.vision_supported is False
    assert "timed out" in health.message.lower()


@patch("urllib.request.urlopen")
def test_provider_health_model_not_installed(mock_urlopen):
    mock_urlopen.return_value = _mock_http_response(
        200,
        {"models": [{"name": "mistral:7b"}, {"name": "llama3:8b"}]},
    )
    provider = OllamaVisionProvider(model="gemma3:4b")
    health = provider.check_health()
    assert health.status == "model_not_installed"
    assert health.vision_supported is False
    assert "ollama pull gemma3:4b" in health.message


@patch("urllib.request.urlopen")
def test_provider_health_vision_unsupported(mock_urlopen):
    mock_urlopen.return_value = _mock_http_response(
        200,
        {"models": [{"name": "llama3:8b", "details": {"family": "llama"}}]},
    )
    provider = OllamaVisionProvider(model="llama3:8b")
    health = provider.check_health()
    assert health.status == "vision_unsupported"
    assert health.vision_supported is False
    assert "does not support" in health.message.lower()


# =========================================================================
# 4. FastAPI Endpoints Integration Tests (Mocked Ollama Client)
# =========================================================================



@patch("urllib.request.urlopen")
def test_api_vision_health(mock_urlopen, client):
    mock_urlopen.return_value = _mock_http_response(
        200,
        {
            "models": [
                {
                    "name": "gemma3:4b",
                    "details": {"family": "gemma3", "families": ["gemma3"]},
                }
            ]
        },
    )
    resp = client.get("/api/vision/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["vision_supported"] is True


@patch("urllib.request.urlopen")
def test_api_vision_analyze_success(mock_urlopen, client):
    # First call: check_health (/api/tags)
    # Second call: generate (/api/generate)
    tags_mock = _mock_http_response(
        200,
        {
            "models": [
                {
                    "name": "gemma3:4b",
                    "details": {"family": "gemma3", "families": ["gemma3"]},
                }
            ]
        },
    )
    generate_mock = _mock_http_response(
        200,
        {
            "response": json.dumps(
                {
                    "visual_description": "Matte black plastic surface showing a longitudinal scratch near USB port.",
                    "potential_defect_category": "cosmetic_scratch",
                    "visual_observations": [
                        "Hairline scratch measuring approx 1.5cm along lower seam",
                        "No signs of deep fracture or internal component exposure",
                    ],
                    "severity": "low",
                    "confidence": 0.88,
                    "relevant_aspect": "build_quality",
                    "limitations": ["Lighting creates reflections on edge bevel"],
                    "suggested_manual_checks": ["Tactile inspection with fingertip to test depth"],
                }
            )
        },
    )
    mock_urlopen.side_effect = [tags_mock, generate_mock]

    files = {"file": ("defect.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")}
    data = {"cluster_context": "Scratched housing on arrival", "aspect_context": "build_quality"}

    resp = client.post("/api/vision/analyze", files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["image_id"].startswith("img_")
    assert result["filename"] == "defect.jpg"
    assert result["potential_defect_category"] == "cosmetic_scratch"
    assert result["severity"] == "low"
    assert result["confidence"] == 0.88
    assert result["is_defect_observed"] is True
    assert result["relevant_issue_cluster"] == "Scratched housing on arrival"
    assert len(result["visual_observations"]) == 2
    assert "data:image/jpeg;base64," in result["thumbnail_data_url"]


@patch("urllib.request.urlopen")
def test_api_vision_analyze_markdown_wrapped_json(mock_urlopen, client):
    tags_mock = _mock_http_response(
        200,
        {"models": [{"name": "gemma3:4b", "details": {"family": "gemma3"}}]},
    )
    markdown_payload = (
        "```json\n"
        "{\n"
        '  "visual_description": "Clean front panel with no noticeable marks.",\n'
        '  "potential_defect_category": "none_detected",\n'
        '  "visual_observations": ["Smooth finish", "Even alignment"],\n'
        '  "severity": "none",\n'
        '  "confidence": 0.95,\n'
        '  "relevant_aspect": "cosmetics",\n'
        '  "limitations": ["Rear side not visible"],\n'
        '  "suggested_manual_checks": ["Inspect other angles"]\n'
        "}\n"
        "```"
    )
    generate_mock = _mock_http_response(200, {"response": markdown_payload})
    mock_urlopen.side_effect = [tags_mock, generate_mock]

    files = {"file": ("clean.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 200
    result = resp.json()
    assert result["potential_defect_category"] == "none_detected"
    assert result["severity"] == "none"
    assert result["is_defect_observed"] is False


@patch("urllib.request.urlopen")
def test_api_vision_analyze_malformed_json_fallback(mock_urlopen, client):
    tags_mock = _mock_http_response(
        200,
        {"models": [{"name": "gemma3:4b", "details": {"family": "gemma3"}}]},
    )
    generate_mock = _mock_http_response(
        200,
        {"response": "The image shows a crack near the right corner of the casing. It looks damaged."},
    )
    mock_urlopen.side_effect = [tags_mock, generate_mock]

    files = {"file": ("defect.webp", io.BytesIO(VALID_WEBP_BYTES), "image/webp")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 200
    result = resp.json()
    # Graceful fallback handled without crashing
    assert "crack near the right corner" in result["visual_description"]
    assert any("not valid JSON" in l for l in result["limitations"])


@patch("urllib.request.urlopen")
def test_api_vision_analyze_ollama_unavailable(mock_urlopen, client):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    files = {"file": ("test.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "ollama_unavailable"


@patch("urllib.request.urlopen")
def test_api_vision_analyze_ollama_timeout(mock_urlopen, client):
    tags_mock = _mock_http_response(
        200,
        {"models": [{"name": "gemma3:4b", "details": {"family": "gemma3"}}]},
    )
    mock_urlopen.side_effect = [tags_mock, socket.timeout("Read timed out")]

    files = {"file": ("test.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "ollama_timeout"


@patch("urllib.request.urlopen")
def test_api_vision_analyze_model_not_installed(mock_urlopen, client):
    tags_mock = _mock_http_response(
        200,
        {"models": [{"name": "other-model:latest"}]},
    )
    mock_urlopen.side_effect = [tags_mock]

    files = {"file": ("test.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model_not_installed"


def test_api_vision_analyze_invalid_file_bytes(client):
    files = {"file": ("fake.jpg", io.BytesIO(b"NOT_A_REAL_IMAGE"), "image/jpeg")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "invalid_image_format"


def test_api_vision_analyze_empty_file(client):
    files = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_image"


def test_api_vision_analyze_mime_mismatch(client):
    # Pass JPEG bytes with declared image/png header
    files = {"file": ("mismatch.png", io.BytesIO(VALID_JPEG_BYTES), "image/png")}
    resp = client.post("/api/vision/analyze", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "mime_mismatch"
