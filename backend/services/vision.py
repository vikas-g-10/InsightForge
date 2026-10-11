"""Image-based defect evidence service with Ollama vision inference provider."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from schemas import ImageDefectAnalysis, VisionHealthResponse
from settings import (
    DEFAULT_ALLOWED_IMAGE_MIMES,
    DEFAULT_MAX_IMAGE_UPLOAD_BYTES,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    Settings,
)

logger = logging.getLogger("insightforge.vision")

# Known vision-capable model identifiers / families
KNOWN_VISION_MODEL_PATTERNS = (
    "gemma3",
    "llava",
    "bakllava",
    "mllama",
    "vision",
    "minicpm",
    "qwen2-vl",
    "qwen2.5-vl",
    "moondream",
    "deepseek-vl",
    "cogvlm",
)

# Known text-only model identifiers (cannot accept images)
KNOWN_TEXT_ONLY_PATTERNS = (
    "llama3:",
    "llama3.",
    "llama2",
    "mistral",
    "deepseek-r1",
    "deepseek-llm",
    "phi3:",
    "phi4",
    "gemma:2b",
    "gemma:7b",
    "gemma2",
    "codellama",
    "starcoder",
    "vicuna",
)


class VisionError(Exception):
    """Base exception for vision service operations."""

    def __init__(self, message: str, code: str = "vision_error", status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class InvalidImageError(VisionError):
    def __init__(self, message: str, code: str = "invalid_image", status_code: int = 400) -> None:
        super().__init__(message, code=code, status_code=status_code)


class ImageTooLargeError(VisionError):
    def __init__(self, message: str, code: str = "payload_too_large", status_code: int = 413) -> None:
        super().__init__(message, code=code, status_code=status_code)


class OllamaUnavailableError(VisionError):
    def __init__(self, message: str, code: str = "ollama_unavailable", status_code: int = 503) -> None:
        super().__init__(message, code=code, status_code=status_code)


class OllamaTimeoutError(VisionError):
    def __init__(self, message: str, code: str = "ollama_timeout", status_code: int = 504) -> None:
        super().__init__(message, code=code, status_code=status_code)


class OllamaModelNotFoundError(VisionError):
    def __init__(self, message: str, code: str = "model_not_installed", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


class OllamaVisionUnsupportedError(VisionError):
    def __init__(self, message: str, code: str = "vision_unsupported", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


def detect_image_mime(image_bytes: bytes) -> str | None:
    """Detect image MIME type from magic bytes."""
    if not image_bytes or len(image_bytes) < 8:
        return None
    # JPEG magic bytes: FF D8 FF
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # WebP magic bytes: RIFF .... WEBP
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def validate_image_bytes(
    image_bytes: bytes,
    filename: str,
    declared_mime: str | None = None,
    max_bytes: int = DEFAULT_MAX_IMAGE_UPLOAD_BYTES,
    allowed_mimes: tuple[str, ...] = DEFAULT_ALLOWED_IMAGE_MIMES,
) -> tuple[str, str]:
    """Validate image file size, magic bytes, and MIME type safely.

    Returns:
        tuple[detected_mime, safe_filename]
    """
    if not image_bytes or len(image_bytes) == 0:
        raise InvalidImageError("Image file cannot be empty.", code="empty_image", status_code=400)

    if len(image_bytes) > max_bytes:
        raise ImageTooLargeError(
            f"Image file size ({len(image_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes.",
            code="payload_too_large",
            status_code=413,
        )

    detected_mime = detect_image_mime(image_bytes)
    if not detected_mime:
        raise InvalidImageError(
            "The uploaded file does not contain valid image contents. Allowed types: JPEG, PNG, WebP.",
            code="invalid_image_format",
            status_code=415,
        )

    if detected_mime not in allowed_mimes:
        raise InvalidImageError(
            f"Image type '{detected_mime}' is not supported. Allowed types: {', '.join(allowed_mimes)}.",
            code="unsupported_media_type",
            status_code=415,
        )

    if declared_mime:
        # Standardize declared mime (strip charset or extra parameters)
        clean_declared = declared_mime.split(";")[0].strip().lower()
        if clean_declared in ("image/jpg", "image/pjpeg"):
            clean_declared = "image/jpeg"
        if clean_declared != detected_mime and clean_declared != "application/octet-stream":
            raise InvalidImageError(
                f"Declared MIME type '{declared_mime}' does not match actual image content '{detected_mime}'.",
                code="mime_mismatch",
                status_code=400,
            )

    safe_name = filename.strip() if filename else f"image_{hashlib.sha256(image_bytes[:64]).hexdigest()[:8]}"
    return detected_mime, safe_name


def is_vision_model(model_name: str, families: list[str] | None = None) -> bool:
    """Determine whether a model supports image/vision input."""
    name_lower = model_name.lower().strip()
    families_lower = [f.lower().strip() for f in (families or [])]

    # Check for known positive patterns in name or family
    for pattern in KNOWN_VISION_MODEL_PATTERNS:
        if pattern in name_lower or any(pattern in fam for fam in families_lower):
            return True

    # Check for known text-only patterns
    for pattern in KNOWN_TEXT_ONLY_PATTERNS:
        if name_lower.startswith(pattern) or pattern in name_lower:
            return False

    # Default heuristic: if model has 'vision' or 'vl' in name, it's vision
    return "vision" in name_lower or "-vl" in name_lower or "_vl" in name_lower


class VisionProvider(ABC):
    """Abstract interface for image defect inference providers."""

    @abstractmethod
    def check_health(self) -> VisionHealthResponse:
        """Verify provider availability and model capabilities."""

    @abstractmethod
    def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str,
        cluster_context: str | None = None,
        aspect_context: str | None = None,
    ) -> ImageDefectAnalysis:
        """Perform vision inference and return structured defect analysis."""


class OllamaVisionProvider(VisionProvider):
    """Ollama REST API vision inference provider."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _http_request(
        self,
        path: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request to Ollama with comprehensive error handling."""
        url = f"{self.base_url}{path}"
        data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_bytes = resp.read()
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if err.code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' was not found in Ollama (HTTP 404). Run: `ollama pull {self.model}`.",
                    code="model_not_installed",
                    status_code=422,
                )
            raise OllamaUnavailableError(
                f"Ollama server returned HTTP {err.code}: {err_body or err.reason}",
                code="ollama_error",
                status_code=502,
            )
        except (socket.timeout, TimeoutError) as err:
            raise OllamaTimeoutError(
                f"Timed out after {self.timeout_seconds}s connecting to Ollama at {self.base_url}.",
                code="ollama_timeout",
                status_code=504,
            )
        except urllib.error.URLError as err:
            if isinstance(err.reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(
                    f"Timed out after {self.timeout_seconds}s connecting to Ollama at {self.base_url}.",
                    code="ollama_timeout",
                    status_code=504,
                )
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}: {err.reason}. Ensure Ollama is running (`ollama serve`).",
                code="ollama_unavailable",
                status_code=503,
            )
        except Exception as err:
            raise OllamaUnavailableError(
                f"Unexpected error communicating with Ollama: {err}",
                code="ollama_error",
                status_code=500,
            )

    def check_health(self) -> VisionHealthResponse:
        """Verify if Ollama is running and whether the configured model supports vision."""
        try:
            tags_resp = self._http_request("/api/tags", method="GET")
        except OllamaTimeoutError as err:
            return VisionHealthResponse(
                status="unavailable",
                provider="ollama",
                base_url=self.base_url,
                model=self.model,
                vision_supported=False,
                message=err.message,
                available_models=[],
            )
        except OllamaUnavailableError as err:
            return VisionHealthResponse(
                status="unavailable",
                provider="ollama",
                base_url=self.base_url,
                model=self.model,
                vision_supported=False,
                message=err.message,
                available_models=[],
            )

        models = tags_resp.get("models", [])
        installed_names: list[str] = [m.get("name", "") for m in models if m.get("name")]

        target_base = self.model.split(":")[0]
        matching_model: dict[str, Any] | None = None

        for m in models:
            name = m.get("name", "")
            if name == self.model or name.startswith(f"{self.model}:") or name == target_base:
                matching_model = m
                break

        if not matching_model and self.model not in installed_names:
            return VisionHealthResponse(
                status="model_not_installed",
                provider="ollama",
                base_url=self.base_url,
                model=self.model,
                vision_supported=False,
                message=f"Model '{self.model}' is not installed in Ollama. Install it via: `ollama pull {self.model}`.",
                available_models=installed_names,
            )

        # Inspect model details
        families: list[str] = []
        if matching_model:
            details = matching_model.get("details", {})
            families = details.get("families", [])
            fam = details.get("family")
            if fam and fam not in families:
                families.append(fam)

        if not is_vision_model(self.model, families):
            return VisionHealthResponse(
                status="vision_unsupported",
                provider="ollama",
                base_url=self.base_url,
                model=self.model,
                vision_supported=False,
                message=f"Model '{self.model}' is a text-only model and does not support image input. Please use 'gemma3:4b' or another vision-capable model.",
                available_models=installed_names,
            )

        return VisionHealthResponse(
            status="ready",
            provider="ollama",
            base_url=self.base_url,
            model=self.model,
            vision_supported=True,
            message=f"Ollama vision inference ready with model '{self.model}'.",
            available_models=installed_names,
        )

    def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str,
        cluster_context: str | None = None,
        aspect_context: str | None = None,
    ) -> ImageDefectAnalysis:
        """Call Ollama /api/generate with image and strictly-grounded prompt."""
        # 1. Check health / capabilities first
        health = self.check_health()
        if health.status == "unavailable":
            raise OllamaUnavailableError(health.message, code="ollama_unavailable", status_code=503)
        if health.status == "model_not_installed":
            raise OllamaModelNotFoundError(health.message, code="model_not_installed", status_code=422)
        if health.status == "vision_unsupported":
            raise OllamaVisionUnsupportedError(health.message, code="vision_unsupported", status_code=422)

        # 2. Encode image to Base64
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        image_id = f"img_{hashlib.sha256(image_bytes).hexdigest()[:12]}"
        thumbnail_data_url = f"data:{mime_type};base64,{b64_image}"

        # 3. Construct prompt enforcing strict empirical observation
        prompt = self._build_prompt(cluster_context, aspect_context)

        # 4. Make Ollama API call
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        }

        gen_resp = self._http_request("/api/generate", method="POST", payload=payload)
        raw_text = gen_resp.get("response", "").strip()

        # 5. Parse and structure the response
        return self._parse_model_response(
            raw_text=raw_text,
            image_id=image_id,
            filename=filename,
            mime_type=mime_type,
            file_size_bytes=len(image_bytes),
            cluster_context=cluster_context,
            aspect_context=aspect_context,
            thumbnail_data_url=thumbnail_data_url,
        )

    def _build_prompt(self, cluster_context: str | None, aspect_context: str | None) -> str:
        context_str = ""
        if cluster_context or aspect_context:
            context_str = f"\nRelevant review context: issue='{cluster_context or 'N/A'}', aspect='{aspect_context or 'N/A'}'."

        return (
            "You are an objective physical quality inspection specialist analyzing product photos for defects.\n"
            "STRICT FACTUAL GUIDELINES:\n"
            "1. Describe ONLY visible product characteristics and physical phenomena directly seen in the image.\n"
            "2. Distinguish observable evidence from speculation. NEVER invent measurements, internal circuit failures, or hidden causes.\n"
            "3. If no physical defect is visibly evident, set potential_defect_category to 'none_detected' and severity to 'none'.\n"
            "4. Clearly list observational limitations (e.g., lighting, angle, internal components not visible).\n"
            f"{context_str}\n\n"
            "You MUST reply with ONLY a single valid JSON object strictly matching this schema:\n"
            "{\n"
            '  "visual_description": "<concise description of visible product and condition>",\n'
            '  "potential_defect_category": "<none_detected | cosmetic_scratch | casing_crack | misalignment | thermal_damage | loose_part | surface_blemish | connector_damage | other>",\n'
            '  "visual_observations": ["<exact observable detail 1>", "<exact observable detail 2>"],\n'
            '  "severity": "<none | low | medium | high | critical>",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "relevant_aspect": "<aspect like build_quality, hardware, packaging, or null>",\n'
            '  "limitations": ["<observable limitation 1>"],\n'
            '  "suggested_manual_checks": ["<physical inspection step for human inspector>"]\n'
            "}"
        )

    def _parse_model_response(
        self,
        raw_text: str,
        image_id: str,
        filename: str,
        mime_type: str,
        file_size_bytes: int,
        cluster_context: str | None,
        aspect_context: str | None,
        thumbnail_data_url: str,
    ) -> ImageDefectAnalysis:
        # Strip markdown formatting if present
        clean_json = raw_text
        if "```" in clean_json:
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json, flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE)
        clean_json = clean_json.strip()

        data: dict[str, Any] = {}
        parse_warning = None

        try:
            data = json.loads(clean_json)
        except Exception:
            # Fallback: attempt to find JSON substring
            match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    pass

        if not data:
            # Safe graceful fallback for malformed output
            logger.warning("Could not parse JSON from Ollama response: %s", raw_text[:200])
            parse_warning = "Vision model output was not valid JSON; raw text preserved."
            visual_desc = raw_text[:300].strip() or "Image analyzed; descriptive summary unformatted."
            observations = [raw_text[:200]] if raw_text else ["Visual inspection completed."]
            category = "unclassified"
            severity = "low"
            confidence = 0.35
            limitations = ["Model output was not in expected JSON format.", "Manual visual verification required."]
            suggested_checks = ["Inspect physical unit under direct illumination."]
            relevant_aspect = aspect_context
        else:
            visual_desc = str(data.get("visual_description") or "Visual observation recorded.")
            category = str(data.get("potential_defect_category") or "none_detected")
            raw_obs = data.get("visual_observations") or []
            observations = [str(o) for o in raw_obs] if isinstance(raw_obs, list) else [str(raw_obs)]
            if not observations:
                observations = [visual_desc]

            raw_sev = str(data.get("severity") or "none").lower()
            severity = raw_sev if raw_sev in ("none", "low", "medium", "high", "critical") else "low"

            try:
                conf = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                confidence = 0.5

            rel_aspect = data.get("relevant_aspect")
            relevant_aspect = str(rel_aspect) if rel_aspect else aspect_context

            raw_lim = data.get("limitations") or []
            limitations = [str(l) for l in raw_lim] if isinstance(raw_lim, list) else [str(raw_lim)]
            if not limitations:
                limitations = ["Visual analysis limited to camera perspective and available resolution."]

            raw_checks = data.get("suggested_manual_checks") or []
            suggested_checks = [str(c) for c in raw_checks] if isinstance(raw_checks, list) else [str(raw_checks)]
            if not suggested_checks:
                suggested_checks = ["Perform direct physical examination of the item."]

        if parse_warning and parse_warning not in limitations:
            limitations.append(parse_warning)

        is_defect = (
            severity != "none"
            and category.lower() not in ("none_detected", "none", "normal_appearance", "no_defect")
        )

        return ImageDefectAnalysis(
            image_id=image_id,
            filename=filename,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            visual_description=visual_desc,
            potential_defect_category=category,
            visual_observations=observations,
            severity=severity,  # type: ignore[arg-type]
            confidence=round(confidence, 2),
            relevant_aspect=relevant_aspect,
            relevant_issue_cluster=cluster_context,
            limitations=limitations,
            suggested_manual_checks=suggested_checks,
            is_defect_observed=is_defect,
            thumbnail_data_url=thumbnail_data_url,
            raw_model_response=raw_text if parse_warning else None,
        )


class VisionService:
    """High-level service coordinating image validation and vision provider inference."""

    def __init__(self, settings: Settings, provider: VisionProvider | None = None) -> None:
        self.settings = settings
        self.provider = provider or OllamaVisionProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    def check_health(self) -> VisionHealthResponse:
        """Check provider status and configured model."""
        return self.provider.check_health()

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        declared_mime: str | None = None,
        cluster_context: str | None = None,
        aspect_context: str | None = None,
    ) -> ImageDefectAnalysis:
        """Validate input image and perform defect inference safely."""
        detected_mime, safe_filename = validate_image_bytes(
            image_bytes=image_bytes,
            filename=filename,
            declared_mime=declared_mime,
            max_bytes=self.settings.max_image_upload_bytes,
            allowed_mimes=self.settings.allowed_image_mimes,
        )
        return self.provider.analyze(
            image_bytes=image_bytes,
            filename=safe_filename,
            mime_type=detected_mime,
            cluster_context=cluster_context,
            aspect_context=aspect_context,
        )
