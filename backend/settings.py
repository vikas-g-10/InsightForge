"""Environment-based configuration for the InsightForge backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

VALID_MODES = ("lite", "full")
DEFAULT_MODE = "lite"
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_ROWS = 2000
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_ALLOWED_IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
DEFAULT_SUPABASE_URL = ""
DEFAULT_SUPABASE_JWT_SECRET = ""
DEFAULT_SUPABASE_JWT_AUDIENCE = "authenticated"
DEFAULT_AUTH_ENABLED = True


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    mode: str = DEFAULT_MODE
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_rows: int = DEFAULT_MAX_ROWS
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    max_image_upload_bytes: int = DEFAULT_MAX_IMAGE_UPLOAD_BYTES
    allowed_image_mimes: tuple[str, ...] = DEFAULT_ALLOWED_IMAGE_MIMES
    supabase_url: str = DEFAULT_SUPABASE_URL
    supabase_jwt_secret: str = DEFAULT_SUPABASE_JWT_SECRET
    supabase_jwt_audience: str = DEFAULT_SUPABASE_JWT_AUDIENCE
    auth_enabled: bool = DEFAULT_AUTH_ENABLED

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        if environ is None:
            backend_env = Path(__file__).resolve().parent / ".env"
            if backend_env.exists():
                load_dotenv(backend_env, override=False)
            root_env = Path(__file__).resolve().parent.parent / ".env"
            if root_env.exists():
                load_dotenv(root_env, override=False)
            env = os.environ
        else:
            env = environ
        return cls(
            mode=_parse_mode(env.get("APP_MODE")),
            allowed_origins=_parse_origins(env.get("ALLOWED_ORIGINS")),
            max_upload_bytes=_parse_positive_int(
                "MAX_UPLOAD_BYTES", env.get("MAX_UPLOAD_BYTES"), DEFAULT_MAX_UPLOAD_BYTES
            ),
            max_rows=_parse_positive_int("MAX_ROWS", env.get("MAX_ROWS"), DEFAULT_MAX_ROWS),
            ollama_base_url=_parse_ollama_url(env.get("OLLAMA_BASE_URL")),
            ollama_model=(env.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL,
            ollama_timeout_seconds=_parse_positive_float(
                "OLLAMA_TIMEOUT_SECONDS", env.get("OLLAMA_TIMEOUT_SECONDS"), DEFAULT_OLLAMA_TIMEOUT_SECONDS
            ),
            max_image_upload_bytes=_parse_positive_int(
                "MAX_IMAGE_UPLOAD_BYTES", env.get("MAX_IMAGE_UPLOAD_BYTES"), DEFAULT_MAX_IMAGE_UPLOAD_BYTES
            ),
            allowed_image_mimes=DEFAULT_ALLOWED_IMAGE_MIMES,
            supabase_url=(env.get("SUPABASE_URL") or "").strip().rstrip("/"),
            supabase_jwt_secret=(env.get("SUPABASE_JWT_SECRET") or "").strip(),
            supabase_jwt_audience=(env.get("SUPABASE_JWT_AUDIENCE") or DEFAULT_SUPABASE_JWT_AUDIENCE).strip(),
            auth_enabled=_parse_bool("AUTH_ENABLED", env.get("AUTH_ENABLED"), DEFAULT_AUTH_ENABLED),
        )


def _parse_mode(raw: str | None) -> str:
    if raw is None or not raw.strip():
        return DEFAULT_MODE
    mode = raw.strip().lower()
    if mode not in VALID_MODES:
        raise ConfigError(
            f"Unsupported APP_MODE {raw!r}. Use one of: {', '.join(VALID_MODES)}."
        )
    return mode


def _parse_positive_int(name: str, raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a positive integer.") from None
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _parse_origins(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_ALLOWED_ORIGINS
    origins = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        parsed = urlparse(origin)
        if (
            origin == "*"
            or parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.path
        ):
            raise ConfigError(
                f"Invalid ALLOWED_ORIGINS entry {item.strip()!r}. Use explicit "
                "origins such as http://localhost:5173 (no wildcards, no paths)."
            )
        origins.append(origin)
    if not origins:
        raise ConfigError("ALLOWED_ORIGINS is set but contains no origins.")
    return tuple(origins)


def _parse_ollama_url(raw: str | None) -> str:
    if raw is None or not raw.strip():
        return DEFAULT_OLLAMA_BASE_URL
    url = raw.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"Invalid OLLAMA_BASE_URL {raw!r}. Must be an http or https URL.")
    return url


def _parse_positive_float(name: str, raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a positive number.") from None
    if val <= 0:
        raise ConfigError(f"{name} must be a positive number.")
    return val


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    raise ConfigError(f"{name} must be a boolean ('true' or 'false').")

