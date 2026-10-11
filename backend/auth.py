"""Supabase Authentication and JWT Verification Module for InsightForge."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from jwt import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    PyJWKClient,
    PyJWKClientError,
    PyJWTError,
)

from settings import Settings

logger = logging.getLogger("insightforge.auth")

# Cache JWKS clients by base URL to avoid re-initializing on each request
_JWKS_CLIENTS: dict[str, PyJWKClient] = {}


@dataclass(frozen=True)
class AuthenticatedUser:
    """Represents a verified Supabase authenticated user identity."""

    user_id: str
    email: str | None
    role: str
    claims: dict[str, Any]


def get_jwks_client(supabase_url: str) -> PyJWKClient:
    """Return a cached PyJWKClient for the given Supabase URL."""
    clean_url = supabase_url.rstrip("/")
    jwks_url = f"{clean_url}/auth/v1/.well-known/jwks.json"
    if jwks_url not in _JWKS_CLIENTS:
        _JWKS_CLIENTS[jwks_url] = PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=3600)
    return _JWKS_CLIENTS[jwks_url]


def verify_token(token: str, settings: Settings) -> AuthenticatedUser:
    """Verify and decode a Supabase JWT access token.

    Validates signature, expiration, audience, issuer, and algorithm.
    Supports asymmetric RS256/ES256 via Supabase JWKS as well as legacy HS256 secret.
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        header = jwt.get_unverified_header(token)
    except DecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    alg = header.get("alg")
    if not alg or alg.lower() == "none":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Algorithm 'none' is strictly forbidden.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    decoded: dict[str, Any] | None = None
    expected_audience = settings.supabase_jwt_audience or "authenticated"
    expected_issuer = f"{settings.supabase_url.rstrip('/')}/auth/v1" if settings.supabase_url else None

    # Asymmetric signing (RS256, ES256, etc.) using Supabase JWKS
    if alg in ("RS256", "ES256", "RS384", "ES384", "RS512", "ES512"):
        if not settings.supabase_url:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Server is not configured with SUPABASE_URL for asymmetric token verification.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            jwks_client = get_jwks_client(settings.supabase_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            decode_kwargs: dict[str, Any] = {
                "key": signing_key.key,
                "algorithms": [alg],
                "audience": expected_audience,
                "options": {"verify_exp": True, "verify_aud": True},
            }
            if expected_issuer:
                decode_kwargs["issuer"] = expected_issuer
                decode_kwargs["options"]["verify_iss"] = True

            decoded = jwt.decode(token, **decode_kwargs)
        except PyJWKClientError as e:
            logger.warning("JWKS key retrieval failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to retrieve signing key for token verification.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Symmetric HS256 signing using SUPABASE_JWT_SECRET
    elif alg == "HS256":
        if not settings.supabase_jwt_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Server is not configured with SUPABASE_JWT_SECRET for symmetric token verification.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        decode_kwargs = {
            "key": settings.supabase_jwt_secret,
            "algorithms": ["HS256"],
            "audience": expected_audience,
            "options": {"verify_exp": True, "verify_aud": True},
        }
        if expected_issuer:
            decode_kwargs["issuer"] = expected_issuer
            decode_kwargs["options"]["verify_iss"] = True

        decoded = jwt.decode(token, **decode_kwargs)

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unsupported signing algorithm '{alg}'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate essential claims
    user_id = decoded.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' user identifier claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(
        user_id=str(user_id),
        email=decoded.get("email"),
        role=decoded.get("role", "authenticated"),
        claims=decoded,
    )


def extract_bearer_token(authorization: str | None = Header(default=None)) -> str:
    """Extract Bearer token from Authorization header or raise 401."""
    if not authorization:
        logger.warning("Auth failure: Missing Authorization header on incoming request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning("Auth failure: Invalid Authorization header format.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must follow 'Bearer <token>' format.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return parts[1]


async def get_current_user(
    request: Request,
    token: str = Depends(extract_bearer_token),
) -> AuthenticatedUser:
    """FastAPI dependency that extracts and verifies Supabase Bearer token."""
    settings: Settings = getattr(request.app.state, "settings", None) or Settings.from_env()
    try:
        return verify_token(token, settings)
    except ExpiredSignatureError:
        logger.warning("Auth failure: Authentication token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidAudienceError:
        logger.warning("Auth failure: Token audience claim does not match expected audience.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token audience claim does not match expected audience.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidIssuerError as e:
        logger.warning("Auth failure: Token issuer claim does not match expected issuer: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token issuer claim does not match expected issuer.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidSignatureError as e:
        logger.warning("Auth failure: Token signature verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signature verification failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (DecodeError, InvalidAlgorithmError, ImmatureSignatureError) as e:
        logger.warning("Auth failure: Invalid authentication token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error during token verification: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication verification failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )
