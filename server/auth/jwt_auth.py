"""M106: Minimal HS256 JWT implementation — stdlib only, no extra dependencies."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

_DEFAULT_TTL = 3600  # 1 hour


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    # Re-add stripped padding
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _get_secret() -> bytes:
    return os.environ.get(
        "QUANTX_JWT_SECRET", "quantx-dev-secret-change-in-production"
    ).encode()


def create_jwt(sub: str, scopes: list[str], ttl_seconds: int | None = None) -> str:
    """Create a signed HS256 JWT.

    Parameters
    ----------
    sub:        Subject (masked API-key identity or user name).
    scopes:     List of scope strings granted to this token.
    ttl_seconds: Token lifetime; defaults to ``QUANTX_JWT_TTL_SECONDS`` env var
                 or 3600 s when the env var is absent.
    """
    if ttl_seconds is None:
        ttl_seconds = int(os.environ.get("QUANTX_JWT_TTL_SECONDS", str(_DEFAULT_TTL)))
    now = int(time.time())
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps({"sub": sub, "scopes": scopes, "iat": now, "exp": now + ttl_seconds}).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    sig = hmac.new(_get_secret(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(sig)}"


def verify_jwt(token: str) -> dict[str, Any] | None:
    """Verify a JWT and return its payload dict, or *None* if invalid/expired."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    try:
        expected_sig = hmac.new(_get_secret(), signing_input, hashlib.sha256).digest()
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, actual_sig):
        return None
    try:
        payload: dict[str, Any] = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) < int(time.time()):
        return None
    return payload
