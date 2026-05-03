"""M106: API-key and Bearer-JWT authentication dependency."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from server.auth.jwt_auth import verify_jwt
from server.auth.rate_limit import check_rate_limit

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def _get_allowed_keys() -> set[str]:
    raw = os.environ.get("QUANTX_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def auth_enabled() -> bool:
    """Return ``True`` when ``QUANTX_API_KEYS`` is configured."""
    return bool(_get_allowed_keys())


def _mask_key(key: str) -> str:
    """Return a masked representation for logging and whoami responses."""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


async def require_api_key(
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> dict:
    """FastAPI dependency: validate ``X-API-Key`` header or ``Bearer`` JWT.

    When ``QUANTX_API_KEYS`` is not set the server runs in *open-access* mode
    and every request is treated as an anonymous admin — useful for local
    development and CI without exposing real credentials.

    When ``QUANTX_API_KEYS`` is set every request must carry either a
    recognised API key in the ``X-API-Key`` header or a valid JWT in the
    ``Authorization: Bearer …`` header.  Rate-limiting (sliding-window, 60 s)
    is applied after successful authentication.

    Returns
    -------
    dict
        ``{"sub": <masked_identity>, "scopes": [<scope>, ...]}``
    """
    allowed = _get_allowed_keys()

    if not allowed:
        # Auth disabled — open-access mode
        return {
            "sub": "anonymous",
            "scopes": ["solver:read", "solver:write", "jobs:submit", "admin"],
        }

    identity: dict | None = None

    # 1. Try API-key header
    if api_key and api_key in allowed:
        identity = {
            "sub": _mask_key(api_key),
            "scopes": ["solver:read", "solver:write", "jobs:submit"],
        }

    # 2. Fall back to Bearer JWT
    if identity is None and bearer is not None:
        payload = verify_jwt(bearer.credentials)
        if payload is not None:
            identity = {"sub": payload["sub"], "scopes": payload.get("scopes", [])}

    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": 'ApiKey realm="QuantX", Bearer realm="QuantX"'},
        )

    # 3. Rate limiting
    allowed_req, retry_after = await check_rate_limit(identity["sub"])
    if not allowed_req:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    return identity
