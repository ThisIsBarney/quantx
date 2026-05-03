"""M106: Auth router — /auth/token and /auth/whoami."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from server.auth.api_key import require_api_key, _get_allowed_keys, _mask_key
from server.auth.jwt_auth import create_jwt

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def get_token(body: TokenRequest) -> TokenResponse:
    """Exchange an API key for a signed JWT access token.

    This endpoint is always open (no ``X-API-Key`` required) so that clients
    can bootstrap their Bearer token.  In *open-access* mode (no
    ``QUANTX_API_KEYS`` configured) any non-empty key string is accepted.
    """
    allowed = _get_allowed_keys()
    if allowed and body.api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    sub = _mask_key(body.api_key) if body.api_key else "anonymous"
    scopes = (
        ["solver:read", "solver:write", "jobs:submit"]
        if allowed
        else ["solver:read", "solver:write", "jobs:submit", "admin"]
    )
    token = create_jwt(sub=sub, scopes=scopes)
    return TokenResponse(access_token=token)


@router.get("/whoami")
async def whoami(identity: dict = Depends(require_api_key)) -> dict:
    """Return the authenticated identity and granted scopes.

    Requires a valid ``X-API-Key`` header or ``Authorization: Bearer`` token
    when auth is enabled.
    """
    return {"sub": identity.get("sub"), "scopes": identity.get("scopes", [])}
