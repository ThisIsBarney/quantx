"""M106: Tests for API-key auth, JWT, and rate limiting."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Import app — reuse the cached module (patched by test_api.py or conftest).
# Auth tests do not call any engine endpoints so we don't override the mock.
# ---------------------------------------------------------------------------

try:
    from server.main import app  # already imported and patched by another test module
except Exception:
    _mock_engine = MagicMock()
    _mock_engine.version.return_value = "0.0.0-test"
    _mock_engine.list_solvers.return_value = []
    with patch("quantx_engine.Engine", return_value=_mock_engine):
        from server.main import app
    app.state.engine = _mock_engine

# Reset the rate-limit windows between tests
from server.auth.rate_limit import reset_all as _reset_rate_limit

_JWT_SECRET = "test-secret-for-unit-tests"


@pytest.fixture(autouse=True)
def _clean_rate_limit():
    _reset_rate_limit()
    yield
    _reset_rate_limit()


def _reload_auth(monkeypatch):
    """Reload auth module so env-var changes take effect in the dependency."""
    import importlib
    import server.auth.api_key as _mod
    importlib.reload(_mod)


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_missing_key_returns_401(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami")
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami", headers={"X-API-Key": "sk-wrong"})
        assert r.status_code == 401

    def test_valid_key_returns_200(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami", headers={"X-API-Key": "sk-test-1"})
        assert r.status_code == 200
        body = r.json()
        assert "sub" in body
        assert "scopes" in body

    def test_open_access_when_no_keys_configured(self, monkeypatch):
        monkeypatch.delenv("QUANTX_API_KEYS", raising=False)
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami")
        assert r.status_code == 200
        assert r.json()["sub"] == "anonymous"

    def test_masked_identity_returned(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-abcdefgh-1234")
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami", headers={"X-API-Key": "sk-abcdefgh-1234"})
        assert r.status_code == 200
        sub = r.json()["sub"]
        # Raw key must not appear in the response
        assert "sk-abcdefgh-1234" not in sub
        assert "****" in sub

    def test_raw_key_not_in_rl_key(self, monkeypatch):
        """P2: _rl_key_for must not return the plaintext key."""
        from server.auth.api_key import _rl_key_for
        key = "sk-secret-key-value"
        rl = _rl_key_for(key)
        assert key not in rl
        # Should be a 64-char hex string (SHA-256)
        assert len(rl) == 64

    def test_two_distinct_keys_have_distinct_rl_buckets(self, monkeypatch):
        """P2: different API keys must get different rate-limit buckets."""
        from server.auth.api_key import _rl_key_for
        assert _rl_key_for("key-A") != _rl_key_for("key-B")
        # Short keys that mask to the same "****" must still differ
        assert _rl_key_for("abc") != _rl_key_for("xyz")


# ---------------------------------------------------------------------------
# JWT authentication
# ---------------------------------------------------------------------------

class TestJwtAuth:
    def test_token_endpoint_open(self, monkeypatch):
        """POST /auth/token is accessible without a prior API key header."""
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        monkeypatch.setenv("QUANTX_JWT_SECRET", _JWT_SECRET)
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/auth/token", json={"api_key": "sk-test-1"})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_invalid_key_on_token_endpoint_returns_401(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        # QUANTX_JWT_SECRET intentionally NOT set: 401 is raised before
        # create_jwt is called so the secret guard must not fire.
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/auth/token", json={"api_key": "sk-bad"})
        assert r.status_code == 401

    def test_jwt_bearer_grants_access(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        monkeypatch.setenv("QUANTX_JWT_SECRET", _JWT_SECRET)
        _reload_auth(monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)
        # Exchange API key for JWT
        token = client.post("/auth/token", json={"api_key": "sk-test-1"}).json()["access_token"]
        # Use JWT as Bearer
        r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "sub" in r.json()

    def test_expired_jwt_returns_401(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        monkeypatch.setenv("QUANTX_JWT_SECRET", _JWT_SECRET)
        _reload_auth(monkeypatch)
        from server.auth.jwt_auth import create_jwt
        # Issue a token that expired 10 seconds ago
        token = create_jwt(sub="test****", scopes=["solver:read"], ttl_seconds=-10)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_tampered_jwt_returns_401(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        monkeypatch.setenv("QUANTX_JWT_SECRET", _JWT_SECRET)
        _reload_auth(monkeypatch)
        import server.auth.jwt_auth as _jwt_mod
        token = _jwt_mod.create_jwt(sub="test****", scopes=["solver:read"])
        # Tamper the FIRST character of the signature segment (carries 6 full
        # data bits, guaranteed to change the decoded bytes and break the HMAC).
        # Avoid flipping the last character: for a 32-byte HMAC the last
        # base64url char only contributes 4 bits and its bottom 2 are zero-
        # padding, so 'A'↔'B' (which differ only in that padding bit) leaves
        # the decoded bytes identical and the signature still validates.
        parts = token.split(".")
        sig = parts[2]
        tampered_sig = ("B" if sig[0] != "B" else "C") + sig[1:]
        tampered = ".".join([parts[0], parts[1], tampered_sig])
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {tampered}"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# JWT unit tests (no HTTP)
# ---------------------------------------------------------------------------

class TestJwtUnit:
    def test_roundtrip(self):
        """Open-access mode: default secret is used, roundtrip succeeds."""
        from server.auth.jwt_auth import create_jwt, verify_jwt
        token = create_jwt(sub="alice", scopes=["solver:read"])
        payload = verify_jwt(token)
        assert payload is not None
        assert payload["sub"] == "alice"
        assert "solver:read" in payload["scopes"]

    def test_expiry(self):
        from server.auth.jwt_auth import create_jwt, verify_jwt
        token = create_jwt(sub="alice", scopes=[], ttl_seconds=-1)
        assert verify_jwt(token) is None

    def test_wrong_secret(self, monkeypatch):
        monkeypatch.setenv("QUANTX_JWT_SECRET", "secret-A")
        from server.auth.jwt_auth import create_jwt
        token = create_jwt(sub="alice", scopes=[])
        monkeypatch.setenv("QUANTX_JWT_SECRET", "secret-B")
        from server.auth.jwt_auth import verify_jwt
        assert verify_jwt(token) is None

    def test_malformed_token(self):
        from server.auth.jwt_auth import verify_jwt
        assert verify_jwt("not.a.valid.jwt.token.parts") is None
        assert verify_jwt("onlytwoparts") is None
        assert verify_jwt("") is None

    def test_p1_secret_required_when_auth_enabled(self, monkeypatch):
        """P1: _get_secret raises RuntimeError when auth enabled but secret missing."""
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-test-1")
        monkeypatch.delenv("QUANTX_JWT_SECRET", raising=False)
        # Test via the module reference so we bypass any caching issues
        import server.auth.jwt_auth as _jwt_mod
        # Verify env is in the expected state
        import os
        assert os.environ.get("QUANTX_API_KEYS") == "sk-test-1"
        assert os.environ.get("QUANTX_JWT_SECRET") is None
        with pytest.raises(RuntimeError, match="QUANTX_JWT_SECRET"):
            _jwt_mod._get_secret()
        with pytest.raises(RuntimeError, match="QUANTX_JWT_SECRET"):
            _jwt_mod.create_jwt(sub="test", scopes=[])

    def test_p1_secret_not_required_in_open_access_mode(self, monkeypatch):
        """P1: default secret is fine when auth is disabled (open-access)."""
        monkeypatch.delenv("QUANTX_API_KEYS", raising=False)
        monkeypatch.delenv("QUANTX_JWT_SECRET", raising=False)
        from server.auth.jwt_auth import create_jwt, verify_jwt
        token = create_jwt(sub="anon", scopes=[])
        payload = verify_jwt(token)
        assert payload is not None
        assert payload["sub"] == "anon"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    @pytest.mark.anyio
    async def test_exceeding_rpm_returns_429(self, monkeypatch):
        monkeypatch.setenv("QUANTX_API_KEYS", "sk-rl-test")
        monkeypatch.setenv("QUANTX_RATE_LIMIT_RPM", "3")
        _reload_auth(monkeypatch)
        _reset_rate_limit()
        client = TestClient(app, raise_server_exceptions=False)
        headers = {"X-API-Key": "sk-rl-test"}
        responses = [client.get("/auth/whoami", headers=headers) for _ in range(5)]
        status_codes = [r.status_code for r in responses]
        assert status_codes[:3] == [200, 200, 200]
        assert 429 in status_codes[3:]
        # 429 must carry a Retry-After header
        first_429 = next(r for r in responses if r.status_code == 429)
        assert "retry-after" in first_429.headers

    @pytest.mark.anyio
    async def test_rate_limit_unit(self, monkeypatch):
        monkeypatch.setenv("QUANTX_RATE_LIMIT_RPM", "2")
        from server.auth.rate_limit import check_rate_limit
        _reset_rate_limit()
        ok1, _ = await check_rate_limit("user-x")
        ok2, _ = await check_rate_limit("user-x")
        ok3, retry = await check_rate_limit("user-x")
        assert ok1 and ok2
        assert not ok3
        assert retry >= 1

    @pytest.mark.anyio
    async def test_rate_limit_disabled_when_rpm_zero(self, monkeypatch):
        monkeypatch.setenv("QUANTX_RATE_LIMIT_RPM", "0")
        from server.auth.rate_limit import check_rate_limit
        _reset_rate_limit()
        results = [await check_rate_limit("user-y") for _ in range(100)]
        assert all(ok for ok, _ in results)
