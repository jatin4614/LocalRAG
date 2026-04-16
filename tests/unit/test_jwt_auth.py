import time
import jwt as pyjwt
import pytest
from ext.services.jwt_verifier import verify_upstream_jwt, JWTError


SECRET = "t0p-s3cr3t"
ALGO   = "HS256"


def _mint(payload: dict, secret: str = SECRET) -> str:
    return pyjwt.encode(payload, secret, algorithm=ALGO)


def test_valid_token_returns_id():
    tok = _mint({"id": "abc-123", "jti": "xyz"})
    claims = verify_upstream_jwt(tok, secret=SECRET)
    assert claims["id"] == "abc-123"


def test_expired_token_raises():
    tok = _mint({"id": "abc", "exp": int(time.time()) - 60})
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_bad_signature_raises():
    tok = _mint({"id": "abc"}, secret="wrong-secret")
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_missing_id_raises():
    tok = _mint({"jti": "x"})
    with pytest.raises(JWTError):
        verify_upstream_jwt(tok, secret=SECRET)


def test_malformed_token_raises():
    with pytest.raises(JWTError):
        verify_upstream_jwt("not-a-jwt", secret=SECRET)
