import jwt as pyjwt
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from ext.services.auth import CurrentUser, get_current_user, require_admin


SECRET = "test-secret-32-chars-or-whatever-x"


def _mk_app():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id, "role": u.role}

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True}

    return app


def test_stub_mode_still_works(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "stub")
    client = TestClient(_mk_app())
    r = client.get("/me", headers={"X-User-Id": "42", "X-User-Role": "user"})
    assert r.status_code == 200
    assert r.json() == {"id": 42, "role": "user"}


def test_jwt_mode_accepts_cookie(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: "user")

    tok = pyjwt.encode({"id": "100"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 200
    assert r.json() == {"id": 100, "role": "user"}


def test_jwt_mode_accepts_bearer(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: "admin")

    tok = pyjwt.encode({"id": "9"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/admin", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200


def test_jwt_mode_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    tok = pyjwt.encode({"id": "1"}, "wrong-secret", algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 401


def test_jwt_mode_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    client = TestClient(_mk_app())
    r = client.get("/me")
    assert r.status_code == 401


def test_jwt_mode_rejects_unknown_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    from ext.services import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_lookup_role_by_id", lambda uid: None)
    tok = pyjwt.encode({"id": "999"}, SECRET, algorithm="HS256")
    client = TestClient(_mk_app())
    r = client.get("/me", cookies={"token": tok})
    assert r.status_code == 401
