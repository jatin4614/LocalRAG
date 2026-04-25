from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from ext.services.auth import CurrentUser, get_current_user, require_admin


def test_missing_headers_returns_401():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id}

    r = TestClient(app).get("/me")
    assert r.status_code == 401


def test_valid_headers_parsed():
    app = FastAPI()

    @app.get("/me")
    def me(u: CurrentUser = Depends(get_current_user)):
        return {"id": u.id, "role": u.role}

    r = TestClient(app).get("/me", headers={"X-User-Id": "42", "X-User-Role": "user"})
    assert r.status_code == 200
    # CurrentUser.id is now str (UUIDs from upstream Open WebUI). Stub mode
    # passes the X-User-Id header through verbatim, so "42" stays "42".
    assert r.json() == {"id": "42", "role": "user"}


def test_non_admin_cannot_access_admin_route():
    app = FastAPI()

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True}

    r = TestClient(app).get("/admin", headers={"X-User-Id": "1", "X-User-Role": "user"})
    assert r.status_code == 403


def test_admin_can_access_admin_route():
    app = FastAPI()

    @app.get("/admin")
    def admin(u: CurrentUser = Depends(require_admin)):
        return {"ok": True, "role": u.role}

    r = TestClient(app).get("/admin", headers={"X-User-Id": "1", "X-User-Role": "admin"})
    assert r.status_code == 200
