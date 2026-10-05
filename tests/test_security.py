import re

import pytest
from fastapi.testclient import TestClient

from code.app import main


@pytest.fixture
def client(monkeypatch):
    # Auth off: these tests are about the hardening around every response, not the PIN gate.
    monkeypatch.setattr(main, "APP_PIN_HASH", None)
    return TestClient(main.app)


def test_csp_nonce_matches_every_script_on_the_page(client):
    r = client.get("/login")
    csp = r.headers["content-security-policy"]
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    scripts = re.findall(r"<script[^>]*>", r.text)
    assert scripts and all(f'nonce="{nonce}"' in s for s in scripts)


def test_nonce_changes_per_request(client):
    a = client.get("/login").headers["content-security-policy"]
    b = client.get("/login").headers["content-security-policy"]
    assert a != b


def test_hardening_headers_present(client):
    h = client.get("/login").headers
    assert h["x-frame-options"] == "DENY"
    assert h["x-content-type-options"] == "nosniff"
    assert "camera=()" in h["permissions-policy"]


def test_cross_site_post_is_blocked(client):
    r = client.post("/api/delete-closed-trades", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_same_origin_post_is_allowed(client, monkeypatch):
    monkeypatch.setattr(main, "delete_all_closed", lambda: {"status": "deleted", "count": 0})
    r = client.post("/api/delete-closed-trades", headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_cross_site_fetch_metadata_is_blocked_without_origin(client):
    r = client.post("/api/delete-closed-trades", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_api_docs_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_headline_links_only_render_http_schemes():
    html = main.TEMPLATES.get_template("_headlines.html").render(
        headlines=[{"title": "x", "link": "javascript:alert(1)", "source": "s", "ts": ""}]
    )
    assert "javascript:" not in html
