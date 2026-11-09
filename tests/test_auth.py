import asyncio
import logging

import pytest
from fastapi.routing import APIRoute
from httpx import AsyncClient

from app.config import settings
from app.main import app

PUBLIC_PATHS = {"/health", "/ready"}
KEY = "correct-key"
# The path fails UUID validation after auth, so 422 proves the request got past auth
# without needing a database.
PAST_AUTH_PATH = "/models/not-a-uuid"


@pytest.fixture
def key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", KEY)


def _api_routes() -> list[tuple[str, str]]:
    return [
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    ]


@pytest.mark.asyncio
async def test_every_route_but_the_probes_requires_the_key(
    async_client: AsyncClient, key_required: None
) -> None:
    routes = [r for r in _api_routes() if r[1] not in PUBLIC_PATHS]
    assert len(routes) > 10

    for method, path in routes:
        response = await async_client.request(method, path)
        assert response.status_code == 401, (method, path)
        assert response.headers["WWW-Authenticate"] == "ApiKey"


@pytest.mark.asyncio
async def test_probes_stay_open(
    async_client: AsyncClient,
    key_required: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ready() -> None:
        return None

    monkeypatch.setattr("app.main.check_database_readiness", ready)

    for path in (
        "/health",
        "/ready",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    ):
        assert (await async_client.get(path)).status_code == 200


@pytest.mark.asyncio
async def test_missing_key_rejects_bad_json_and_logs(
    async_client: AsyncClient, key_required: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="driftwatch"):
        response = await async_client.post(
            "/ingest/snapshot",
            content=b"{",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    assert len([r for r in caplog.records if "status_code=401" in r.message]) == 1


@pytest.mark.asyncio
async def test_missing_key_does_not_consume_body(key_required: None) -> None:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ingest/snapshot",
        "raw_path": b"/ingest/snapshot",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    body_consumed = False
    response_complete = asyncio.Event()
    messages = []

    async def receive():
        nonlocal body_consumed
        if not body_consumed:
            body_consumed = True
            return {"type": "http.request", "body": b"{", "more_body": False}
        await response_complete.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            response_complete.set()

    await app(scope, receive, send)

    assert (
        next(m for m in messages if m["type"] == "http.response.start")["status"] == 401
    )
    assert not body_consumed


@pytest.mark.asyncio
async def test_correct_key_is_accepted(
    async_client: AsyncClient, key_required: None
) -> None:
    response = await async_client.get(PAST_AUTH_PATH, headers={"X-API-Key": KEY})

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrong", [b"wrong", b"", KEY.encode() + b"x", "é".encode("latin-1")]
)
async def test_wrong_key_is_rejected(
    async_client: AsyncClient, key_required: None, wrong: bytes
) -> None:
    response = await async_client.get(PAST_AUTH_PATH, headers={"X-API-Key": wrong})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unset_key_disables_auth(async_client: AsyncClient) -> None:
    assert (await async_client.get(PAST_AUTH_PATH)).status_code == 422
    assert (
        await async_client.get(PAST_AUTH_PATH, headers={"X-API-Key": KEY})
    ).status_code == 422
