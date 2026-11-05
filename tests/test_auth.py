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

    assert (await async_client.get("/health")).status_code == 200
    assert (await async_client.get("/ready")).status_code == 200


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
