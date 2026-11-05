import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Reject requests without the configured key; an empty setting disables auth."""
    expected = settings.api_key
    # Compare bytes: compare_digest raises TypeError on non-ASCII str input.
    if expected and not secrets.compare_digest((key or "").encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
