from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.model_service import get_model


async def require_model(model_id: UUID, db: AsyncSession = Depends(get_db)) -> None:
    if await get_model(db=db, model_id=model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
