from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.model import ModelCreate, ModelListResponse, ModelResponse
from app.services.model_service import create_model, get_model, list_models

router = APIRouter(tags=["models"])


@router.post("", response_model=ModelResponse, status_code=201)
async def create_model_endpoint(
    payload: ModelCreate,
    db: AsyncSession = Depends(get_db),
) -> ModelResponse:
    """Create a new monitored model record."""
    try:
        return await create_model(db=db, name=payload.name, version=payload.version)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="Model name already exists"
        ) from exc


@router.get("", response_model=ModelListResponse)
async def list_models_endpoint(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> ModelListResponse:
    """List monitored models using pagination."""
    return await list_models(db=db, page=page, page_size=page_size)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ModelResponse:
    """Fetch a model by ID."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
