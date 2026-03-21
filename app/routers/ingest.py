from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.ingest import ReferenceListResponse, ReferencePayload, ReferenceResponse
from app.services.model_service import get_model
from app.services.reference_service import list_reference_features, register_reference

router = APIRouter(tags=["ingest"])


@router.post("/reference", response_model=ReferenceResponse)
async def register_reference_endpoint(
    payload: ReferencePayload,
    db: AsyncSession = Depends(get_db),
) -> ReferenceResponse:
    """Register reference distributions and optional feature importances."""
    model = await get_model(db=db, model_id=payload.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return await register_reference(db=db, payload=payload)


@router.get("/reference/{model_id}", response_model=ReferenceListResponse)
async def list_reference_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ReferenceListResponse:
    """List all registered reference features for a model."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return await list_reference_features(db=db, model_id=model_id)
