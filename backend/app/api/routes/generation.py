from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.api import deps
from app.tasks import generation_tasks

router = APIRouter()


@router.post("/pages/{page_id}/optimize", response_model=schemas.OptimizedPage)
async def optimize_page(
    *,
    page_id: UUID,
    optimization_in: schemas.OptimizationCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Generate optimized version of a page based on analysis and recommendations.
    """
    # Check page exists and user has access
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Check if page has been analyzed
    analysis = await crud.page_analysis.get_by_page_id(session, page_id=page_id)
    if not analysis or analysis.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Page must be completely analyzed before optimization",
        )

    # Check if page already has an optimized version
    existing_optimized = await crud.optimized_page.get_by_page_id(
        session, page_id=page_id
    )

    if existing_optimized:
        # Update existing optimized page status
        optimized_page = await crud.optimized_page.update(
            session,
            db_obj=existing_optimized,
            obj_in={
                "status": "draft",
                "applied_recommendations": optimization_in.recommendation_ids,
            },
        )
    else:
        # Create new optimized page record
        optimized_page = await crud.optimized_page.create(
            session,
            obj_in=schemas.OptimizedPageCreate(
                page_id=page_id,
                status="draft",
                applied_recommendations=optimization_in.recommendation_ids,
            ),
        )

    # Start optimization in background
    background_tasks.add_task(
        generation_tasks.generate_optimized_content,
        str(page_id),
        optimization_in.recommendation_ids,
    )

    return optimized_page


@router.get("/pages/{page_id}/optimized", response_model=schemas.OptimizedPage)
async def get_optimized_page(
    page_id: UUID,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get optimized version of a page.
    """
    # Check page exists and user has access
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Get optimized page
    optimized_page = await crud.optimized_page.get_by_page_id(session, page_id=page_id)
    if not optimized_page:
        raise HTTPException(
            status_code=404, detail="Optimized version not found for this page"
        )

    return optimized_page


@router.put("/optimized/{optimized_id}/status", response_model=schemas.OptimizedPage)
async def update_optimized_page_status(
    *,
    optimized_id: UUID,
    status_update: schemas.OptimizedPageStatusUpdate,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Update status of an optimized page (publish or reject).
    """
    optimized_page = await crud.optimized_page.get(session, id=optimized_id)
    if not optimized_page:
        raise HTTPException(status_code=404, detail="Optimized page not found")

    # Check if user has access
    page = await crud.page.get(session, id=optimized_page.page_id)
    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Update status
    optimized_page = await crud.optimized_page.update(
        session, db_obj=optimized_page, obj_in={"status": status_update.status}
    )

    # If publishing, update page status to optimized
    if status_update.status == "published":
        await crud.page.update(session, db_obj=page, obj_in={"status": "optimized"})

    return optimized_page
