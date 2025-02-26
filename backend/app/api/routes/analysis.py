from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.api import deps
from app.tasks import analysis_tasks

router = APIRouter()


@router.get("/pages/{page_id}/analysis", response_model=schemas.PageAnalysis)
async def read_page_analysis(
    page_id: UUID,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get analysis for a specific page.
    """
    # Check page exists and user has access
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Get analysis
    analysis = await crud.page_analysis.get_by_page_id(session, page_id=page_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found for this page")

    return analysis


@router.post("/pages/{page_id}/analyze", response_model=schemas.PageAnalysis)
async def analyze_page(
    *,
    page_id: UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Start or restart analysis for a page.
    """
    # Check page exists and user has access
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Verify page has been parsed
    if page.status not in ["analyzed", "optimized"]:
        raise HTTPException(
            status_code=400,
            detail="Page must be successfully parsed before analysis",
            project=await crud.project.get(session, id=page.project_id),
        )

    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Check if page already has analysis
    existing_analysis = await crud.page_analysis.get_by_page_id(
        session, page_id=page_id
    )

    if existing_analysis:
        # Update existing analysis status to pending
        analysis = await crud.page_analysis.update(
            session, db_obj=existing_analysis, obj_in={"status": "pending"}
        )
    else:
        # Create new analysis record
        analysis = await crud.page_analysis.create(
            session,
            obj_in=schemas.PageAnalysisCreate(page_id=page_id, status="pending"),
        )

    # Start analysis in background
    background_tasks.add_task(analysis_tasks.analyze_page_content, str(page_id))

    return analysis


@router.get(
    "/pages/{page_id}/recommendations", response_model=List[schemas.PageRecommendation]
)
async def read_page_recommendations(
    page_id: UUID,
    category: Optional[str] = Query(
        None, description="Filter by recommendation category"
    ),
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get recommendations for a specific page.
    """
    # Check page exists and user has access
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Get analysis
    analysis = await crud.page_analysis.get_by_page_id(session, page_id=page_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found for this page")

    # Get recommendations
    if category:
        recommendations = (
            await crud.page_recommendation.get_by_analysis_id_and_category(
                session, analysis_id=analysis.id, category=category
            )
        )
    else:
        recommendations = await crud.page_recommendation.get_by_analysis_id(
            session, analysis_id=analysis.id
        )

    return recommendations
