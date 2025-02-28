from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app import crud, models, schemas
from app.api import deps
from app.tasks import parsing_tasks

# Create router without prefix since we'll have nested routes
router = APIRouter(tags=["Pages"])


@router.get("/projects/{project_id}/pages", response_model=List[schemas.Page])
async def read_pages(
        project_id: UUID,
        skip: int = 0,
        limit: int = 100,
        session: AsyncSession = Depends(deps.get_db),
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve pages for a specific project.
    """
    try:
        # Check if project exists and user has access
        project = await crud.project.get(session, id=project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if project.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        pages = await crud.page.get_multi_by_project(
            session, project_id=project_id, skip=skip, limit=limit
        )
        return pages
    except Exception as e:
        logger.error(f"Error in read_pages: {str(e)}")
        # Better error handling with detailed error message
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/projects/{project_id}/pages",
    response_model=schemas.Page,
    status_code=status.HTTP_201_CREATED,
)
async def create_page(
        *,
        project_id: UUID,
        page_in: schemas.PageCreate,
        background_tasks: BackgroundTasks,
        session: AsyncSession = Depends(deps.get_db),
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Create a new page and start parsing process.
    """
    try:
        # Check if project exists and user has access
        project = await crud.project.get(session, id=project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if project.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        # Convert URL to string for database compatibility
        url_str = str(page_in.url)
        logger.debug(f"Processing URL: {url_str} for project: {project_id}")

        # Check if URL already exists in this project
        existing_page = await crud.page.get_by_url_and_project(
            session, url=url_str, project_id=project_id
        )
        if existing_page:
            raise HTTPException(
                status_code=400, detail="This URL is already being analyzed in this project"
            )

        # Create page with explicit project_id
        page_data = page_in.dict()
        page_data["project_id"] = project_id  # Ensure project_id is set
        page_data["url"] = url_str  # Use string version of URL

        page = await crud.page.create(
            session,
            obj_in=schemas.PageCreate(**page_data),
        )

        # Commit before starting background task to ensure the database has the record
        await session.commit()

        # Instead of using background tasks directly, use Celery task
        # This avoids the event loop issue
        parsing_tasks.parse_page.delay(str(page.id))

        return page
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        logger.error(f"Error in create_page: {str(e)}")
        # Better error handling with detailed error message
        raise HTTPException(
            status_code=500,
            detail=f"Error creating page: {str(e)}"
        )


@router.post("/pages/{page_id}/reparse", response_model=schemas.Page)
async def reparse_page(
        *,
        page_id: UUID,
        background_tasks: BackgroundTasks,
        session: AsyncSession = Depends(deps.get_db),
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Restart parsing for a page.
    """
    try:
        page = await crud.page.get(session, id=page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        # Check if user has access to the project
        project = await crud.project.get(session, id=page.project_id)
        if project.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        # Update page status and increment retry counter
        page = await crud.page.update(
            session,
            db_obj=page,
            obj_in={"status": "pending", "retry_count": page.retry_count + 1},
        )

        # Commit before starting Celery task
        await session.commit()

        # Use Celery task directly instead of background task
        parsing_tasks.parse_page.delay(str(page.id))

        return page
    except Exception as e:
        logger.error(f"Error in reparse_page: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error reparsing page: {str(e)}")


@router.get("/pages/{page_id}", response_model=schemas.Page)
async def read_page(
        page_id: UUID,
        session: AsyncSession = Depends(deps.get_db),
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get page by ID.
    """
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    # Check if user has access to the project
    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return page