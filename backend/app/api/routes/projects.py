from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.api import deps

# Define the router with explicit prefix
router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("", response_model=List[schemas.Project])
async def read_projects(
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve projects for the current user.
    """
    projects = await crud.project.get_multi_by_user(
        session, user_id=current_user.id, skip=skip, limit=limit
    )
    return projects


@router.post("", response_model=schemas.Project, status_code=status.HTTP_201_CREATED)
async def create_project(
    *,
    project_in: schemas.ProjectCreate,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Create a new project.
    """
    project = await crud.project.create_with_user(
        session, obj_in=project_in, user_id=current_user.id
    )
    return project


@router.get("/{project_id}", response_model=schemas.Project)
async def read_project(
    project_id: UUID,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get project by ID.
    """
    project = await crud.project.get(session, id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return project


@router.put("/{project_id}", response_model=schemas.Project)
async def update_project(
    *,
    project_id: UUID,
    project_in: schemas.ProjectUpdate,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Update a project.
    """
    project = await crud.project.get(session, id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    project = await crud.project.update(session, db_obj=project, obj_in=project_in)
    return project


@router.delete("/{project_id}", response_model=schemas.Project)
async def delete_project(
    *,
    project_id: UUID,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Delete a project.
    """
    project = await crud.project.get(session, id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    project = await crud.project.remove(session, id=project_id)
    return project