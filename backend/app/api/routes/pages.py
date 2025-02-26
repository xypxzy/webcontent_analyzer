from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.api import deps
from app.tasks import parsing_tasks

router = APIRouter()


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
    Создание новой страницы и запуск процесса парсинга.
    """
    # Проверка существования проекта и доступа пользователя
    project = await crud.project.get(session, id=project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    # Проверка, существует ли URL уже в этом проекте
    existing_page = await crud.page.get_by_url_and_project(
        session, url=page_in.url, project_id=project_id
    )
    if existing_page:
        raise HTTPException(
            status_code=400, detail="Этот URL уже анализируется в этом проекте"
        )

    # Создание страницы
    page = await crud.page.create(
        session,
        obj_in=schemas.PageCreate(
            url=page_in.url,
            project_id=project_id,
            parse_settings=page_in.parse_settings,
        ),
    )

    # Запуск парсинга в фоне
    background_tasks.add_task(parsing_tasks.parse_page, str(page.id))

    return page


@router.post("/pages/{page_id}/reparse", response_model=schemas.Page)
async def reparse_page(
    *,
    page_id: UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Перезапуск парсинга для страницы.
    """
    page = await crud.page.get(session, id=page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Страница не найдена")

    # Проверка доступа пользователя к проекту
    project = await crud.project.get(session, id=page.project_id)
    if project.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    # Обновление статуса страницы и увеличение счетчика попыток
    page = await crud.page.update(
        session,
        db_obj=page,
        obj_in={"status": "pending", "retry_count": page.retry_count + 1},
    )

    # Запуск парсинга в фоне
    background_tasks.add_task(parsing_tasks.parse_page, str(page.id))

    return page


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
