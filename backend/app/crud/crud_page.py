from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.page import Page
from app.schemas.page import PageCreate, PageUpdate


class CRUDPage(CRUDBase[Page, PageCreate, PageUpdate]):
    async def get_multi_by_project(
        self, db: AsyncSession, *, project_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Page]:
        """
        Get multiple pages for a specific project.
        """
        result = await db.execute(
            select(Page).filter(Page.project_id == project_id).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def get_by_url_and_project(
        self, db: AsyncSession, *, url: str, project_id: UUID
    ) -> Optional[Page]:
        """
        Get a page by URL and project ID.
        """
        result = await db.execute(
            select(Page).filter(Page.url == url, Page.project_id == project_id)
        )
        return result.scalars().first()


page = CRUDPage(Page)
