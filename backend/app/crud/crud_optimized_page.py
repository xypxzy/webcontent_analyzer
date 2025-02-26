from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.page import OptimizedPage
from app.schemas.optimization import OptimizedPageCreate, OptimizedPageUpdate


class CRUDOptimizedPage(
    CRUDBase[OptimizedPage, OptimizedPageCreate, OptimizedPageUpdate]
):
    async def get_by_page_id(
        self, db: AsyncSession, *, page_id: UUID
    ) -> Optional[OptimizedPage]:
        """
        Get optimized page by page ID.
        """
        result = await db.execute(
            select(OptimizedPage).filter(OptimizedPage.page_id == page_id)
        )
        return result.scalars().first()


optimized_page = CRUDOptimizedPage(OptimizedPage)
