from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.page import PageAnalysis
from app.schemas.analysis import PageAnalysisCreate, PageAnalysisUpdate


class CRUDPageAnalysis(CRUDBase[PageAnalysis, PageAnalysisCreate, PageAnalysisUpdate]):
    async def get_by_page_id(
        self, db: AsyncSession, *, page_id: UUID
    ) -> Optional[PageAnalysis]:
        """
        Get analysis by page ID.
        """
        result = await db.execute(
            select(PageAnalysis).filter(PageAnalysis.page_id == page_id)
        )
        return result.scalars().first()


page_analysis = CRUDPageAnalysis(PageAnalysis)
