from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.page import PageRecommendation
from app.schemas.analysis import PageRecommendationCreate, PageRecommendationUpdate


class CRUDPageRecommendation(
    CRUDBase[PageRecommendation, PageRecommendationCreate, PageRecommendationUpdate]
):
    async def get_by_analysis_id(
        self, db: AsyncSession, *, analysis_id: UUID
    ) -> List[PageRecommendation]:
        """
        Get recommendations by analysis ID.
        """
        result = await db.execute(
            select(PageRecommendation)
            .filter(PageRecommendation.analysis_id == analysis_id)
            .order_by(PageRecommendation.priority)
        )
        return result.scalars().all()

    async def get_by_analysis_id_and_category(
        self, db: AsyncSession, *, analysis_id: UUID, category: str
    ) -> List[PageRecommendation]:
        """
        Get recommendations by analysis ID and category.
        """
        result = await db.execute(
            select(PageRecommendation)
            .filter(
                PageRecommendation.analysis_id == analysis_id,
                PageRecommendation.category == category,
            )
            .order_by(PageRecommendation.priority)
        )
        return result.scalars().all()


page_recommendation = CRUDPageRecommendation(PageRecommendation)
