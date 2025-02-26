import asyncio
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.analyzer.content_analyzer import ContentAnalyzer
from app.tasks.worker import celery_app


@celery_app.task(name="analyze_page_content")
def analyze_page_content(page_id: str) -> Dict[str, Any]:
    """
    Analyze page content and generate recommendations.
    """
    return asyncio.run(_analyze_page_content_async(page_id))


async def _analyze_page_content_async(page_id: str) -> Dict[str, Any]:
    """
    Asynchronous implementation of content analysis.
    """
    logger.info(f"Starting content analysis for page ID: {page_id}")

    async with AsyncSessionLocal() as session:
        # Get page from database
        page = await crud.page.get(session, id=UUID(page_id))
        if not page:
            logger.error(f"Page with ID {page_id} not found")
            return {"success": False, "error": "Page not found"}

        # Get or create analysis record
        analysis = await crud.page_analysis.get_by_page_id(
            session, page_id=UUID(page_id)
        )
        if not analysis:
            analysis = await crud.page_analysis.create(
                session, obj_in=schemas.PageAnalysisCreate(page_id=UUID(page_id))
            )

        # Update analysis status to in_progress
        await crud.page_analysis.update(
            session, db_obj=analysis, obj_in={"status": "in_progress"}
        )

        try:
            # Check if page has content to analyze
            if not page.text_content or not page.html_content:
                raise ValueError("Page has no content to analyze")

            # Initialize analyzer
            analyzer = ContentAnalyzer()

            # Perform analysis
            analysis_result = await analyzer.analyze_content(
                text=page.text_content,
                html=page.html_content,
                metadata=page.page_metadata or {},
                url=str(page.url),
                structure=page.structure or {},
            )

            # Update analysis with results
            await crud.page_analysis.update(
                session,
                db_obj=analysis,
                obj_in={
                    "status": "completed",
                    "analyzed_at": datetime.utcnow(),
                    "seo_metrics": analysis_result.get("seo_metrics"),
                    "readability_metrics": analysis_result.get("readability_metrics"),
                    "keywords": analysis_result.get("keywords"),
                    "entities": analysis_result.get("entities"),
                    "sentiment": analysis_result.get("sentiment"),
                    "topics": analysis_result.get("topics"),
                    "content_quality": analysis_result.get("content_quality"),
                    "ux_metrics": analysis_result.get("ux_metrics"),
                },
            )

            # Create recommendations
            await _create_recommendations(
                session, analysis.id, analysis_result.get("recommendations", [])
            )

            logger.info(f"Successfully analyzed content for page {page.url}")
            return {"success": True, "page_id": page_id}

        except Exception as e:
            # Update analysis with error status
            await crud.page_analysis.update(
                session, db_obj=analysis, obj_in={"status": "error"}
            )
            logger.exception(f"Exception analyzing content for page {page.url}")
            return {"success": False, "error": str(e)}


async def _create_recommendations(
    session: AsyncSession, analysis_id: UUID, recommendations: List[Dict[str, Any]]
) -> None:
    """
    Create recommendation records from analysis results.
    """
    # Remove existing recommendations
    result = await session.execute(
        select(models.PageRecommendation).filter(
            models.PageRecommendation.analysis_id == analysis_id
        )
    )
    existing_recommendations = result.scalars().all()
    for rec in existing_recommendations:
        await session.delete(rec)

    # Create new recommendations
    for rec_data in recommendations:
        recommendation = models.PageRecommendation(
            analysis_id=analysis_id,
            category=rec_data.get("category", "content"),
            priority=rec_data.get("priority", 3),
            title=rec_data.get("title"),
            description=rec_data.get("description"),
            suggestion=rec_data.get("suggestion"),
            affected_elements=rec_data.get("affected_elements"),
            is_implemented=False,
        )
        session.add(recommendation)

    await session.commit()
