import asyncio
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models, schemas
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.generator.code_generator import CodeGenerator
from app.tasks.worker import celery_app


@celery_app.task(name="generate_optimized_content")
def generate_optimized_content(
    page_id: str, recommendation_ids: List[str]
) -> Dict[str, Any]:
    """
    Generate optimized HTML/CSS based on recommendations.
    """
    return asyncio.run(_generate_optimized_content_async(page_id, recommendation_ids))


async def _generate_optimized_content_async(
    page_id: str, recommendation_ids: List[str]
) -> Dict[str, Any]:
    """
    Asynchronous implementation of optimized content generation.
    """
    logger.info(f"Starting optimization for page ID: {page_id}")

    async with AsyncSessionLocal() as session:
        # Get page from database
        page = await crud.page.get(session, id=UUID(page_id))
        if not page:
            logger.error(f"Page with ID {page_id} not found")
            return {"success": False, "error": "Page not found"}

        # Get or create optimized page record
        optimized_page = await crud.optimized_page.get_by_page_id(
            session, page_id=UUID(page_id)
        )
        if not optimized_page:
            optimized_page = await crud.optimized_page.create(
                session,
                obj_in=schemas.OptimizedPageCreate(
                    page_id=UUID(page_id),
                    applied_recommendations=[
                        UUID(rec_id) for rec_id in recommendation_ids
                    ],
                ),
            )

        try:
            # Check if page has content to optimize
            if not page.html_content:
                raise ValueError("Page has no HTML content to optimize")

            # Get analysis
            analysis = await crud.page_analysis.get_by_page_id(
                session, page_id=UUID(page_id)
            )
            if not analysis:
                raise ValueError("Page has not been analyzed")

            # Get selected recommendations
            recommendations = []
            for rec_id in recommendation_ids:
                rec = await crud.page_recommendation.get(session, id=UUID(rec_id))
                if rec and rec.analysis_id == analysis.id:
                    recommendations.append(rec)

            # Initialize generator
            generator = CodeGenerator()

            # Generate optimized content
            generation_result = await generator.generate_optimized_content(
                original_html=page.html_content,
                recommendations=recommendations,
                page_metadata=page.page_metadata or {},
                page_structure=page.structure or {},
            )

            # Update optimized page with results
            await crud.optimized_page.update(
                session,
                db_obj=optimized_page,
                obj_in={
                    "html_content": generation_result.get("html_content"),
                    "css_content": generation_result.get("css_content"),
                    "diff": generation_result.get("diff"),
                    "generated_at": datetime.utcnow(),
                },
            )

            # Mark recommendations as implemented
            for rec in recommendations:
                await crud.page_recommendation.update(
                    session, db_obj=rec, obj_in={"is_implemented": True}
                )

            logger.info(f"Successfully generated optimized content for page {page.url}")
            return {"success": True, "page_id": page_id}

        except Exception as e:
            # Update optimized page with error status
            await crud.optimized_page.update(
                session, db_obj=optimized_page, obj_in={"status": "draft"}
            )
            logger.exception(
                f"Exception generating optimized content for page {page.url}"
            )
            return {"success": False, "error": str(e)}
