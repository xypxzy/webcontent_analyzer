import asyncio
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app import crud, schemas
from app.core.config import settings
from app.db.session import engine

# from app.services.generator.code_generator import CodeGenerator
from app.tasks.worker import celery_app

# Create a dedicated sessionmaker for use in tasks
task_session_factory = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@celery_app.task(name="generate_optimized_content")
def generate_optimized_content(
        page_id: str, recommendation_ids: List[str]
) -> Dict[str, Any]:
    """
    Generate optimized HTML/CSS based on recommendations.
    """
    try:
        # Create and use a dedicated event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_generate_optimized_content_async(page_id, recommendation_ids))
    except Exception as e:
        logger.error(f"Error in generate_optimized_content task: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        loop.close()


async def _generate_optimized_content_async(
        page_id: str, recommendation_ids: List[str]
) -> Dict[str, Any]:
    """
    Asynchronous implementation of optimized content generation.
    """
    logger.info(f"Starting optimization for page ID: {page_id}")

    # Use a dedicated session (not from dependency)
    async with task_session_factory() as session:
        try:
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
                await session.commit()  # Make sure creation is committed

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

                # # Initialize generator
                # generator = CodeGenerator()

                # # Generate optimized content
                # generation_result = await generator.generate_optimized_content(
                #     original_html=page.html_content,
                #     recommendations=recommendations,
                #     page_metadata=page.page_metadata or {},
                #     page_structure=page.structure or {},
                # )

                # Placeholder until we implement the actual generator
                generation_result = {
                    "html_content": "optimized html content",
                    "css_content": "optimized css content",
                    "diff": "diff content",
                }

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
                await session.commit()  # Make sure update is committed

                # Mark recommendations as implemented
                for rec in recommendations:
                    await crud.page_recommendation.update(
                        session, db_obj=rec, obj_in={"is_implemented": True}
                    )
                await session.commit()  # Make sure updates are committed

                logger.info(f"Successfully generated optimized content for page {page.url}")
                return {"success": True, "page_id": page_id}

            except Exception as e:
                # Update optimized page with error status
                await crud.optimized_page.update(
                    session, db_obj=optimized_page, obj_in={"status": "draft"}
                )
                await session.commit()  # Make sure update is committed

                logger.exception(
                    f"Exception generating optimized content for page {page.url}: {str(e)}"
                )
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.exception(f"Error in _generate_optimized_content_async: {str(e)}")
            return {"success": False, "error": str(e)}