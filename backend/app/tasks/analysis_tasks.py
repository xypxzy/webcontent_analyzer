import asyncio
from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app import crud, models, schemas
from app.db.session import engine
from app.services.analyzer.content_analyzer import ContentAnalyzer
from app.tasks.worker import celery_app

# Create a dedicated sessionmaker for use in tasks
task_session_factory = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@celery_app.task(name="analyze_page_content")
def analyze_page_content(page_id: str) -> Dict[str, Any]:
    """
    Analyze page content and generate recommendations.
    """
    try:
        # Create and use a dedicated event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_analyze_page_content_async(page_id))
    except Exception as e:
        logger.error(f"Error in analyze_page_content task: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        loop.close()


async def _analyze_page_content_async(page_id: str) -> Dict[str, Any]:
    """
    Asynchronous implementation of content analysis.
    """
    logger.info(f"Starting content analysis for page ID: {page_id}")

    # Use a dedicated session (not from dependency)
    async with task_session_factory() as session:
        try:
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
                await session.commit()  # Make sure changes are committed

            # Update analysis status to in_progress
            await crud.page_analysis.update(
                session, db_obj=analysis, obj_in={"status": "in_progress"}
            )
            await session.commit()  # Make sure changes are committed

            try:
                # Check if page has content to analyze
                if not page.text_content or not page.html_content:
                    raise ValueError("Page has no content to analyze")

                # Initialize analyzer with appropriate configuration
                analyzer = ContentAnalyzer(
                    config={
                        "language": "en",  # Default language
                        "max_text_length": 100000,  # Max text length to analyze
                        "enable_cache": True,  # Enable caching for better performance
                    }
                )

                # Extract target keywords from page metadata if available
                target_keywords = None
                if page.page_metadata and page.page_metadata.get("keywords"):
                    keywords = page.page_metadata.get("keywords", [])
                    if isinstance(keywords, str):
                        target_keywords = [kw.strip() for kw in keywords.split(",")]
                    elif isinstance(keywords, list):
                        target_keywords = keywords

                # Perform analysis
                analysis_result = await analyzer.analyze_content(
                    text=page.text_content,
                    html=page.html_content,
                    metadata=page.page_metadata or {},
                    url=str(page.url),
                    structure=page.structure or {},
                    target_keywords=target_keywords,
                )

                # Update analysis with results
                await crud.page_analysis.update(
                    session,
                    db_obj=analysis,
                    obj_in={
                        "status": "completed",
                        "analyzed_at": datetime.utcnow(),
                        "seo_metrics": analysis_result.get("seo_metrics"),
                        "keywords": analysis_result.get("semantic_analysis", {}).get(
                            "keywords", {}
                        ),
                        "readability_metrics": analysis_result.get("basic_metrics", {}).get(
                            "readability", {}
                        ),
                        "sentiment": analysis_result.get("sentiment_analysis", {}),
                        "topics": analysis_result.get("semantic_analysis", {}).get(
                            "topics", {}
                        ),
                        "content_quality": analysis_result.get("basic_metrics", {}),
                    },
                )
                await session.commit()  # Make sure changes are committed

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
                await session.commit()  # Make sure changes are committed

                logger.exception(
                    f"Exception analyzing content for page {page.url}: {str(e)}"
                )
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.exception(f"Error in _analyze_page_content_async: {str(e)}")
            return {"success": False, "error": str(e)}


async def _create_recommendations(
        session: AsyncSession, analysis_id: UUID, recommendations: List[Dict[str, Any]]
) -> None:
    """
    Create recommendation records from analysis results.
    """
    try:
        # Remove existing recommendations
        existing_recommendations = await crud.page_recommendation.get_by_analysis_id(
            session, analysis_id=analysis_id
        )
        for rec in existing_recommendations:
            await session.delete(rec)

        await session.flush()  # Make sure deletes are processed

        # Create new recommendations
        for rec_data in recommendations:
            # Map category from analyzer to database category
            category_mapping = {
                "seo": "seo",
                "content": "content",
                "structure": "structure",
                "ux": "ux",
                "conversion": "conversion",
                # Backward compatibility mapping
                "seo_meta_tags": "seo",
                "seo_headings": "seo",
                "seo_url_structure": "seo",
                "seo_keyword_usage": "seo",
                "seo_optimization": "seo",
                "seo_content_relevance": "content",
                "seo_lsi_keywords": "content",
            }

            category = category_mapping.get(rec_data.get("category"), "content")

            # Extract fields from recommendation data
            title = rec_data.get("title", "")
            if not title and "text" in rec_data:
                # Handle different recommendation formats
                title = rec_data["text"]

            description = rec_data.get("description", "")
            if not description and "text" in rec_data:
                description = rec_data["text"]

            suggestion = rec_data.get("suggestion", "")
            if not suggestion:
                suggestion = description

            # Default priority (1=high, 3=medium, 5=low)
            priority = rec_data.get("priority", 3)

            recommendation = models.PageRecommendation(
                analysis_id=analysis_id,
                category=category,
                priority=priority,
                title=title,
                description=description,
                suggestion=suggestion,
                affected_elements=rec_data.get("affected_elements"),
                is_implemented=False,
            )
            session.add(recommendation)

        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.exception(f"Error creating recommendations: {str(e)}")
        raise