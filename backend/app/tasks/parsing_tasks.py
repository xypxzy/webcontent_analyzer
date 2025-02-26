import asyncio
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud, models
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.parser.page_parser import PageParser
from app.tasks.worker import celery_app


@celery_app.task(name="parse_page")
def parse_page(page_id: str) -> Dict[str, Any]:
    """
    Parse a webpage and store the results.
    """
    return asyncio.run(_parse_page_async(page_id))


async def _parse_page_async(page_id: str) -> Dict[str, Any]:
    """
    Asynchronous implementation of page parsing.
    """
    logger.info(f"Starting parsing for page ID: {page_id}")

    async with AsyncSessionLocal() as session:
        # Get page from database
        page = await crud.page.get(session, id=UUID(page_id))
        if not page:
            logger.error(f"Page with ID {page_id} not found")
            return {"success": False, "error": "Page not found"}

        # Update page status to parsing
        await crud.page.update(session, db_obj=page, obj_in={"status": "parsing"})

        try:
            # Initialize parser and parse URL
            parser = PageParser()
            result = await parser.parse_url(str(page.url))

            if not result.get("success"):
                # Update page with error status
                await crud.page.update(
                    session,
                    db_obj=page,
                    obj_in={
                        "status": "error",
                        "errors": {"message": result.get("error", "Unknown error")},
                        "last_parsed_at": datetime.utcnow(),
                    },
                )
                logger.error(f"Error parsing page {page.url}: {result.get('error')}")
                return result

            # Update page with parsing results
            await crud.page.update(
                session,
                db_obj=page,
                obj_in={
                    "status": "analyzed",
                    "title": result.get("title"),
                    "html_content": result.get("html_content"),
                    "text_content": result.get("text_content"),
                    "page_metadata": result.get("page_metadata"),
                    "structure": result.get("structure"),
                    "last_parsed_at": datetime.utcnow(),
                },
            )

            logger.info(f"Successfully parsed page {page.url}")
            return {"success": True, "page_id": page_id}

        except Exception as e:
            # Update page with error status
            await crud.page.update(
                session,
                db_obj=page,
                obj_in={
                    "status": "error",
                    "errors": {"message": str(e)},
                    "last_parsed_at": datetime.utcnow(),
                },
            )
            logger.exception(f"Exception parsing page {page.url}")
            return {"success": False, "error": str(e)}
