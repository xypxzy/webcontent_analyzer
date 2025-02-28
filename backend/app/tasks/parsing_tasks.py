import asyncio
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app import crud
from app.db.session import engine
from app.services.parser.page_parser import PageParser
from app.tasks.worker import celery_app

# Create a dedicated sessionmaker for use in tasks
task_session_factory = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@celery_app.task(name="parse_page")
def parse_page(page_id: str) -> Dict[str, Any]:
    """
    Parse a webpage and store the results.
    """
    # Since this is a Celery task, we should use synchronous code here
    # or use a dedicated event loop for this task
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_parse_page_async(page_id))
    except Exception as e:
        logger.error(f"Error in parse_page task: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        loop.close()


async def _parse_page_async(page_id: str) -> Dict[str, Any]:
    """
    Асинхронная реализация парсинга страницы.
    """
    logger.info(f"Начало парсинга для страницы ID: {page_id}")

    # Use a dedicated session (not from dependency)
    async with task_session_factory() as session:
        try:
            # Get page from database
            page = await crud.page.get(session, id=UUID(page_id))
            if not page:
                logger.error(f"Страница с ID {page_id} не найдена")
                return {"success": False, "error": "Страница не найдена"}

            # Update page status to "parsing"
            await crud.page.update(session, db_obj=page, obj_in={"status": "parsing"})
            await session.commit()  # Explicitly commit the update

            try:
                # Initialize parser
                parser = PageParser()

                # Parse URL
                result = await parser.parse_url(str(page.url))

                if not result.get("success"):
                    # Update page with error status
                    await crud.page.update(
                        session,
                        db_obj=page,
                        obj_in={
                            "status": "error",
                            "errors": {
                                "message": result.get("error", "Неизвестная ошибка")
                            },
                            "last_parsed_at": datetime.utcnow(),
                        },
                    )
                    await session.commit()  # Explicitly commit the update
                    logger.error(
                        f"Ошибка парсинга страницы {page.url}: {result.get('error')}"
                    )
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
                await session.commit()  # Explicitly commit the update

                logger.info(f"Успешно распарсена страница {page.url}")
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
                await session.commit()  # Explicitly commit the update
                logger.exception(f"Исключение при парсинге страницы {page.url}")
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.exception(f"Ошибка в _parse_page_async: {str(e)}")
            return {"success": False, "error": str(e)}