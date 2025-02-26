import asyncio
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

from loguru import logger

from app import crud
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
    Асинхронная реализация парсинга страницы.
    """
    logger.info(f"Начало парсинга для страницы ID: {page_id}")

    async with AsyncSessionLocal() as session:
        # Получаем страницу из БД
        page = await crud.page.get(session, id=UUID(page_id))
        if not page:
            logger.error(f"Страница с ID {page_id} не найдена")
            return {"success": False, "error": "Страница не найдена"}

        # Получаем настройки парсинга
        parse_settings = page.parse_settings or {}

        # Обновляем статус страницы на "парсинг"
        await crud.page.update(session, db_obj=page, obj_in={"status": "parsing"})

        try:
            # Инициализируем парсер
            parser = PageParser()

            # Парсим URL
            result = await parser.parse_url(str(page.url))

            if not result.get("success"):
                # Обновляем страницу со статусом ошибки
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
                logger.error(
                    f"Ошибка парсинга страницы {page.url}: {result.get('error')}"
                )
                return result

            # Обновляем страницу с результатами парсинга
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

            logger.info(f"Успешно распарсена страница {page.url}")
            return {"success": True, "page_id": page_id}

        except Exception as e:
            # Обновляем страницу со статусом ошибки
            await crud.page.update(
                session,
                db_obj=page,
                obj_in={
                    "status": "error",
                    "errors": {"message": str(e)},
                    "last_parsed_at": datetime.utcnow(),
                },
            )
            logger.exception(f"Исключение при парсинге страницы {page.url}")
            return {"success": False, "error": str(e)}
