from typing import List, Optional, Union
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from pydantic import HttpUrl

from app.crud.base import CRUDBase
from app.models.page import Page
from app.schemas.page import PageCreate, PageUpdate


class CRUDPage(CRUDBase[Page, PageCreate, PageUpdate]):
    async def get_multi_by_project(
            self, db: AsyncSession, *, project_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Page]:
        """
        Get multiple pages for a specific project.
        """
        try:
            # Log the query for debugging
            logger.debug(f"Getting pages for project_id: {project_id}, skip: {skip}, limit: {limit}")

            # Execute the query with explicit type casting for project_id
            result = await db.execute(
                select(Page)
                .filter(Page.project_id == project_id)
                .offset(skip)
                .limit(limit)
            )

            # Get all results
            pages = result.scalars().all()
            logger.debug(f"Found {len(pages)} pages for project_id: {project_id}")
            return pages

        except Exception as e:
            # Log the error with full details
            logger.error(f"Error in get_multi_by_project: {str(e)}")
            raise

    async def get_by_url_and_project(
            self, db: AsyncSession, *, url: Union[str, HttpUrl], project_id: UUID
    ) -> Optional[Page]:
        """
        Get a page by URL and project ID.
        """
        try:
            # Convert HttpUrl to string if needed
            url_str = str(url) if isinstance(url, HttpUrl) else url

            logger.debug(f"Searching for page with URL: {url_str} in project: {project_id}")

            result = await db.execute(
                select(Page).filter(Page.url == url_str, Page.project_id == project_id)
            )
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error in get_by_url_and_project: {str(e)}")
            raise

    async def create(self, db: AsyncSession, *, obj_in: PageCreate) -> Page:
        """
        Create a new page with explicit error handling.
        """
        try:
            # Convert obj_in to dict and handle HttpUrl conversion
            page_data = obj_in.dict()

            # Convert HttpUrl to string explicitly
            if 'url' in page_data and hasattr(page_data['url'], '__str__'):
                page_data['url'] = str(page_data['url'])

            logger.debug(f"Creating new page with data: {page_data}")

            # Create the page instance
            db_obj = Page(**page_data)

            # Add to session and commit
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)

            return db_obj
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating page: {str(e)}")
            raise


page = CRUDPage(Page)