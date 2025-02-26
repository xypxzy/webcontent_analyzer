from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate


class CRUDProject(CRUDBase[Project, ProjectCreate, ProjectUpdate]):
    async def get_multi_by_user(
        self, db: AsyncSession, *, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Project]:
        """
        Get multiple projects for a specific user.
        """
        result = await db.execute(
            select(Project).filter(Project.user_id == user_id).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def create_with_user(
        self, db: AsyncSession, *, obj_in: ProjectCreate, user_id: UUID
    ) -> Project:
        """
        Create a new project with a user ID.
        """
        db_obj = Project(
            **obj_in.dict(),
            user_id=user_id,
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj


project = CRUDProject(Project)
