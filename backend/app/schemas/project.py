from typing import Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, UUID4


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    page_limit: Optional[int] = 100


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(ProjectBase):
    name: Optional[str] = None
    status: Optional[str] = None


class ProjectInDBBase(ProjectBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime
    user_id: UUID4
    status: str

    class Config:
        from_attributes = True


class Project(ProjectInDBBase):
    pass
