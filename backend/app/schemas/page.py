from typing import Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, UUID4, HttpUrl


class PageBase(BaseModel):
    url: HttpUrl
    parse_settings: Optional[Dict[str, Any]] = None


class PageCreate(PageBase):
    project_id: Optional[UUID4] = None


class PageUpdate(PageBase):
    url: Optional[HttpUrl] = None
    status: Optional[str] = None
    retry_count: Optional[int] = None


class PageInDBBase(PageBase):
    id: UUID4
    project_id: UUID4
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    status: str
    last_parsed_at: Optional[datetime] = None
    page_metadata: Optional[Dict[str, Any]] = None
    text_content: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    errors: Optional[Dict[str, Any]] = None
    retry_count: int

    class Config:
        from_attributes = True


class Page(PageInDBBase):
    pass
