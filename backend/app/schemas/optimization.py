from typing import Optional, Dict, Any, List
from datetime import datetime

from pydantic import BaseModel, UUID4


class OptimizedPageBase(BaseModel):
    status: str = "draft"
    applied_recommendations: Optional[List[UUID4]] = None


class OptimizedPageCreate(OptimizedPageBase):
    page_id: UUID4


class OptimizedPageUpdate(OptimizedPageBase):
    html_content: Optional[str] = None
    css_content: Optional[str] = None
    diff: Optional[str] = None


class OptimizedPageStatusUpdate(BaseModel):
    status: str


class OptimizedPageInDBBase(OptimizedPageBase):
    id: UUID4
    page_id: UUID4
    created_at: datetime
    updated_at: datetime
    html_content: Optional[str] = None
    css_content: Optional[str] = None
    generated_at: Optional[datetime] = None
    diff: Optional[str] = None

    class Config:
        from_attributes = True


class OptimizedPage(OptimizedPageInDBBase):
    pass


class OptimizationCreate(BaseModel):
    recommendation_ids: List[UUID4]
