from typing import Optional, Dict, Any, List
from datetime import datetime

from pydantic import BaseModel, UUID4


class PageAnalysisBase(BaseModel):
    status: str = "pending"


class PageAnalysisCreate(PageAnalysisBase):
    page_id: UUID4


class PageAnalysisUpdate(PageAnalysisBase):
    seo_metrics: Optional[Dict[str, Any]] = None
    readability_metrics: Optional[Dict[str, Any]] = None
    keywords: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    topics: Optional[Dict[str, Any]] = None
    content_quality: Optional[Dict[str, Any]] = None
    ux_metrics: Optional[Dict[str, Any]] = None


class PageAnalysisInDBBase(PageAnalysisBase):
    id: UUID4
    page_id: UUID4
    created_at: datetime
    updated_at: datetime
    analyzed_at: Optional[datetime] = None
    seo_metrics: Optional[Dict[str, Any]] = None
    readability_metrics: Optional[Dict[str, Any]] = None
    keywords: Optional[Dict[str, Any]] = None
    entities: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    topics: Optional[Dict[str, Any]] = None
    content_quality: Optional[Dict[str, Any]] = None
    ux_metrics: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class PageAnalysis(PageAnalysisInDBBase):
    pass


class PageRecommendationBase(BaseModel):
    category: str
    priority: int = 3
    title: str
    description: str
    suggestion: Optional[str] = None
    affected_elements: Optional[Dict[str, Any]] = None
    is_implemented: bool = False


class PageRecommendationCreate(PageRecommendationBase):
    analysis_id: UUID4


class PageRecommendationUpdate(PageRecommendationBase):
    category: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    is_implemented: Optional[bool] = None


class PageRecommendationInDBBase(PageRecommendationBase):
    id: UUID4
    analysis_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PageRecommendation(PageRecommendationInDBBase):
    pass


class ContentAnalysisRequest(BaseModel):
    """Schema for direct content analysis request."""

    text: str
    html: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    target_keywords: Optional[List[str]] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text content for analysis...",
                "html": "<html><body><h1>Sample Page</h1><p>This is sample content.</p></body></html>",
                "url": "https://example.com/page",
                "metadata": {
                    "title": "Sample Page Title",
                    "description": "This is a sample page description for testing.",
                },
                "target_keywords": ["sample", "testing", "analysis"],
            }
        }
