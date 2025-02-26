from sqlalchemy import (
    Column,
    ForeignKey,
    String,
    Text,
    Enum,
    DateTime,
    Integer,
    JSON,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class Page(BaseModel):
    """Page model for storing parsed web pages."""

    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id"), nullable=False)
    url = Column(String(2048), nullable=False, index=True)
    title = Column(String(512), nullable=True)
    status = Column(
        Enum(
            "pending", "parsing", "analyzed", "error", "optimized", name="page_status"
        ),
        default="pending",
        nullable=False,
    )
    last_parsed_at = Column(DateTime, nullable=True)
    parse_settings = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    html_content = Column(Text, nullable=True)
    text_content = Column(Text, nullable=True)
    structure = Column(JSON, nullable=True)
    errors = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="pages")
    analysis = relationship(
        "PageAnalysis",
        back_populates="page",
        uselist=False,
        cascade="all, delete-orphan",
    )
    optimized_page = relationship(
        "OptimizedPage",
        back_populates="page",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Page {self.url}>"


class PageAnalysis(BaseModel):
    """Model for storing analysis results."""

    page_id = Column(
        UUID(as_uuid=True), ForeignKey("page.id"), nullable=False, unique=True
    )
    analyzed_at = Column(DateTime, nullable=True)
    seo_metrics = Column(JSON, nullable=True)
    readability_metrics = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    sentiment = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    content_quality = Column(JSON, nullable=True)
    ux_metrics = Column(JSON, nullable=True)
    status = Column(
        Enum("pending", "in_progress", "completed", "error", name="analysis_status"),
        default="pending",
        nullable=False,
    )

    # Relationships
    page = relationship("Page", back_populates="analysis")
    recommendations = relationship(
        "PageRecommendation", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<PageAnalysis for page_id={self.page_id}>"


class PageRecommendation(BaseModel):
    """Model for storing recommendations based on analysis."""

    analysis_id = Column(
        UUID(as_uuid=True), ForeignKey("pageanalysis.id"), nullable=False
    )
    category = Column(
        Enum(
            "seo",
            "content",
            "structure",
            "ux",
            "conversion",
            name="recommendation_category",
        ),
        nullable=False,
    )
    priority = Column(Integer, default=3, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=True)
    affected_elements = Column(JSON, nullable=True)
    is_implemented = Column(Boolean, default=False, nullable=False)

    # Relationships
    analysis = relationship("PageAnalysis", back_populates="recommendations")

    def __repr__(self):
        return f"<PageRecommendation {self.title}>"


class OptimizedPage(BaseModel):
    """Model for storing optimized page content."""

    page_id = Column(
        UUID(as_uuid=True), ForeignKey("page.id"), nullable=False, unique=True
    )
    html_content = Column(Text, nullable=False)
    css_content = Column(Text, nullable=True)
    applied_recommendations = Column(JSON, nullable=True)
    generated_at = Column(DateTime, nullable=False)
    status = Column(
        Enum("draft", "published", "rejected", name="optimized_page_status"),
        default="draft",
        nullable=False,
    )
    diff = Column(Text, nullable=True)

    # Relationships
    page = relationship("Page", back_populates="optimized_page")

    def __repr__(self):
        return f"<OptimizedPage for page_id={self.page_id}>"
