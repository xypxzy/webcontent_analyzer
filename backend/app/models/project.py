from sqlalchemy import Column, ForeignKey, String, Text, Enum, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class Project(BaseModel):
    """Project model for organizing analyzed pages."""

    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    status = Column(
        Enum("active", "archived", "deleted", name="project_status"),
        default="active",
        nullable=False,
    )
    settings = Column(JSON, nullable=True)
    page_limit = Column(Integer, default=100, nullable=False)

    # Relationships
    pages = relationship("Page", back_populates="project", cascade="all, delete-orphan")
    user = relationship("User", back_populates="projects")

    def __repr__(self):
        return f"<Project {self.name}>"
