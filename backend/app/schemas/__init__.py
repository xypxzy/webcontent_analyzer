from app.schemas.token import Token, TokenPayload
from app.schemas.user import User, UserCreate, UserUpdate
from app.schemas.project import Project, ProjectCreate, ProjectUpdate
from app.schemas.page import Page, PageCreate, PageUpdate
from app.schemas.analysis import (
    PageAnalysis,
    PageAnalysisCreate,
    PageRecommendation,
    PageRecommendationCreate,
    ContentAnalysisRequest,
)
from app.schemas.optimization import (
    OptimizedPage,
    OptimizedPageCreate,
    OptimizedPageStatusUpdate,
    OptimizationCreate,
)
