from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, projects, pages, analysis, generation
from app.core.config import settings
from app.db.session import engine
from app.models import base

app = FastAPI(
    title="WebContentAnalyzer API",
    description="API for analyzing and optimizing website content",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(projects.router, prefix="/api", tags=["Projects"])
app.include_router(pages.router, prefix="/api", tags=["Pages"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(generation.router, prefix="/api", tags=["Generation"])


@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)


@app.get("/api/health", tags=["Health"])
async def health_check():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
