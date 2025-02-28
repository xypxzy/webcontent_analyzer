from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlalchemy
from loguru import logger

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
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(projects.router, prefix="/api/v1", tags=["Projects"])
app.include_router(pages.router, prefix="/api/v1", tags=["Pages"])
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(generation.router, prefix="/api/v1", tags=["Generation"])


@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    try:
        # Try to establish a connection and create tables
        logger.info(f"Connecting to database at {settings.DATABASE_URI}")
        async with engine.begin() as conn:
            logger.info("Creating database tables if they don't exist...")
            await conn.run_sync(base.Base.metadata.create_all)
            logger.info("Database setup completed successfully")
    except sqlalchemy.exc.OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        logger.warning(
            "Application will continue to run, but database functionality will be limited."
        )
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        logger.warning(
            "Application will continue to run, but database functionality may be limited."
        )


@app.get("/api/health", tags=["Health"])
async def health_check():
    # Basic health check endpoint
    try:
        # Quick database connection check
        async with engine.begin() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return JSONResponse(
        {
            "status": "ok",
            "database": db_status,
            "version": "0.1.0",
            "environment": settings.ENVIRONMENT,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
