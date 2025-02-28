from typing import List, Union, Optional

from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "development_secret_key"  # Default for development
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    POSTGRES_SERVER: str = "localhost"  # Default for development
    POSTGRES_USER: str = "postgres"  # Default for development
    POSTGRES_PASSWORD: str = "postgres"  # Default for development
    POSTGRES_DB: str = "webcontent_analyzer"  # Default for development
    POSTGRES_PORT: str = "5432"
    DATABASE_URI: Optional[str] = None

    # Redis and Celery
    REDIS_HOST: str = "localhost"  # Default for development
    REDIS_PORT: str = "6379"
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # Parser settings
    PARSER_TIMEOUT: int = 30
    PARSER_USER_AGENT: str = "WebContentAnalyzer/1.0"
    PARSER_REQUESTS_PER_SECOND: int = 5
    PARSER_MAX_RETRIES: int = 3

    # NLP settings
    NLP_DEFAULT_LANGUAGE: str = "en"
    NLP_MODELS_CACHE_DIR: str = "/app/nlp_models"

    # Environment
    ENVIRONMENT: str = "development"  # Default to development
    DEBUG: bool = True  # Default to True for development

    @model_validator(mode="after")
    def assemble_db_connection(self) -> "Settings":
        if not self.DATABASE_URI:
            self.DATABASE_URI = f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return self

    @model_validator(mode="after")
    def assemble_redis_connection(self) -> "Settings":
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = (
                f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
