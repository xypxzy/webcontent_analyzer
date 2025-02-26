from typing import List, Union

from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
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
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    DATABASE_URI: PostgresDsn = None

    @field_validator("DATABASE_URI", mode="before")
    def assemble_db_connection(cls, v: str, values: dict) -> str:
        if v:
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"{values.get('POSTGRES_DB') or ''}",
        )

    # Redis and Celery
    REDIS_HOST: str
    REDIS_PORT: str = "6379"
    CELERY_BROKER_URL: RedisDsn = None
    CELERY_RESULT_BACKEND: RedisDsn = None

    @field_validator("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", mode="before")
    def assemble_redis_connection(cls, v: str, values: dict) -> str:
        if v:
            return v
        return RedisDsn.build(
            scheme="redis",
            host=values.get("REDIS_HOST"),
            port=values.get("REDIS_PORT"),
            path="/0",
        )

    # Parser settings
    PARSER_TIMEOUT: int = 30
    PARSER_USER_AGENT: str = "WebContentAnalyzer/1.0"
    PARSER_REQUESTS_PER_SECOND: int = 5
    PARSER_MAX_RETRIES: int = 3

    # NLP settings
    NLP_DEFAULT_LANGUAGE: str = "en"
    NLP_MODELS_CACHE_DIR: str = "/app/nlp_models"

    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
