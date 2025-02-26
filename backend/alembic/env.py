import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

# Import your models here
from app.models.base import Base
from app.models.project import Project
from app.models.page import Page, PageAnalysis, PageRecommendation, OptimizedPage
from app.models.user import User  # Ensure you have this model
from app.core.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Set the SQLAlchemy URL from environment
# Use a sync connection string for offline mode and async for online mode
sqlalchemy_url = str(settings.DATABASE_URI).replace("postgresql+asyncpg", "postgresql")
config.set_main_option("sqlalchemy.url", sqlalchemy_url)

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

# Add your model's MetaData object here
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    # Using async approach requires special handling
    # Convert the asyncpg URL to a psycopg2 URL for Alembic
    url = str(settings.DATABASE_URI).replace("postgresql+asyncpg", "postgresql")

    # Create configuration with our converted URL
    configuration = {
        "sqlalchemy.url": url,
        "script_location": "alembic",
    }

    # Create a sync engine for Alembic to use
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # We're using synchronous approach for simplicity to avoid the async complexity
    url = config.get_main_option("sqlalchemy.url")
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # For simplicity, we'll use the synchronous approach to avoid async issues
    run_migrations_online()
