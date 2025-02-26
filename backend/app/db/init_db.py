import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.utils.security import get_password_hash


async def init_db() -> None:
    """Initialize database with initial data."""
    try:
        db: AsyncSession = AsyncSessionLocal()

        # Check if we already have users
        result = await db.execute(select(User).limit(1))
        user = result.scalars().first()

        # Create initial superuser if no users exist
        if not user:
            admin_user = User(
                email="admin@example.com",
                hashed_password=get_password_hash("admin"),
                full_name="Administrator",
                is_superuser=True,
            )
            db.add(admin_user)
            await db.commit()
            logging.info("Created initial admin user")

        await db.close()
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_db())
