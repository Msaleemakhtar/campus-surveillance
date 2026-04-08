from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# NullPool engine for daemon worker threads that call asyncio.run().
# asyncpg connections are bound to the event loop that created them.
# Worker threads use asyncio.run() (new event loop each call), which conflicts
# with the main pool's loop. NullPool creates/closes a fresh connection per
# session — no loop binding, no "connection belongs to different event loop" errors.
_worker_engine = create_async_engine(settings.DATABASE_URL, echo=False, poolclass=NullPool)
WorkerAsyncSessionLocal = async_sessionmaker(_worker_engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
