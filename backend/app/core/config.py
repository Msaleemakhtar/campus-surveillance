from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/campus_db"
    REDIS_URL: str = "redis://localhost:6379"

    INFER_EVERY_N_FRAMES: int = 1   # Phase 1: submit every frame; tune up in Phase 4
    JPEG_QUALITY: int = 75
    TARGET_FPS: int = 10            # display rate; inference runs at ~1 Hz independently

    AFTER_HOURS_START: int = 18
    AFTER_HOURS_END: int = 6

    VIDEOS_BASE: str = "/home/salim/Desktop/campus-survelliance/videos"


settings = Settings()
