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
    FRAME_STORE: str = "/home/salim/Desktop/campus-surveillance/backend/frame_store"

    # Alert cooldowns (seconds) — how often the same alert type can fire per track
    ALERT_COOLDOWN_LOITERING: float = 60.0
    ALERT_COOLDOWN_RUNNING: float = 10.0
    ALERT_COOLDOWN_UNAUTHORIZED: float = 120.0
    ALERT_COOLDOWN_AFTER_HOURS: float = 300.0

    # Re-ID crop quality filters
    REID_MIN_CROP_PX: int = 40        # min(w, h) must be at least this many pixels
    REID_MAX_ASPECT_RATIO: float = 2.0  # w/h above this → horizontal blob, skip


settings = Settings()
