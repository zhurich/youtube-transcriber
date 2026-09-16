from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"{name} должен быть числом, получено: {raw!r}") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_ids(name: str) -> frozenset[int]:
    raw = _env_str(name)
    if not raw:
        return frozenset()
    return frozenset(int(part) for part in raw.replace(";", ",").split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    bot_token: str
    openai_api_key: str
    allowed_user_ids: frozenset[int]
    transcribe_model: str
    transcribe_language: str
    timestamps: bool
    summary_model: str
    max_video_minutes: int
    workers: int
    transcribe_concurrency: int
    cookies_file: str
    yt_cookies_file: str
    instagram_cookies_file: str
    vk_cookies_file: str
    tiktok_cookies_file: str
    yt_proxy: str
    telegram_api_url: str
    max_file_mb: int
    max_upload_mb: int
    max_download_height: int
    group_debounce_seconds: float
    max_group_items: int
    log_level: str
    data_dir: Path

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def links_dir(self) -> Path:
        return self.data_dir / "links"

    def cookies_for(self, platform_key: str) -> str:
        """Cookies конкретной площадки, иначе общий файл для всех сразу."""
        specific = {
            "yt": self.yt_cookies_file,
            "ig": self.instagram_cookies_file,
            "vk": self.vk_cookies_file,
            "tt": self.tiktok_cookies_file,
        }.get(platform_key, "")
        return specific or self.cookies_file

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def is_allowed(self, user_id: int) -> bool:
        return not self.allowed_user_ids or user_id in self.allowed_user_ids


def load_settings() -> Settings:
    bot_token = _env_str("BOT_TOKEN")
    openai_api_key = _env_str("OPENAI_API_KEY")

    missing = [name for name, value in (("BOT_TOKEN", bot_token), ("OPENAI_API_KEY", openai_api_key)) if not value]
    if missing:
        raise SystemExit(f"Не заданы переменные окружения: {', '.join(missing)}. Скопируйте .env.example в .env.")

    data_dir = Path(_env_str("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    settings = Settings(
        bot_token=bot_token,
        openai_api_key=openai_api_key,
        allowed_user_ids=_env_ids("ALLOWED_USER_IDS"),
        transcribe_model=_env_str("TRANSCRIBE_MODEL", "whisper-1"),
        transcribe_language=_env_str("TRANSCRIBE_LANGUAGE"),
        timestamps=_env_bool("TIMESTAMPS", False),
        summary_model=_env_str("SUMMARY_MODEL", "gpt-4o-mini"),
        max_video_minutes=_env_int("MAX_VIDEO_MINUTES", 180),
        workers=max(1, _env_int("WORKERS", 1)),
        transcribe_concurrency=max(1, _env_int("TRANSCRIBE_CONCURRENCY", 3)),
        cookies_file=_env_str("COOKIES_FILE"),
        yt_cookies_file=_env_str("YT_COOKIES_FILE"),
        instagram_cookies_file=_env_str("INSTAGRAM_COOKIES_FILE"),
        vk_cookies_file=_env_str("VK_COOKIES_FILE"),
        tiktok_cookies_file=_env_str("TIKTOK_COOKIES_FILE"),
        yt_proxy=_env_str("YT_PROXY"),
        telegram_api_url=_env_str("TELEGRAM_API_URL").rstrip("/"),
        # Публичный Bot API не отдаёт боту файлы больше 20 МБ, локальный сервер — до 2000 МБ
        max_file_mb=_env_int("MAX_FILE_MB", 2000 if _env_str("TELEGRAM_API_URL") else 20),
        # Обратно Telegram принимает от бота до 50 МБ, локальный сервер — до 2000 МБ
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 2000 if _env_str("TELEGRAM_API_URL") else 50),
        max_download_height=max(144, _env_int("MAX_DOWNLOAD_HEIGHT", 1080)),
        group_debounce_seconds=max(0.5, _env_float("GROUP_DEBOUNCE_SECONDS", 4.0)),
        max_group_items=max(2, _env_int("MAX_GROUP_ITEMS", 50)),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        data_dir=data_dir,
    )

    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    settings.links_dir.mkdir(parents=True, exist_ok=True)
    return settings
