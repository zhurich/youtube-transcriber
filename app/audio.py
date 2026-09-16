from __future__ import annotations

import asyncio
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.config import Settings

logger = logging.getLogger(__name__)

# 64 kbps моно: OpenAI принимает файлы до 25 МБ, 15 минут такого аудио весят ~7 МБ.
AUDIO_BITRATE = "64k"
CHUNK_SECONDS = 900
SINGLE_FILE_LIMIT_BYTES = 20 * 1024 * 1024

@dataclass(frozen=True)
class Platform:
    """Площадка, с которой бот умеет работать через yt-dlp."""

    key: str
    title: str
    icon: str
    pattern: re.Pattern[str]
    cookies_var: str


PLATFORMS: tuple[Platform, ...] = (
    Platform(
        key="yt",
        title="YouTube",
        icon="🎬",
        pattern=re.compile(
            r"https?://(?:www\.|m\.|music\.)?"
            r"(?:youtube\.com/(?:watch\?|shorts/|live/|embed/)\S*|youtu\.be/\S+)",
            re.IGNORECASE,
        ),
        cookies_var="YT_COOKIES_FILE",
    ),
    Platform(
        key="ig",
        title="Instagram",
        icon="📸",
        pattern=re.compile(
            r"https?://(?:www\.)?instagram\.com/(?:[\w.]+/)?(?:p|reel|reels|tv|share)/\S+",
            re.IGNORECASE,
        ),
        cookies_var="INSTAGRAM_COOKIES_FILE",
    ),
    Platform(
        key="vk",
        title="VK",
        icon="🎞",
        pattern=re.compile(
            r"https?://(?:www\.|m\.)?(?:vk\.com|vkvideo\.ru|vk\.ru)/\S*(?:video|clip)\S*",
            re.IGNORECASE,
        ),
        cookies_var="VK_COOKIES_FILE",
    ),
    Platform(
        key="tt",
        title="TikTok",
        icon="🎵",
        pattern=re.compile(
            r"https?://(?:www\.|m\.|vm\.|vt\.)?tiktok\.com/\S+",
            re.IGNORECASE,
        ),
        cookies_var="TIKTOK_COOKIES_FILE",
    ),
)

PLATFORM_BY_KEY = {platform.key: platform for platform in PLATFORMS}
PLATFORM_NAMES = ", ".join(platform.title for platform in PLATFORMS)


class AudioError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


@dataclass
class Progress:
    """Разделяемое состояние прогресса: пишется из потока yt-dlp, читается ботом."""

    percent: float = 0.0

    def as_suffix(self) -> str:
        return f" {self.percent:.0f}%" if self.percent > 0 else ""


@dataclass
class VideoInfo:
    video_id: str
    title: str
    duration: int
    uploader: str
    url: str
    thumbnail: str = ""
    view_count: int = 0
    platform_key: str = "yt"

    @property
    def platform(self) -> Platform | None:
        return PLATFORM_BY_KEY.get(self.platform_key)

    @property
    def icon(self) -> str:
        platform = self.platform
        return platform.icon if platform else "🎬"


@dataclass
class DownloadedVideo:
    path: Path
    title: str
    duration: int
    width: int
    height: int


@dataclass
class AudioChunk:
    path: Path
    offset: float


@dataclass
class PreparedAudio:
    chunks: list[AudioChunk] = field(default_factory=list)
    duration: float = 0.0


def detect_platform(url: str) -> Platform | None:
    for platform in PLATFORMS:
        if platform.pattern.match(url or ""):
            return platform
    return None


def find_media_url(text: str) -> tuple[str, Platform] | None:
    """Ищет в тексте первую ссылку на поддерживаемую площадку."""
    best: tuple[int, str, Platform] | None = None
    for platform in PLATFORMS:
        match = platform.pattern.search(text or "")
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.group(0).rstrip(".,;)»"), platform)
    return (best[1], best[2]) if best else None


def ensure_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise AudioError(
                f"Не найден {binary}. Установите ffmpeg: sudo apt install -y ffmpeg"
            )


def check_js_runtime() -> None:
    if not any(shutil.which(binary) for binary in ("deno", "node", "bun")):
        logger.warning(
            "Не найден JavaScript-рантайм (deno). yt-dlp сможет получить не все форматы YouTube; "
            "установите deno — см. README."
        )


def _base_ydl_opts(settings: Settings, url: str = "") -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "logger": logging.getLogger("yt_dlp"),
    }
    platform = detect_platform(url)
    cookies = settings.cookies_for(platform.key if platform else "")
    if cookies:
        opts["cookiefile"] = cookies
    if settings.yt_proxy:
        opts["proxy"] = settings.yt_proxy
    return opts


def _cookies_hint(platform: Platform | None) -> str:
    variable = platform.cookies_var if platform else "COOKIES_FILE"
    return f"Добавьте файл cookies ({variable}) или прокси (YT_PROXY) в .env."


AUTH_MARKERS = (
    "login required",
    "requires login",
    "logged-in",
    "log in",
    "cookies",
    "authentication",
    "not granting access",
    "account",
)


def _short_reason(message: str) -> str:
    """Вытаскивает суть из многострочной ругани yt-dlp."""
    first = message.splitlines()[0] if message.splitlines() else ""
    first = re.sub(r"^ERROR:\s*", "", first)
    first = re.sub(r"^\[[^\]]+\]\s*", "", first)  # [vk], [Instagram]
    first = re.sub(r"^[\w.-]{1,40}:\s+", "", first)  # id ролика перед текстом
    return first.strip()[:200]


def _humanize_download_error(error: Exception, platform: Platform | None = None) -> str:
    message = str(error)
    low = message.lower()
    site = platform.title if platform else "Площадка"

    if "confirm you" in low or "sign in to confirm" in low:
        return f"{site} требует подтвердить, что запрос не от бота. {_cookies_hint(platform)}"
    if any(marker in low for marker in AUTH_MARKERS):
        return f"{site} отдаёт это видео только авторизованным. {_cookies_hint(platform)}"
    if "rate-limit" in low or "429" in low or "too many requests" in low:
        return f"{site} временно ограничила частоту запросов. Попробуйте позже."
    if "private" in low:
        return "Это приватное видео."
    if "members-only" in low:
        return "Видео доступно только участникам канала."
    if "unsupported url" in low:
        return "Не понимаю эту ссылку. Пришлите прямую ссылку на конкретное видео."
    if "unavailable" in low or "not available" in low or "not found" in low or "404" in low:
        return "Видео недоступно: удалено, приватное или закрыто для региона сервера."

    reason = _short_reason(message)
    if len(reason) < 12:
        # Бывает у VK: yt-dlp отдаёт голое "Error" без всякой конкретики
        return (
            f"{site} не отдала это видео. Обычно так бывает, когда ролик удалён, "
            f"приватный или доступен только авторизованным. {_cookies_hint(platform)}"
        )
    if platform:
        return f"Не удалось получить видео с {platform.title}: {reason}"
    return f"Не удалось получить видео: {reason}"


def _clean_title(info: dict) -> str:
    """У TikTok и Instagram названия нет — берём подпись к ролику."""
    for field in ("title", "description", "alt_title"):
        value = str(info.get(field) or "").strip().replace("\n", " ")
        if value:
            return value[:120]
    return "Без названия"


def _extract_info(url: str, settings: Settings) -> VideoInfo:
    with YoutubeDL(_base_ydl_opts(settings, url)) as ydl:
        info = ydl.extract_info(url, download=False)

    if info.get("_type") == "playlist":
        entries = [entry for entry in info.get("entries") or [] if entry]
        if not entries:
            raise AudioError("По ссылке нет доступных видео.")
        info = entries[0]

    if info.get("is_live"):
        raise AudioError("Это прямой эфир, дождитесь окончания трансляции.")

    platform = detect_platform(str(info.get("webpage_url") or url)) or detect_platform(url)
    return VideoInfo(
        video_id=str(info.get("id") or ""),
        title=_clean_title(info),
        duration=int(info.get("duration") or 0),
        uploader=str(info.get("uploader") or info.get("channel") or info.get("uploader_id") or ""),
        url=str(info.get("webpage_url") or url),
        thumbnail=str(info.get("thumbnail") or ""),
        view_count=int(info.get("view_count") or 0),
        platform_key=platform.key if platform else "",
    )


async def get_video_info(url: str, settings: Settings) -> VideoInfo:
    platform = detect_platform(url)
    try:
        return await asyncio.to_thread(_extract_info, url, settings)
    except AudioError:
        raise
    except DownloadError as exc:
        raise AudioError(_humanize_download_error(exc, platform)) from exc
    except Exception as exc:  # noqa: BLE001 - yt-dlp бросает разнородные исключения
        logger.exception("Не удалось получить информацию о видео")
        raise AudioError(_humanize_download_error(exc, platform)) from exc


def _progress_hook(progress: Progress) -> Callable[[dict], None]:
    def hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        done = status.get("downloaded_bytes") or 0
        if total:
            progress.percent = min(99.0, done / total * 100)

    return hook


def _download(url: str, workdir: Path, settings: Settings, progress: Progress) -> Path:
    opts = _base_ydl_opts(settings, url) | {
        # У TikTok и Instagram отдельной аудиодорожки нет — там сработает запасной "best"
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "progress_hooks": [_progress_hook(progress)],
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloads = info.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])

    candidates = sorted(workdir.glob("source.*"))
    if not candidates:
        raise AudioError("Аудиодорожка скачалась, но файл не найден.")
    return candidates[0]


def _download_video_sync(url: str, workdir: Path, settings: Settings, progress: Progress) -> DownloadedVideo:
    height = settings.max_download_height
    opts = _base_ydl_opts(settings, url) | {
        # mp4+m4a — то, что Telegram играет прямо в чате; остальные варианты запасные.
        # TikTok и Instagram отдают готовый прогрессивный файл, его подхватит "b[height<=N]"
        "format": (
            f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/"
            f"bv*[height<={height}]+ba/b[height<={height}]/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "video.%(ext)s"),
        "progress_hooks": [_progress_hook(progress)],
        # Обрывает закачку, не дожидаясь конца, если поток заведомо больше лимита Telegram
        "max_filesize": settings.max_upload_bytes,
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads and downloads[0].get("filepath") else None
    if path is None or not path.exists():
        candidates = sorted(workdir.glob("video.*"))
        path = candidates[0] if candidates else None
    if path is None or not path.exists():
        raise AudioError(
            f"Видео не скачалось целиком — скорее всего, оно больше {settings.max_upload_mb} МБ. "
            f"Уменьшите MAX_DOWNLOAD_HEIGHT (сейчас {height}p) или скачайте видео вручную."
        )

    formats = info.get("requested_formats") or [info]
    return DownloadedVideo(
        path=path,
        title=str(info.get("title") or "video"),
        duration=int(info.get("duration") or 0),
        width=int(next((fmt.get("width") for fmt in formats if fmt.get("width")), 0) or 0),
        height=int(next((fmt.get("height") for fmt in formats if fmt.get("height")), 0) or 0),
    )


async def download_video(
    url: str, workdir: Path, settings: Settings, progress: Progress
) -> DownloadedVideo:
    """Скачивает видео со звуком в mp4 не выше MAX_DOWNLOAD_HEIGHT."""
    platform = detect_platform(url)
    try:
        video = await asyncio.to_thread(_download_video_sync, url, workdir, settings, progress)
    except AudioError:
        raise
    except DownloadError as exc:
        raise AudioError(_humanize_download_error(exc, platform)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось скачать видео")
        raise AudioError(_humanize_download_error(exc, platform)) from exc

    size = video.path.stat().st_size
    if size > settings.max_upload_bytes:
        raise AudioError(
            f"Видео весит {size / 1024 / 1024:.0f} МБ, а Telegram принимает от бота не больше "
            f"{settings.max_upload_mb} МБ. Уменьшите MAX_DOWNLOAD_HEIGHT "
            f"(сейчас {settings.max_download_height}p) или поднимите локальный Bot API сервер."
        )
    return video


async def _run(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(f"Ошибка ffmpeg: {detail[-1] if detail else 'неизвестная ошибка'}")


async def probe_duration(path: Path) -> float:
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0


async def prepare_remote_audio(
    url: str, workdir: Path, settings: Settings, progress: Progress
) -> PreparedAudio:
    """Скачивает звук по ссылке и готовит его к отправке в OpenAI."""
    platform = detect_platform(url)
    try:
        source = await asyncio.to_thread(_download, url, workdir, settings, progress)
    except AudioError:
        raise
    except DownloadError as exc:
        raise AudioError(_humanize_download_error(exc, platform)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось скачать аудио")
        raise AudioError(_humanize_download_error(exc, platform)) from exc

    return await prepare_audio_file(source, workdir)


async def prepare_audio_file(source: Path, workdir: Path) -> PreparedAudio:
    """Извлекает звук в mp3 16 кГц моно и при необходимости режет на части по лимиту OpenAI."""
    audio = workdir / "audio.mp3"
    await _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        AUDIO_BITRATE,
        str(audio),
    )
    source.unlink(missing_ok=True)

    duration = await probe_duration(audio)

    if audio.stat().st_size <= SINGLE_FILE_LIMIT_BYTES:
        return PreparedAudio(chunks=[AudioChunk(path=audio, offset=0.0)], duration=duration)

    chunks_dir = workdir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    await _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio),
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-c",
        "copy",
        str(chunks_dir / "chunk_%04d.mp3"),
    )

    paths = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not paths:
        raise AudioError("Не удалось разрезать аудио на части.")

    chunks: list[AudioChunk] = []
    offset = 0.0
    for path in paths:
        chunks.append(AudioChunk(path=path, offset=offset))
        offset += await probe_duration(path)

    audio.unlink(missing_ok=True)
    return PreparedAudio(chunks=chunks, duration=duration or offset)
