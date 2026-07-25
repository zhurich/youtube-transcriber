from __future__ import annotations

import asyncio
import logging
import re
import shutil
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

YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com/(?:watch\?|shorts/|live/|embed/)\S*|youtu\.be/\S+)",
    re.IGNORECASE,
)


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


@dataclass
class AudioChunk:
    path: Path
    offset: float


@dataclass
class PreparedAudio:
    chunks: list[AudioChunk] = field(default_factory=list)
    duration: float = 0.0


def find_youtube_url(text: str) -> str | None:
    match = YOUTUBE_URL_RE.search(text or "")
    return match.group(0) if match else None


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


def _base_ydl_opts(settings: Settings) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "logger": logging.getLogger("yt_dlp"),
    }
    if settings.yt_cookies_file:
        opts["cookiefile"] = settings.yt_cookies_file
    if settings.yt_proxy:
        opts["proxy"] = settings.yt_proxy
    return opts


def _humanize_download_error(error: Exception) -> str:
    message = str(error)
    if "confirm you" in message or "Sign in to confirm" in message:
        return (
            "YouTube требует подтверждения, что запрос не от бота. "
            "Добавьте файл cookies (YT_COOKIES_FILE) или прокси (YT_PROXY) в .env."
        )
    if "Private video" in message:
        return "Это приватное видео."
    if "Video unavailable" in message:
        return "Видео недоступно (удалено или заблокировано в регионе сервера)."
    if "members-only" in message.lower():
        return "Видео доступно только участникам канала."
    return f"Не удалось получить видео: {message.splitlines()[0][:300]}"


def _extract_info(url: str, settings: Settings) -> VideoInfo:
    with YoutubeDL(_base_ydl_opts(settings)) as ydl:
        info = ydl.extract_info(url, download=False)

    if info.get("_type") == "playlist":
        entries = [entry for entry in info.get("entries") or [] if entry]
        if not entries:
            raise AudioError("По ссылке нет доступных видео.")
        info = entries[0]

    if info.get("is_live"):
        raise AudioError("Это прямой эфир, дождитесь окончания трансляции.")

    return VideoInfo(
        video_id=str(info.get("id") or ""),
        title=str(info.get("title") or "Без названия"),
        duration=int(info.get("duration") or 0),
        uploader=str(info.get("uploader") or ""),
        url=str(info.get("webpage_url") or url),
    )


async def get_video_info(url: str, settings: Settings) -> VideoInfo:
    try:
        return await asyncio.to_thread(_extract_info, url, settings)
    except AudioError:
        raise
    except DownloadError as exc:
        raise AudioError(_humanize_download_error(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - yt-dlp бросает разнородные исключения
        logger.exception("Не удалось получить информацию о видео")
        raise AudioError(_humanize_download_error(exc)) from exc


def _download(url: str, workdir: Path, settings: Settings, progress: Progress) -> Path:
    def hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        done = status.get("downloaded_bytes") or 0
        if total:
            progress.percent = min(99.0, done / total * 100)

    opts = _base_ydl_opts(settings) | {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "progress_hooks": [hook],
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


async def prepare_youtube_audio(
    url: str, workdir: Path, settings: Settings, progress: Progress
) -> PreparedAudio:
    """Скачивает аудиодорожку с YouTube и готовит её к отправке в OpenAI."""
    try:
        source = await asyncio.to_thread(_download, url, workdir, settings, progress)
    except AudioError:
        raise
    except DownloadError as exc:
        raise AudioError(_humanize_download_error(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось скачать аудио")
        raise AudioError(_humanize_download_error(exc)) from exc

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
