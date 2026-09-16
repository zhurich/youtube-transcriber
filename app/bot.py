from __future__ import annotations

import asyncio
import contextlib
import hashlib
import html
import logging
import mimetypes
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReactionTypeEmoji,
)

from app import lesson as lesson_ui
from app.audio import (
    PLATFORM_NAMES,
    AudioError,
    PreparedAudio,
    Progress,
    VideoInfo,
    check_js_runtime,
    download_video,
    ensure_ffmpeg,
    find_media_url,
    get_video_info,
    prepare_audio_file,
    prepare_remote_audio,
)
from app.config import Settings
from app.transcriber import Transcriber, TranscriptionError
from app.utils import format_hms, periodic, split_text

logger = logging.getLogger(__name__)
router = Router()
router.message.filter(F.from_user)

MESSAGE_TEXT_LIMIT = 3500
CAPTION_LIMIT = 900
MAX_JOBS_PER_USER = 3
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
LINK_KEY_RE = re.compile(r"^[a-f0-9]{16}$")
PREVIEW_CACHE_SIZE = 200

MEDIA_EXTENSIONS = frozenset(
    {
        ".3gp", ".aac", ".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov", ".mp3", ".mp4",
        ".mpeg", ".mpg", ".oga", ".ogg", ".ogv", ".opus", ".wav", ".webm", ".wma",
    }
)

HELP_TEXT = (
    f"Пришлите ссылку на видео ({PLATFORM_NAMES}) — я покажу превью и спрошу, "
    "что с ним сделать: скачать файлом или расшифровать.\n"
    "Можно прислать и сам файл — видео, аудио, голосовое или видеосообщение: "
    "его я сразу расшифрую.\n\n"
    "Под готовой расшифровкой будут кнопки:\n"
    "📝 конспект — краткий пересказ текстом\n"
    "❓ вопрос по видео — отвечу по содержанию расшифровки\n"
    "🧩 интерактивный конспект — разделы с листалкой и мини-тест\n\n"
    "Команды:\n"
    "/group — режим группы: копится несколько записей, затем одна общая расшифровка\n"
    "/cancel — выйти из режима группы или из режима вопросов\n"
    "/help — эта справка\n"
    "/id — ваш Telegram ID (нужен для списка доступа)"
)


class AskState(StatesGroup):
    """Пользователь задаёт вопросы по конкретной расшифровке."""

    waiting = State()


@dataclass
class MediaSource:
    """Видео или аудио, присланное в Telegram."""

    file_id: str
    unique_id: str
    title: str
    duration: int
    file_size: int
    suffix: str


@dataclass
class Job:
    chat_id: int
    user_id: int
    status_message_id: int
    url: str | None = None
    media: MediaSource | None = None
    batch: list[MediaSource] | None = None
    kind: str = "transcribe"  # transcribe | download

    @property
    def items(self) -> list[MediaSource]:
        if self.batch:
            return self.batch
        return [self.media] if self.media else []


@dataclass
class Meta:
    """Что показываем пользователю и под каким ключом кэшируем расшифровку."""

    title: str
    duration: int
    token: str
    icon: str

    def header(self) -> str:
        head = f"{self.icon} <b>{html.escape(self.title)}</b>"
        return f"{head}\n⏱ {format_hms(self.duration)}" if self.duration else head


# Что показали в превью, помним, чтобы по нажатию кнопки не ходить на площадку второй раз
_video_cache: dict[str, VideoInfo] = {}


def remember_video(info: VideoInfo) -> None:
    """Кладёт разобранный ролик в кэш по его ссылке."""
    _video_cache.pop(info.url, None)
    while len(_video_cache) >= PREVIEW_CACHE_SIZE:
        _video_cache.pop(next(iter(_video_cache)))
    _video_cache[info.url] = info


def _cache_base(info: VideoInfo) -> str:
    """Ключ кэша расшифровки. У площадок id пересекаются, поэтому добавляем префикс.

    У YouTube префикса нет намеренно: так расшифровки, накопленные до появления
    остальных площадок, остаются в кэше.
    """
    video_id = info.video_id or link_key(info.url)
    return video_id if info.platform_key in ("yt", "") else f"{info.platform_key}_{video_id}"


def link_key(url: str) -> str:
    """Короткий ключ ссылки: ссылки площадок в callback_data целиком не влезают."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def remember_link(settings: Settings, url: str) -> str:
    """Сохраняет ссылку на диск, чтобы кнопки превью работали и после перезапуска."""
    key = link_key(url)
    path = settings.links_dir / f"{key}.txt"
    if not path.exists():
        path.write_text(url, encoding="utf-8")
    return key


def recall_link(settings: Settings, key: str) -> str | None:
    path = settings.links_dir / f"{key}.txt"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def safe_filename(title: str, fallback: str) -> str:
    clean = re.sub(r"[^\w\s.-]", "", title, flags=re.UNICODE).strip()[:60]
    return clean or fallback


def extract_media(message: Message) -> MediaSource | None:
    if message.video:
        video = message.video
        return MediaSource(
            file_id=video.file_id,
            unique_id=video.file_unique_id,
            title=video.file_name or "Видео из Telegram",
            duration=video.duration or 0,
            file_size=video.file_size or 0,
            suffix=Path(video.file_name or "").suffix or ".mp4",
        )

    if message.audio:
        audio = message.audio
        name = " — ".join(part for part in (audio.performer, audio.title) if part)
        return MediaSource(
            file_id=audio.file_id,
            unique_id=audio.file_unique_id,
            title=name or audio.file_name or "Аудио из Telegram",
            duration=audio.duration or 0,
            file_size=audio.file_size or 0,
            suffix=Path(audio.file_name or "").suffix or ".mp3",
        )

    if message.voice:
        voice = message.voice
        return MediaSource(
            file_id=voice.file_id,
            unique_id=voice.file_unique_id,
            title="Голосовое сообщение",
            duration=voice.duration or 0,
            file_size=voice.file_size or 0,
            suffix=".ogg",
        )

    if message.video_note:
        note = message.video_note
        return MediaSource(
            file_id=note.file_id,
            unique_id=note.file_unique_id,
            title="Видеосообщение",
            duration=note.duration or 0,
            file_size=note.file_size or 0,
            suffix=".mp4",
        )

    document = message.document
    if document:
        mime = document.mime_type or ""
        suffix = Path(document.file_name or "").suffix.lower()
        if not suffix and mime:
            suffix = mimetypes.guess_extension(mime) or ""
        if mime.startswith(("video/", "audio/")) or suffix in MEDIA_EXTENSIONS:
            return MediaSource(
                file_id=document.file_id,
                unique_id=document.file_unique_id,
                title=document.file_name or "Файл из Telegram",
                duration=0,
                file_size=document.file_size or 0,
                suffix=suffix or ".bin",
            )

    return None


@dataclass
class GroupSession:
    """Записи, накопленные в режиме /group для одной общей расшифровки."""

    session_id: str
    items: list[MediaSource] = field(default_factory=list)
    panel_message_id: int | None = None
    timer: asyncio.Task[None] | None = None

    @property
    def duration(self) -> int:
        return sum(item.duration for item in self.items)


class GroupCollector:
    """Копит присланные записи и показывает кнопку только после паузы в отправке."""

    def __init__(self, bot: Bot, settings: Settings) -> None:
        self._bot = bot
        self._settings = settings
        self._sessions: dict[tuple[int, int], GroupSession] = {}

    def open(self, chat_id: int, user_id: int) -> GroupSession:
        self.discard(chat_id, user_id)
        session = GroupSession(session_id=uuid.uuid4().hex[:8])
        self._sessions[(chat_id, user_id)] = session
        return session

    def get(self, chat_id: int, user_id: int) -> GroupSession | None:
        return self._sessions.get((chat_id, user_id))

    def discard(self, chat_id: int, user_id: int) -> GroupSession | None:
        session = self._sessions.pop((chat_id, user_id), None)
        if session and session.timer:
            session.timer.cancel()
        return session

    def shutdown(self) -> None:
        for session in self._sessions.values():
            if session.timer:
                session.timer.cancel()
        self._sessions.clear()

    async def add(self, message: Message, media: MediaSource) -> bool:
        """Кладёт запись в открытую сессию. False — режим группы не включён."""
        chat_id, user_id = message.chat.id, message.from_user.id
        session = self._sessions.get((chat_id, user_id))
        if session is None:
            return False

        if len(session.items) >= self._settings.max_group_items:
            await message.answer(
                f"Собрано максимум записей ({self._settings.max_group_items}). "
                "Нажмите кнопку расшифровки или /cancel."
            )
            return True

        session.items.append(media)
        with contextlib.suppress(TelegramAPIError):
            await self._bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message.message_id,
                reaction=[ReactionTypeEmoji(emoji="👍")],
            )

        if session.timer:
            session.timer.cancel()
        session.timer = asyncio.create_task(self._show_panel_later(chat_id, session))
        return True

    async def _show_panel_later(self, chat_id: int, session: GroupSession) -> None:
        try:
            await asyncio.sleep(self._settings.group_debounce_seconds)
            await self._show_panel(chat_id, session)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - сбой панели не должен ломать сессию
            logger.exception("Не удалось показать панель группы")

    async def _show_panel(self, chat_id: int, session: GroupSession) -> None:
        text = (
            f"🎙 Собрано записей: {len(session.items)} · {format_hms(session.duration)}\n"
            "Пришлите ещё или расшифруйте всё одним файлом."
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📄 Расшифровать в единый файл",
                        callback_data=f"grp:go:{session.session_id}",
                    )
                ],
                [InlineKeyboardButton(text="✖️ Отмена", callback_data=f"grp:cancel:{session.session_id}")],
            ]
        )

        # Прошлую панель убираем, чтобы кнопка всегда оставалась последним сообщением
        if session.panel_message_id:
            with contextlib.suppress(TelegramAPIError):
                await self._bot.delete_message(chat_id=chat_id, message_id=session.panel_message_id)
            session.panel_message_id = None

        panel = await self._bot.send_message(chat_id, text, reply_markup=keyboard)
        session.panel_message_id = panel.message_id


class Status:
    """Одно сообщение, которое переписывается по ходу обработки."""

    def __init__(self, bot: Bot, chat_id: int, message_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id
        self._last = ""

    async def set(self, text: str) -> None:
        if text == self._last:
            return
        self._last = text
        try:
            await self._bot.edit_message_text(text, chat_id=self._chat_id, message_id=self._message_id)
        except TelegramAPIError as exc:
            logger.debug("Не удалось обновить статус: %s", exc)

    async def delete(self) -> None:
        with contextlib.suppress(TelegramBadRequest):
            await self._bot.delete_message(chat_id=self._chat_id, message_id=self._message_id)


def preview_keyboard(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⬇️ Скачать видео", callback_data=f"act:dl:{key}"),
                InlineKeyboardButton(text="📝 Транскрибировать", callback_data=f"act:tr:{key}"),
            ]
        ]
    )


def transcript_keyboard(token: str, with_summary: bool = True) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    top = [InlineKeyboardButton(text="❓ Вопрос по видео", callback_data=f"ask:{token}")]
    if with_summary:
        top.insert(0, InlineKeyboardButton(text="📝 Конспект", callback_data=f"sum:{token}"))
    rows.append(top)
    rows.append(
        [InlineKeyboardButton(text="🧩 Интерактивный конспект + тест", callback_data=f"les:{token}")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def ask_keyboard(token: str) -> InlineKeyboardMarkup:
    """Кнопки под ответом: остаёмся в режиме вопросов или выходим из него."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🧩 Конспект + тест", callback_data=f"les:{token}"),
                InlineKeyboardButton(text="✖️ Хватит вопросов", callback_data="askend"),
            ]
        ]
    )


class JobQueue:
    def __init__(self, bot: Bot, settings: Settings, transcriber: Transcriber) -> None:
        self._bot = bot
        self._settings = settings
        self._transcriber = transcriber
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._active: dict[int, int] = {}

    def start(self) -> None:
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"worker-{index}")
            for index in range(self._settings.workers)
        ]

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def jobs_of(self, user_id: int) -> int:
        return self._active.get(user_id, 0)

    async def submit(self, job: Job) -> int:
        self._active[job.user_id] = self._active.get(job.user_id, 0) + 1
        await self._queue.put(job)
        return self._queue.qsize()

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - воркер не должен умирать из-за одной задачи
                logger.exception("Задача завершилась с необработанной ошибкой")
                with contextlib.suppress(Exception):
                    await self._bot.send_message(job.chat_id, "❌ Внутренняя ошибка, попробуйте ещё раз.")
            finally:
                remaining = self._active.get(job.user_id, 1) - 1
                if remaining > 0:
                    self._active[job.user_id] = remaining
                else:
                    self._active.pop(job.user_id, None)
                self._queue.task_done()

    async def _process(self, job: Job) -> None:
        status = Status(self._bot, job.chat_id, job.status_message_id)
        started = time.monotonic()
        workdir = self._settings.tmp_dir / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            if job.kind == "download":
                await self._download_and_send(job, status, workdir)
                return

            meta = await self._resolve_meta(job, status)
            transcript_path = self._settings.transcripts_dir / f"{meta.token}.txt"

            if transcript_path.exists():
                await status.set(f"{meta.header()}\n\n♻️ Нашёл готовую расшифровку в кэше.")
                text = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
            else:
                prepared = await self._fetch_audio(job, meta, workdir, status)
                if not meta.duration:
                    meta.duration = int(sum(item.duration for item in prepared))
                    self._check_duration(meta.duration)
                text = await self._run_transcription(prepared, meta, status)
                await asyncio.to_thread(transcript_path.write_text, text, encoding="utf-8")

            elapsed = format_hms(time.monotonic() - started)
            await status.set(f"{meta.header()}\n\n✅ Готово за {elapsed}")
            await self._send_transcript(
                job.chat_id, meta.title, text, meta.token, as_document=bool(job.batch)
            )

        except (AudioError, TranscriptionError) as exc:
            await status.set(f"❌ {html.escape(str(exc))}")
        finally:
            await asyncio.to_thread(shutil.rmtree, workdir, True)

    async def _download_and_send(self, job: Job, status: Status, workdir: Path) -> None:
        url = job.url or ""
        info = _video_cache.get(url)
        if info is None:
            await status.set("🔎 Читаю информацию о видео…")
            info = await get_video_info(url, self._settings)
            remember_video(info)
        self._check_duration(info.duration)

        header = f"{info.icon} <b>{html.escape(info.title)}</b>"
        if info.duration:
            header += f"\n⏱ {format_hms(info.duration)}"

        progress = Progress()
        quality = f"до {self._settings.max_download_height}p"
        await status.set(f"{header}\n\n⬇️ Скачиваю видео ({quality})…")
        async with periodic(
            lambda: status.set(f"{header}\n\n⬇️ Скачиваю видео ({quality})…{progress.as_suffix()}")
        ):
            video = await download_video(url, workdir, self._settings, progress)

        size_mb = video.path.stat().st_size / 1024 / 1024
        await status.set(f"{header}\n\n📤 Отправляю файл ({size_mb:.0f} МБ)…")
        await self._bot.send_chat_action(job.chat_id, ChatAction.UPLOAD_VIDEO)

        caption = header[:CAPTION_LIMIT]
        key = await asyncio.to_thread(remember_link, self._settings, url)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📝 Ещё и расшифровать", callback_data=f"act:tr:{key}")]
            ]
        )

        await self._bot.send_video(
            job.chat_id,
            FSInputFile(video.path, filename=f"{safe_filename(video.title, 'video')}.mp4"),
            caption=caption,
            duration=video.duration or info.duration,
            width=video.width or None,
            height=video.height or None,
            supports_streaming=True,
            reply_markup=keyboard,
            request_timeout=1800,
        )
        await status.delete()

    async def _resolve_meta(self, job: Job, status: Status) -> Meta:
        if job.batch:
            duration = sum(item.duration for item in job.batch)
            self._check_duration(duration)
            digest = hashlib.sha1("|".join(item.unique_id for item in job.batch).encode()).hexdigest()
            return Meta(
                title=f"Группа из {len(job.batch)} записей",
                duration=duration,
                token=self._token(f"g{digest[:24]}"),
                icon="🎙",
            )

        if job.media:
            self._check_duration(job.media.duration)
            return Meta(
                title=job.media.title,
                duration=job.media.duration,
                token=self._token(job.media.unique_id),
                icon="🎧",
            )

        url = job.url or ""
        info = _video_cache.get(url)
        if info is None:
            await status.set("🔎 Читаю информацию о видео…")
            info = await get_video_info(url, self._settings)
            remember_video(info)
        self._check_duration(info.duration)
        return Meta(
            title=info.title,
            duration=info.duration,
            token=self._token(_cache_base(info)),
            icon=info.icon,
        )

    def _token(self, base: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]", "", base)[:32] or uuid.uuid4().hex
        return f"{clean}_ts" if self._settings.timestamps else clean

    def _check_duration(self, duration: int) -> None:
        limit = self._settings.max_video_minutes * 60
        if limit and duration > limit:
            raise AudioError(
                f"Длительность {format_hms(duration)} больше лимита "
                f"{self._settings.max_video_minutes} мин. Обработка отменена."
            )

    async def _fetch_audio(self, job: Job, meta: Meta, workdir: Path, status: Status) -> list[PreparedAudio]:
        header = meta.header()
        items = job.items

        if items:
            prepared: list[PreparedAudio] = []
            for number, media in enumerate(items, start=1):
                counter = f" ({number}/{len(items)})" if len(items) > 1 else ""
                await status.set(f"{header}\n\n⬇️ Забираю файл из Telegram…{counter}")
                part_dir = workdir / f"part{number:03d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                source = await self._download_media(media, part_dir)
                await status.set(f"{header}\n\n🎚 Извлекаю звук…{counter}")
                prepared.append(await prepare_audio_file(source, part_dir))
            return prepared

        progress = Progress()
        await status.set(f"{header}\n\n⬇️ Скачиваю аудио…")
        async with periodic(lambda: status.set(f"{header}\n\n⬇️ Скачиваю аудио…{progress.as_suffix()}")):
            return [await prepare_remote_audio(job.url or "", workdir, self._settings, progress)]

    async def _download_media(self, media: MediaSource, workdir: Path) -> Path:
        destination = workdir / f"source{media.suffix}"
        try:
            await self._bot.download(media.file_id, destination=destination, timeout=1800)
        except TelegramAPIError as exc:
            message = str(exc)
            if "too big" in message.lower():
                raise AudioError(
                    f"Telegram не отдаёт боту файлы больше {self._settings.max_file_mb} МБ. "
                    "Пришлите ссылку на YouTube или настройте локальный Bot API сервер."
                ) from exc
            raise AudioError(f"Не удалось скачать файл из Telegram: {message}") from exc

        if not destination.exists() or destination.stat().st_size == 0:
            raise AudioError("Файл из Telegram скачался пустым, попробуйте отправить его заново.")
        return destination

    async def _run_transcription(self, prepared: list[PreparedAudio], meta: Meta, status: Status) -> str:
        groups = [item.chunks for item in prepared]
        total = sum(len(group) for group in groups)
        state = {"done": 0}

        def note(done: int, _total: int) -> None:
            state["done"] = done

        def transcribe_status() -> str:
            header = meta.header()
            if total == 1:
                return f"{header}\n\n🧠 Расшифровываю…"
            return f"{header}\n\n🧠 Расшифровываю: {state['done']}/{total} фрагментов"

        await status.set(transcribe_status())
        async with periodic(lambda: status.set(transcribe_status())):
            texts = await self._transcriber.transcribe_groups(groups, on_progress=note)

        if len(texts) == 1:
            if not texts[0]:
                raise TranscriptionError("Модель вернула пустую расшифровку — возможно, здесь нет речи.")
            return texts[0]

        blocks = []
        for number, (item, text) in enumerate(zip(prepared, texts, strict=True), start=1):
            label = f"[Запись {number} · {format_hms(item.duration)}]"
            blocks.append(f"{label}\n{text}" if text else f"{label}\n(речь не распознана)")
        return "\n\n".join(blocks)

    async def _send_transcript(
        self,
        chat_id: int,
        title: str,
        text: str,
        token: str,
        as_document: bool = False,
    ) -> None:
        keyboard = transcript_keyboard(token)
        if not as_document and len(text) <= MESSAGE_TEXT_LIMIT:
            await self._bot.send_message(chat_id, text, parse_mode=None, reply_markup=keyboard)
            return

        filename = safe_filename(title, "transcript")
        document = BufferedInputFile(text.encode("utf-8"), filename=f"{filename}.txt")
        await self._bot.send_document(
            chat_id,
            document,
            caption=f"Расшифровка целиком: {len(text)} символов",
            reply_markup=keyboard,
        )


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(f"Привет! {HELP_TEXT}")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    await message.answer(f"Ваш Telegram ID: <code>{message.from_user.id}</code>")


async def _enqueue(
    message: Message,
    user_id: int,
    settings: Settings,
    queue: JobQueue,
    *,
    url: str | None = None,
    media: MediaSource | None = None,
    batch: list[MediaSource] | None = None,
    kind: str = "transcribe",
) -> None:
    if queue.jobs_of(user_id) >= MAX_JOBS_PER_USER:
        await message.answer(f"У вас уже {MAX_JOBS_PER_USER} задачи в работе. Дождитесь их завершения.")
        return

    status_message = await message.answer("⏳ Задача принята…")
    position = await queue.submit(
        Job(
            chat_id=message.chat.id,
            user_id=user_id,
            status_message_id=status_message.message_id,
            url=url,
            media=media,
            batch=batch,
            kind=kind,
        )
    )
    if position > settings.workers:
        with contextlib.suppress(TelegramBadRequest):
            await status_message.edit_text(f"⏳ В очереди, позиция {position}")


@router.message(Command("group"))
async def cmd_group(message: Message, settings: Settings, collector: GroupCollector) -> None:
    if not settings.is_allowed(message.from_user.id):
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

    collector.open(message.chat.id, message.from_user.id)
    await message.answer(
        "🎙 Режим группы включён. Присылайте голосовые или файлы — я отмечу каждое "
        f"реакцией и через {settings.group_debounce_seconds:g} с после последнего покажу кнопку "
        "«Расшифровать в единый файл».\n\n"
        "Выйти без обработки: /cancel"
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, collector: GroupCollector, state: FSMContext) -> None:
    was_asking = await state.get_state() == AskState.waiting.state
    await state.clear()

    session = collector.discard(message.chat.id, message.from_user.id)
    if session and session.panel_message_id:
        with contextlib.suppress(TelegramAPIError):
            await message.bot.delete_message(chat_id=message.chat.id, message_id=session.panel_message_id)

    if session:
        await message.answer(f"Режим группы выключен, забыто записей: {len(session.items)}.")
    elif was_asking:
        await message.answer("Вопросы по видео закончены. Присылайте новую ссылку или файл.")
    else:
        await message.answer("Нечего отменять. Режим группы включается командой /group")


@router.message(F.video | F.audio | F.voice | F.video_note | F.document)
async def handle_media(
    message: Message,
    settings: Settings,
    queue: JobQueue,
    collector: GroupCollector,
    state: FSMContext,
) -> None:
    if not settings.is_allowed(message.from_user.id):
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

    # Новая запись отменяет вопросы по прошлому видео
    await state.clear()

    media = extract_media(message)
    if media is None:
        await message.answer("Этот файл не похож на видео или аудио. Пришлите медиафайл или ссылку на YouTube.")
        return

    if media.file_size > settings.max_file_bytes:
        size_mb = media.file_size / 1024 / 1024
        hint = (
            "Загрузите видео на YouTube и пришлите ссылку"
            if not settings.telegram_api_url
            else "Уменьшите файл или поднимите MAX_FILE_MB"
        )
        await message.answer(
            f"Файл весит {size_mb:.0f} МБ, а бот может скачать не больше "
            f"{settings.max_file_mb} МБ. {hint}."
        )
        return

    if await collector.add(message, media):
        return

    await _enqueue(message, message.from_user.id, settings, queue, media=media)


@router.message(AskState.waiting, F.text)
async def handle_question(
    message: Message,
    settings: Settings,
    transcriber: Transcriber,
    state: FSMContext,
) -> None:
    if not settings.is_allowed(message.from_user.id):
        await state.clear()
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

    # Ссылка вместо вопроса — значит, человек перешёл к следующему видео
    found = find_media_url(message.text)
    if found:
        await state.clear()
        await show_preview(message, found[0], settings)
        return

    data = await state.get_data()
    token = str(data.get("token") or "")
    transcript_path = settings.transcripts_dir / f"{token}.txt"
    if not TOKEN_RE.match(token) or not transcript_path.exists():
        await state.clear()
        await message.answer("Расшифровка больше недоступна, пришлите ссылку или файл заново.")
        return

    question = (message.text or "").strip()
    if len(question) < 3:
        await message.answer("Вопрос слишком короткий — сформулируйте подробнее.")
        return

    thinking = await message.answer("🤔 Ищу ответ в расшифровке…")
    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        text = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
        answer = await transcriber.answer(question, text)
    except TranscriptionError as exc:
        await thinking.edit_text(f"❌ {html.escape(str(exc))}")
        return

    with contextlib.suppress(TelegramAPIError):
        await thinking.delete()

    title = str(data.get("title") or "")
    parts = split_text(f"💬 {answer}", MESSAGE_TEXT_LIMIT)
    for number, part in enumerate(parts, start=1):
        last = number == len(parts)
        await message.answer(
            part,
            parse_mode=None,
            reply_markup=ask_keyboard(token) if last else None,
        )
    logger.info("Ответ на вопрос по %s (%s), длина %s", token, title, len(answer))


@router.message(F.text)
async def handle_link(message: Message, settings: Settings, state: FSMContext) -> None:
    if not settings.is_allowed(message.from_user.id):
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

    found = find_media_url(message.text)
    if not found:
        await message.answer(
            f"Не вижу ссылку на видео. Я понимаю {PLATFORM_NAMES} — пришлите ссылку "
            "на конкретный ролик или отправьте видео либо аудио файлом."
        )
        return

    await state.clear()
    await show_preview(message, found[0], settings)


async def show_preview(message: Message, url: str, settings: Settings) -> None:
    """Показывает превью ролика и спрашивает, скачать его или расшифровать."""
    status = await message.answer("🔎 Смотрю, что это за видео…")
    try:
        info = await get_video_info(url, settings)
    except AudioError as exc:
        await status.edit_text(f"❌ {html.escape(str(exc))}")
        return

    remember_video(info)
    # Ссылку кладём на диск: в callback_data она не влезает, а кнопка должна
    # работать и после перезапуска бота
    key = await asyncio.to_thread(remember_link, settings, info.url)

    facts = []
    if info.duration:
        facts.append(f"⏱ {format_hms(info.duration)}")
    if info.view_count:
        facts.append(f"👁 {info.view_count:,}".replace(",", " "))

    lines = [f"{info.icon} <b>{html.escape(info.title)}</b>"]
    if info.uploader:
        lines.append(f"👤 {html.escape(info.uploader)}")
    if facts:
        lines.append(" · ".join(facts))
    lines += ["", "Что сделать с видео?"]
    caption = "\n".join(lines)[:CAPTION_LIMIT]
    keyboard = preview_keyboard(key)

    if info.thumbnail:
        try:
            await message.answer_photo(info.thumbnail, caption=caption, reply_markup=keyboard)
        except TelegramAPIError as exc:
            logger.debug("Не удалось отправить обложку: %s", exc)
        else:
            with contextlib.suppress(TelegramAPIError):
                await status.delete()
            return

    await status.edit_text(caption, reply_markup=keyboard)


@router.callback_query(F.data.startswith("act:"), F.message)
async def handle_preview_action(
    callback: CallbackQuery,
    settings: Settings,
    queue: JobQueue,
    state: FSMContext,
) -> None:
    if not settings.is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return

    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[1] not in {"dl", "tr"} or not LINK_KEY_RE.match(parts[2]):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    _, action, key = parts
    url = await asyncio.to_thread(recall_link, settings, key)
    if url is None:
        await callback.answer("Ссылка устарела, пришлите её заново", show_alert=True)
        return

    await state.clear()
    await callback.answer("Скачиваю видео" if action == "dl" else "Отправляю на расшифровку")
    with contextlib.suppress(TelegramAPIError):
        await callback.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=callback.message.message_id,
            reply_markup=None,
        )

    await _enqueue(
        callback.message,
        callback.from_user.id,
        settings,
        queue,
        url=url,
        kind="download" if action == "dl" else "transcribe",
    )


@router.callback_query(F.data.startswith("grp:"), F.message)
async def handle_group_action(
    callback: CallbackQuery,
    settings: Settings,
    queue: JobQueue,
    collector: GroupCollector,
) -> None:
    if not settings.is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return

    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    _, action, session_id = parts
    chat_id = callback.message.chat.id
    message_id = callback.message.message_id
    session = collector.get(chat_id, callback.from_user.id)

    if session is None or session.session_id != session_id or not session.items:
        await callback.answer("Эта группа уже обработана или устарела", show_alert=True)
        with contextlib.suppress(TelegramAPIError):
            await callback.bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
        return

    if action == "cancel":
        collector.discard(chat_id, callback.from_user.id)
        await callback.answer("Группа отменена")
        with contextlib.suppress(TelegramAPIError):
            await callback.bot.edit_message_text(
                f"✖️ Режим группы выключен, забыто записей: {len(session.items)}.",
                chat_id=chat_id,
                message_id=message_id,
            )
        return

    # Сессию закрываем только когда задача точно уйдёт в очередь, иначе собранное потеряется
    if queue.jobs_of(callback.from_user.id) >= MAX_JOBS_PER_USER:
        await callback.answer(
            f"У вас уже {MAX_JOBS_PER_USER} задачи в работе. Дождитесь их и нажмите кнопку снова.",
            show_alert=True,
        )
        return

    collector.discard(chat_id, callback.from_user.id)
    await callback.answer("Отправляю в обработку")
    with contextlib.suppress(TelegramAPIError):
        await callback.bot.edit_message_text(
            f"🎙 Группа из {len(session.items)} записей · {format_hms(session.duration)}",
            chat_id=chat_id,
            message_id=message_id,
        )

    await _enqueue(callback.message, callback.from_user.id, settings, queue, batch=session.items)


@router.callback_query(F.data.startswith("sum:"), F.message)
async def handle_summary(
    callback: CallbackQuery,
    settings: Settings,
    transcriber: Transcriber,
) -> None:
    if not settings.is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return

    token = callback.data.removeprefix("sum:")
    if not TOKEN_RE.match(token):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    transcript_path = settings.transcripts_dir / f"{token}.txt"
    if not transcript_path.exists():
        await callback.answer("Расшифровка больше недоступна, пришлите ссылку заново", show_alert=True)
        return

    await callback.answer("Готовлю конспект…")
    chat_id = callback.message.chat.id
    # Убираем только кнопку конспекта: вопросы и тест по этому видео ещё пригодятся
    with contextlib.suppress(TelegramBadRequest):
        await callback.bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=callback.message.message_id,
            reply_markup=transcript_keyboard(token, with_summary=False),
        )

    summary_path = settings.transcripts_dir / f"{token}.summary.txt"
    try:
        if summary_path.exists():
            summary = await asyncio.to_thread(summary_path.read_text, encoding="utf-8")
        else:
            await callback.bot.send_chat_action(chat_id, ChatAction.TYPING)
            text = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
            summary = await transcriber.summarize(text)
            await asyncio.to_thread(summary_path.write_text, summary, encoding="utf-8")
    except TranscriptionError as exc:
        await callback.bot.send_message(chat_id, f"❌ {html.escape(str(exc))}")
        return

    parts = split_text(f"📝 Конспект\n\n{summary}", MESSAGE_TEXT_LIMIT)
    for number, part in enumerate(parts, start=1):
        await callback.bot.send_message(
            chat_id,
            part,
            parse_mode=None,
            reply_markup=ask_keyboard(token) if number == len(parts) else None,
        )


@router.callback_query(F.data.startswith("ask:"), F.message)
async def handle_ask(callback: CallbackQuery, settings: Settings, state: FSMContext) -> None:
    if not settings.is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return

    token = callback.data.removeprefix("ask:")
    if not TOKEN_RE.match(token):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    if not (settings.transcripts_dir / f"{token}.txt").exists():
        await callback.answer("Расшифровка больше недоступна, пришлите ссылку заново", show_alert=True)
        return

    await state.set_state(AskState.waiting)
    await state.update_data(token=token)
    await callback.answer()
    await callback.message.answer(
        "❓ Задайте вопрос по этому видео — отвечу по тексту расшифровки.\n"
        "Например: «какие выводы сделал автор?» или «что там про цены?».\n\n"
        "Вопросов можно задать сколько угодно. Выйти — /cancel"
    )


@router.callback_query(F.data == "askend")
async def handle_ask_end(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Режим вопросов выключен")
    with contextlib.suppress(TelegramAPIError):
        await callback.message.edit_reply_markup(reply_markup=None)


@router.callback_query(F.data == "nop")
async def handle_noop(callback: CallbackQuery) -> None:
    await callback.answer()


@router.callback_query(F.data.startswith("les:"), F.message)
async def handle_lesson_start(
    callback: CallbackQuery,
    settings: Settings,
    transcriber: Transcriber,
) -> None:
    if not settings.is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа", show_alert=True)
        return

    token = callback.data.removeprefix("les:")
    if not TOKEN_RE.match(token):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    await callback.answer("Собираю конспект…")
    status = await callback.message.answer("🧩 Собираю разделы и вопросы теста…")
    try:
        lesson = await lesson_ui.load(settings, transcriber, token)
    except TranscriptionError as exc:
        await status.edit_text(f"❌ {html.escape(str(exc))}")
        return

    await status.edit_text(
        lesson_ui.section_text(lesson, 0),
        reply_markup=lesson_ui.section_keyboard(lesson, token, 0),
    )


async def _open_lesson(callback: CallbackQuery, settings: Settings, token: str) -> dict | None:
    """Достаёт готовый конспект из кэша для листалки и теста."""
    lesson = await asyncio.to_thread(lesson_ui.cached, settings, token)
    if lesson is None:
        await callback.answer("Конспект устарел, соберите его заново", show_alert=True)
        with contextlib.suppress(TelegramAPIError):
            await callback.message.edit_reply_markup(reply_markup=None)
        return None
    return lesson


async def _render(callback: CallbackQuery, text: str, keyboard: InlineKeyboardMarkup) -> None:
    try:
        await callback.message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest as exc:
        # «message is not modified» — значит, пользователь нажал ту же кнопку ещё раз
        logger.debug("Экран конспекта не изменился: %s", exc)


@router.callback_query(F.data.startswith("ls:"), F.message)
async def handle_lesson_page(callback: CallbackQuery, settings: Settings) -> None:
    parsed = _parse_callback(callback.data, "ls", 1)
    if parsed is None or not settings.is_allowed(callback.from_user.id):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    token, (index,) = parsed
    lesson = await _open_lesson(callback, settings, token)
    if lesson is None:
        return

    if not 0 <= index < len(lesson_ui.sections(lesson)):
        await callback.answer("Этого раздела больше нет", show_alert=True)
        return

    await callback.answer()
    await _render(
        callback,
        lesson_ui.section_text(lesson, index),
        lesson_ui.section_keyboard(lesson, token, index),
    )


@router.callback_query(F.data.startswith("qs:"), F.message)
async def handle_quiz_question(callback: CallbackQuery, settings: Settings) -> None:
    parsed = _parse_callback(callback.data, "qs", 2)
    if parsed is None or not settings.is_allowed(callback.from_user.id):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    token, (index, score) = parsed
    lesson = await _open_lesson(callback, settings, token)
    if lesson is None:
        return

    if not 0 <= index < len(lesson_ui.quiz(lesson)):
        await callback.answer("Вопросы закончились", show_alert=True)
        return

    await callback.answer()
    await _render(
        callback,
        lesson_ui.question_text(lesson, index, score),
        lesson_ui.question_keyboard(token, lesson, index, score),
    )


@router.callback_query(F.data.startswith("qa:"), F.message)
async def handle_quiz_answer(callback: CallbackQuery, settings: Settings) -> None:
    parsed = _parse_callback(callback.data, "qa", 3)
    if parsed is None or not settings.is_allowed(callback.from_user.id):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    token, (index, chosen, score) = parsed
    lesson = await _open_lesson(callback, settings, token)
    if lesson is None:
        return

    questions = lesson_ui.quiz(lesson)
    if not 0 <= index < len(questions) or not 0 <= chosen < len(questions[index]["options"]):
        await callback.answer("Этого вопроса больше нет", show_alert=True)
        return

    correct = chosen == questions[index]["correct_index"]
    score += int(correct)
    await callback.answer("Верно!" if correct else "Мимо")
    await _render(
        callback,
        lesson_ui.answer_text(lesson, index, chosen, score),
        lesson_ui.answer_keyboard(token, lesson, index, score),
    )


@router.callback_query(F.data.startswith("qr:"), F.message)
async def handle_quiz_result(callback: CallbackQuery, settings: Settings) -> None:
    parsed = _parse_callback(callback.data, "qr", 1)
    if parsed is None or not settings.is_allowed(callback.from_user.id):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    token, (score,) = parsed
    lesson = await _open_lesson(callback, settings, token)
    if lesson is None:
        return

    await callback.answer()
    await _render(callback, lesson_ui.result_text(lesson, score), lesson_ui.result_keyboard(token))


def _parse_callback(data: str | None, prefix: str, numbers: int) -> tuple[str, tuple[int, ...]] | None:
    """Разбирает callback вида prefix:token:число:… — всё состояние теста лежит в кнопке."""
    parts = (data or "").split(":")
    if len(parts) != numbers + 2 or parts[0] != prefix or not TOKEN_RE.match(parts[1]):
        return None
    try:
        values = tuple(int(part) for part in parts[2:])
    except ValueError:
        return None
    if any(value < 0 or value > 999 for value in values):
        return None
    return parts[1], values


async def run(settings: Settings) -> None:
    ensure_ffmpeg()
    check_js_runtime()

    session = None
    if settings.telegram_api_url:
        # Локальный telegram-bot-api отдаёт файлы до 2 ГБ и кладёт их прямо на диск
        session = AiohttpSession(api=TelegramAPIServer.from_base(settings.telegram_api_url, is_local=True))
        logger.info("Использую локальный Bot API: %s", settings.telegram_api_url)

    bot = Bot(
        token=settings.bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    transcriber = Transcriber(settings)
    queue = JobQueue(bot, settings, transcriber)
    collector = GroupCollector(bot, settings)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    dispatcher.workflow_data.update(
        settings=settings,
        queue=queue,
        transcriber=transcriber,
        collector=collector,
    )

    await asyncio.to_thread(_clean_tmp, settings.tmp_dir)
    queue.start()

    me = await bot.me()
    logger.info("Бот @%s запущен, воркеров: %s", me.username, settings.workers)

    try:
        await dispatcher.start_polling(bot, drop_pending_updates=True)
    finally:
        collector.shutdown()
        await queue.stop()
        await transcriber.close()
        await bot.session.close()


def _clean_tmp(tmp_dir: Path) -> None:
    for path in tmp_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
