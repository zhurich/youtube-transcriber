from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import mimetypes
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.audio import (
    AudioError,
    PreparedAudio,
    Progress,
    check_js_runtime,
    ensure_ffmpeg,
    find_youtube_url,
    get_video_info,
    prepare_audio_file,
    prepare_youtube_audio,
)
from app.config import Settings
from app.transcriber import Transcriber, TranscriptionError
from app.utils import format_hms, periodic, split_text

logger = logging.getLogger(__name__)
router = Router()
router.message.filter(F.from_user)

MESSAGE_TEXT_LIMIT = 3500
MAX_JOBS_PER_USER = 3
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

MEDIA_EXTENSIONS = frozenset(
    {
        ".3gp", ".aac", ".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov", ".mp3", ".mp4",
        ".mpeg", ".mpg", ".oga", ".ogg", ".ogv", ".opus", ".wav", ".webm", ".wma",
    }
)

HELP_TEXT = (
    "Пришлите ссылку на видео с YouTube или сам файл — видео, аудио, голосовое "
    "или видеосообщение. Я извлеку звук, расшифрую его через OpenAI и верну текст. "
    "Под расшифровкой будет кнопка для краткого конспекта.\n\n"
    "Команды:\n"
    "/help — эта справка\n"
    "/id — ваш Telegram ID (нужен для списка доступа)"
)


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


def summary_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="📝 Сделать конспект", callback_data=f"sum:{token}")]]
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
            meta = await self._resolve_meta(job, status)
            transcript_path = self._settings.transcripts_dir / f"{meta.token}.txt"

            if transcript_path.exists():
                await status.set(f"{meta.header()}\n\n♻️ Нашёл готовую расшифровку в кэше.")
                text = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
            else:
                prepared = await self._fetch_audio(job, meta, workdir, status)
                if not meta.duration:
                    meta.duration = int(prepared.duration)
                    self._check_duration(meta.duration)
                text = await self._run_transcription(prepared, meta, status)
                await asyncio.to_thread(transcript_path.write_text, text, encoding="utf-8")

            elapsed = format_hms(time.monotonic() - started)
            await status.set(f"{meta.header()}\n\n✅ Готово за {elapsed}")
            await self._send_transcript(job.chat_id, meta.title, text, meta.token)

        except (AudioError, TranscriptionError) as exc:
            await status.set(f"❌ {html.escape(str(exc))}")
        finally:
            await asyncio.to_thread(shutil.rmtree, workdir, True)

    async def _resolve_meta(self, job: Job, status: Status) -> Meta:
        if job.media:
            self._check_duration(job.media.duration)
            return Meta(
                title=job.media.title,
                duration=job.media.duration,
                token=self._token(job.media.unique_id),
                icon="🎧",
            )

        await status.set("🔎 Читаю информацию о видео…")
        info = await get_video_info(job.url or "", self._settings)
        self._check_duration(info.duration)
        return Meta(
            title=info.title,
            duration=info.duration,
            token=self._token(info.video_id or uuid.uuid4().hex),
            icon="🎬",
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

    async def _fetch_audio(self, job: Job, meta: Meta, workdir: Path, status: Status) -> PreparedAudio:
        header = meta.header()

        if job.media:
            await status.set(f"{header}\n\n⬇️ Забираю файл из Telegram…")
            source = await self._download_media(job.media, workdir)
            await status.set(f"{header}\n\n🎚 Извлекаю звук…")
            return await prepare_audio_file(source, workdir)

        progress = Progress()
        await status.set(f"{header}\n\n⬇️ Скачиваю аудио…")
        async with periodic(lambda: status.set(f"{header}\n\n⬇️ Скачиваю аудио…{progress.as_suffix()}")):
            return await prepare_youtube_audio(job.url or "", workdir, self._settings, progress)

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

    async def _run_transcription(self, prepared: PreparedAudio, meta: Meta, status: Status) -> str:
        total = len(prepared.chunks)
        state = {"done": 0}

        def note(done: int, _total: int) -> None:
            state["done"] = done

        def transcribe_status() -> str:
            header = meta.header()
            if total == 1:
                return f"{header}\n\n🧠 Расшифровываю…"
            return f"{header}\n\n🧠 Расшифровываю: {state['done']}/{total} частей"

        await status.set(transcribe_status())
        async with periodic(lambda: status.set(transcribe_status())):
            return await self._transcriber.transcribe(prepared.chunks, on_progress=note)

    async def _send_transcript(self, chat_id: int, title: str, text: str, token: str) -> None:
        keyboard = summary_keyboard(token)
        if len(text) <= MESSAGE_TEXT_LIMIT:
            await self._bot.send_message(chat_id, text, parse_mode=None, reply_markup=keyboard)
            return

        filename = re.sub(r"[^\w\s.-]", "", title, flags=re.UNICODE).strip()[:60] or "transcript"
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
    settings: Settings,
    queue: JobQueue,
    *,
    url: str | None = None,
    media: MediaSource | None = None,
) -> None:
    if queue.jobs_of(message.from_user.id) >= MAX_JOBS_PER_USER:
        await message.answer(f"У вас уже {MAX_JOBS_PER_USER} задачи в работе. Дождитесь их завершения.")
        return

    status_message = await message.answer("⏳ Задача принята…")
    position = await queue.submit(
        Job(
            chat_id=message.chat.id,
            user_id=message.from_user.id,
            status_message_id=status_message.message_id,
            url=url,
            media=media,
        )
    )
    if position > settings.workers:
        with contextlib.suppress(TelegramBadRequest):
            await status_message.edit_text(f"⏳ В очереди, позиция {position}")


@router.message(F.video | F.audio | F.voice | F.video_note | F.document)
async def handle_media(message: Message, settings: Settings, queue: JobQueue) -> None:
    if not settings.is_allowed(message.from_user.id):
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

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

    await _enqueue(message, settings, queue, media=media)


@router.message(F.text)
async def handle_link(message: Message, settings: Settings, queue: JobQueue) -> None:
    if not settings.is_allowed(message.from_user.id):
        await message.answer("Нет доступа. Попросите владельца бота добавить ваш ID из /id в ALLOWED_USER_IDS.")
        return

    url = find_youtube_url(message.text)
    if not url:
        await message.answer(
            "Не вижу ссылку на YouTube. Пришлите ссылку вида https://youtu.be/… "
            "или отправьте видео либо аудио файлом."
        )
        return

    await _enqueue(message, settings, queue, url=url)


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
    with contextlib.suppress(TelegramBadRequest):
        await callback.bot.edit_message_reply_markup(
            chat_id=chat_id, message_id=callback.message.message_id, reply_markup=None
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

    for part in split_text(f"📝 Конспект\n\n{summary}", MESSAGE_TEXT_LIMIT):
        await callback.bot.send_message(chat_id, part, parse_mode=None)


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

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    dispatcher.workflow_data.update(settings=settings, queue=queue, transcriber=transcriber)

    await asyncio.to_thread(_clean_tmp, settings.tmp_dir)
    queue.start()

    me = await bot.me()
    logger.info("Бот @%s запущен, воркеров: %s", me.username, settings.workers)

    try:
        await dispatcher.start_polling(bot, drop_pending_updates=True)
    finally:
        await queue.stop()
        await transcriber.close()
        await bot.session.close()


def _clean_tmp(tmp_dir: Path) -> None:
    for path in tmp_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
