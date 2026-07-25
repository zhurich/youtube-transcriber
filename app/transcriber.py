from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

from openai import APIStatusError, AsyncOpenAI, OpenAIError

from app.audio import AudioChunk
from app.config import Settings
from app.utils import format_hms, split_text

logger = logging.getLogger(__name__)

SUMMARY_INPUT_LIMIT = 60_000

SYSTEM_SUMMARY = (
    "Ты редактор, который делает конспекты расшифровок видео. "
    "Пиши по-русски, только по содержанию текста, без домыслов."
)

PROMPT_SUMMARY = """Составь конспект расшифровки видео.

Формат:
1. Одно предложение о том, чему посвящено видео.
2. Основные тезисы списком (5-12 пунктов), каждый — законченная мысль.
3. Выводы или практические рекомендации, если они есть в тексте.

Расшифровка:
{text}"""

PROMPT_MERGE = """Ниже конспекты последовательных фрагментов одного видео.
Собери их в один связный конспект в том же формате, убери повторы.

{text}"""


class TranscriptionError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def _humanize_api_error(error: Exception) -> str:
    if isinstance(error, APIStatusError):
        if error.status_code == 401:
            return "OpenAI отклонил ключ API. Проверьте OPENAI_API_KEY."
        if error.status_code == 429:
            return "OpenAI вернул 429: закончилась квота или превышен лимит запросов."
        if error.status_code == 413:
            return "Файл слишком большой для OpenAI. Уменьшите длительность видео."
        if error.status_code >= 500:
            return "OpenAI временно недоступен, попробуйте позже."
        return f"OpenAI вернул ошибку {error.status_code}."
    return f"Ошибка обращения к OpenAI: {error}"


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=600.0)
        self._semaphore = asyncio.Semaphore(settings.transcribe_concurrency)

    @property
    def _with_timestamps(self) -> bool:
        return self._settings.timestamps and self._settings.transcribe_model == "whisper-1"

    async def close(self) -> None:
        await self._client.close()

    async def transcribe(
        self,
        chunks: Sequence[AudioChunk],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        texts = await self.transcribe_groups([chunks], on_progress=on_progress)
        if not texts[0]:
            raise TranscriptionError("Модель вернула пустую расшифровку — возможно, в видео нет речи.")
        return texts[0]

    async def transcribe_groups(
        self,
        groups: Sequence[Sequence[AudioChunk]],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        """Расшифровывает несколько записей одним пакетом, возвращая текст каждой отдельно.

        Все части всех записей идут в OpenAI через общий семафор, поэтому пакет из
        коротких голосовых обрабатывается так же параллельно, как одно длинное видео.
        """
        done = 0
        total = sum(len(group) for group in groups)
        lock = asyncio.Lock()

        async def run(chunk: AudioChunk) -> str:
            nonlocal done
            async with self._semaphore:
                text = await self._transcribe_chunk(chunk)
            async with lock:
                done += 1
                if on_progress:
                    on_progress(done, total)
            return text

        try:
            results = await asyncio.gather(*(run(chunk) for group in groups for chunk in group))
        except OpenAIError as exc:
            logger.exception("Транскрибация не удалась")
            raise TranscriptionError(_humanize_api_error(exc)) from exc

        texts: list[str] = []
        position = 0
        for group in groups:
            parts = results[position : position + len(group)]
            texts.append("\n\n".join(part for part in parts if part).strip())
            position += len(group)

        if not any(texts):
            raise TranscriptionError("Модель вернула пустую расшифровку — возможно, в записях нет речи.")
        return texts

    async def _transcribe_chunk(self, chunk: AudioChunk) -> str:
        data = await asyncio.to_thread(chunk.path.read_bytes)
        kwargs: dict = {
            "model": self._settings.transcribe_model,
            "file": (chunk.path.name, data, "audio/mpeg"),
            "response_format": "verbose_json" if self._with_timestamps else "text",
        }
        if self._settings.transcribe_language:
            kwargs["language"] = self._settings.transcribe_language

        response = await self._client.audio.transcriptions.create(**kwargs)

        if self._with_timestamps:
            segments = getattr(response, "segments", None) or []
            lines = [
                f"[{format_hms(chunk.offset + (segment.start or 0))}] {(segment.text or '').strip()}"
                for segment in segments
                if (segment.text or "").strip()
            ]
            if lines:
                return "\n".join(lines)

        if isinstance(response, str):
            return response.strip()
        return (getattr(response, "text", "") or "").strip()

    async def summarize(self, text: str) -> str:
        parts = split_text(text, SUMMARY_INPUT_LIMIT)
        try:
            if len(parts) == 1:
                return await self._complete(PROMPT_SUMMARY.format(text=parts[0]))

            partials = await asyncio.gather(
                *(self._complete(PROMPT_SUMMARY.format(text=part)) for part in parts)
            )
            merged = "\n\n---\n\n".join(partials)
            return await self._complete(PROMPT_MERGE.format(text=merged))
        except OpenAIError as exc:
            logger.exception("Не удалось составить конспект")
            raise TranscriptionError(_humanize_api_error(exc)) from exc

    async def _complete(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._settings.summary_model,
            messages=[
                {"role": "system", "content": SYSTEM_SUMMARY},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
