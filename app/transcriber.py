from __future__ import annotations

import asyncio
import json
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

QA_INPUT_LIMIT = 100_000

SYSTEM_QA = (
    "Ты отвечаешь на вопросы по расшифровке видео. Пиши по-русски, опирайся только на текст "
    "расшифровки. Если ответа в тексте нет — прямо скажи, что в видео об этом не говорится."
)

PROMPT_QA = """Расшифровка видео:
{text}

---
Вопрос пользователя: {question}

Ответь по существу, без вступлений. Если в расшифровке есть подходящая цитата — приведи её."""

PROMPT_QA_MERGE = """Вопрос пользователя: {question}

Ниже ответы, полученные по разным фрагментам одного видео. Собери из них один связный ответ:
убери повторы и упоминания о том, что в каком-то фрагменте ответа не было.

{text}"""

LESSON_INPUT_LIMIT = 40_000
LESSON_SECTIONS = 5
LESSON_QUESTIONS = 5

SYSTEM_LESSON = (
    "Ты методист, который превращает расшифровку видео в учебный материал. "
    "Пиши по-русски, опирайся только на содержание расшифровки. "
    "Возвращай строго валидный JSON без markdown-разметки и пояснений вокруг."
)

PROMPT_LESSON = """Сделай интерактивный конспект видео и мини-тест по его содержанию.

Верни JSON такой структуры:
{{
  "title": "краткое название темы видео",
  "sections": [
    {{
      "heading": "заголовок раздела",
      "content": "2-4 предложения, раскрывающих раздел",
      "key_points": ["короткий тезис", "ещё тезис"]
    }}
  ],
  "quiz": [
    {{
      "question": "вопрос по содержанию видео",
      "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"],
      "correct_index": 0,
      "explanation": "одно предложение, почему этот ответ верный"
    }}
  ]
}}

Требования:
- разделов: от 3 до {sections}, они должны идти в логике видео;
- вопросов: ровно {questions}, ровно по 4 варианта в каждом;
- проверяй понимание содержания, а не дословные формулировки;
- неверные варианты должны быть правдоподобными, но однозначно неверными по тексту;
- correct_index — индекс верного варианта с нуля;
- никакого текста вне JSON.

Расшифровка:
{text}"""


class TranscriptionError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def _normalize_lesson(data: dict) -> dict:
    """Приводит ответ модели к виду, на который может рассчитывать бот."""
    sections: list[dict] = []
    for item in data.get("sections") or []:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or "").strip()
        content = str(item.get("content") or "").strip()
        points = [str(point).strip() for point in item.get("key_points") or [] if str(point).strip()]
        if heading and (content or points):
            sections.append({"heading": heading, "content": content, "key_points": points[:6]})

    quiz: list[dict] = []
    for item in data.get("quiz") or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        options = [str(option).strip() for option in item.get("options") or [] if str(option).strip()]
        try:
            correct = int(item.get("correct_index"))
        except (TypeError, ValueError):
            continue
        if question and 2 <= len(options) <= 6 and 0 <= correct < len(options):
            quiz.append(
                {
                    "question": question,
                    "options": options,
                    "correct_index": correct,
                    "explanation": str(item.get("explanation") or "").strip(),
                }
            )

    title = str(data.get("title") or "").strip() or "Конспект видео"
    return {"title": title, "sections": sections[:LESSON_SECTIONS], "quiz": quiz[:LESSON_QUESTIONS]}


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

    async def answer(self, question: str, text: str) -> str:
        """Отвечает на вопрос пользователя по тексту расшифровки."""
        parts = split_text(text, QA_INPUT_LIMIT)
        try:
            if len(parts) == 1:
                return await self._complete(
                    PROMPT_QA.format(text=parts[0], question=question), system=SYSTEM_QA
                )

            partials = await asyncio.gather(
                *(
                    self._complete(PROMPT_QA.format(text=part, question=question), system=SYSTEM_QA)
                    for part in parts
                )
            )
            merged = "\n\n---\n\n".join(partials)
            return await self._complete(
                PROMPT_QA_MERGE.format(text=merged, question=question), system=SYSTEM_QA
            )
        except OpenAIError as exc:
            logger.exception("Не удалось ответить на вопрос")
            raise TranscriptionError(_humanize_api_error(exc)) from exc

    async def build_lesson(self, text: str) -> dict:
        """Собирает интерактивный конспект: разделы и тест с вариантами ответов."""
        source = text
        if len(source) > LESSON_INPUT_LIMIT:
            # Длинную расшифровку сначала ужимаем — иначе тест соберётся только по началу видео
            source = await self.summarize(source)

        prompt = PROMPT_LESSON.format(
            text=source[:LESSON_INPUT_LIMIT], sections=LESSON_SECTIONS, questions=LESSON_QUESTIONS
        )
        try:
            raw = await self._complete(prompt, system=SYSTEM_LESSON, as_json=True)
        except OpenAIError as exc:
            logger.exception("Не удалось составить интерактивный конспект")
            raise TranscriptionError(_humanize_api_error(exc)) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Модель вернула не JSON: %s", raw[:500])
            raise TranscriptionError("Модель вернула конспект в непонятном виде, попробуйте ещё раз.") from exc

        lesson = _normalize_lesson(data)
        if not lesson["sections"] or not lesson["quiz"]:
            raise TranscriptionError(
                "Не получилось собрать конспект с тестом по этому видео — возможно, в нём мало содержания."
            )
        return lesson

    async def _complete(self, prompt: str, system: str = SYSTEM_SUMMARY, as_json: bool = False) -> str:
        kwargs: dict = {
            "model": self._settings.summary_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if as_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
