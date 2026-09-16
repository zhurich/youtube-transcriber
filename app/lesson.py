"""Интерактивный конспект: разделы с навигацией и мини-тест по содержанию видео.

Состояние теста целиком живёт в callback_data кнопок (номер вопроса и накопленный
счёт), поэтому тест переживает перезапуск бота и не занимает память.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
from pathlib import Path

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.config import Settings
from app.transcriber import Transcriber, TranscriptionError

logger = logging.getLogger(__name__)

LETTERS = "ABCDEF"


def path_for(settings: Settings, token: str) -> Path:
    return settings.transcripts_dir / f"{token}.lesson.json"


def cached(settings: Settings, token: str) -> dict | None:
    path = path_for(settings, token)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Повреждённый кэш конспекта: %s", path)
        return None


async def load(settings: Settings, transcriber: Transcriber, token: str) -> dict:
    """Отдаёт готовый конспект из кэша или собирает новый по расшифровке."""
    lesson = await asyncio.to_thread(cached, settings, token)
    if lesson:
        return lesson

    transcript_path = settings.transcripts_dir / f"{token}.txt"
    if not transcript_path.exists():
        raise TranscriptionError("Расшифровка больше недоступна, пришлите ссылку заново.")

    text = await asyncio.to_thread(transcript_path.read_text, encoding="utf-8")
    lesson = await transcriber.build_lesson(text)
    await asyncio.to_thread(
        path_for(settings, token).write_text,
        json.dumps(lesson, ensure_ascii=False),
        encoding="utf-8",
    )
    return lesson


def sections(lesson: dict) -> list[dict]:
    return lesson.get("sections") or []


def quiz(lesson: dict) -> list[dict]:
    return lesson.get("quiz") or []


def section_text(lesson: dict, index: int) -> str:
    section = sections(lesson)[index]
    lines = [
        f"🧩 <b>{html.escape(str(lesson.get('title') or 'Конспект видео'))}</b>",
        "",
        f"<b>{index + 1}. {html.escape(section['heading'])}</b>",
    ]
    if section.get("content"):
        lines += ["", html.escape(section["content"])]
    for point in section.get("key_points") or []:
        lines.append(f"• {html.escape(point)}")
    return "\n".join(lines)


def section_keyboard(lesson: dict, token: str, index: int) -> InlineKeyboardMarkup:
    total = len(sections(lesson))
    nav: list[InlineKeyboardButton] = []
    if index > 0:
        nav.append(InlineKeyboardButton(text="◀️ Назад", callback_data=f"ls:{token}:{index - 1}"))
    nav.append(InlineKeyboardButton(text=f"{index + 1}/{total}", callback_data="nop"))
    if index < total - 1:
        nav.append(InlineKeyboardButton(text="Вперёд ▶️", callback_data=f"ls:{token}:{index + 1}"))

    rows = [nav]
    questions = len(quiz(lesson))
    if questions:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🧪 Пройти тест · {questions} вопросов",
                    callback_data=f"qs:{token}:0:0",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _question_body(lesson: dict, index: int) -> list[str]:
    item = quiz(lesson)[index]
    lines = [
        f"🧪 <b>Вопрос {index + 1} из {len(quiz(lesson))}</b>",
        "",
        html.escape(item["question"]),
        "",
    ]
    lines += [
        f"<b>{LETTERS[number]}.</b> {html.escape(option)}"
        for number, option in enumerate(item["options"])
    ]
    return lines


def question_text(lesson: dict, index: int, score: int) -> str:
    lines = _question_body(lesson, index)
    if index:
        lines += ["", f"<i>Счёт: {score} из {index}</i>"]
    return "\n".join(lines)


def question_keyboard(token: str, lesson: dict, index: int, score: int) -> InlineKeyboardMarkup:
    options = quiz(lesson)[index]["options"]
    buttons = [
        InlineKeyboardButton(text=LETTERS[number], callback_data=f"qa:{token}:{index}:{number}:{score}")
        for number in range(len(options))
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def answer_text(lesson: dict, index: int, chosen: int, score: int) -> str:
    item = quiz(lesson)[index]
    correct = item["correct_index"]
    lines = [
        f"🧪 <b>Вопрос {index + 1} из {len(quiz(lesson))}</b>",
        "",
        html.escape(item["question"]),
        "",
    ]
    for number, option in enumerate(item["options"]):
        mark = "✅" if number == correct else ("❌" if number == chosen else "▫️")
        lines.append(f"{mark} <b>{LETTERS[number]}.</b> {html.escape(option)}")

    verdict = "Верно!" if chosen == correct else "Неверно."
    lines += ["", f"<b>{verdict}</b>"]
    if item.get("explanation"):
        lines.append(html.escape(item["explanation"]))
    lines += ["", f"<i>Счёт: {score} из {index + 1}</i>"]
    return "\n".join(lines)


def answer_keyboard(token: str, lesson: dict, index: int, score: int) -> InlineKeyboardMarkup:
    if index + 1 < len(quiz(lesson)):
        button = InlineKeyboardButton(
            text="Следующий вопрос ▶️", callback_data=f"qs:{token}:{index + 1}:{score}"
        )
    else:
        button = InlineKeyboardButton(text="🏁 Показать результат", callback_data=f"qr:{token}:{score}")
    return InlineKeyboardMarkup(inline_keyboard=[[button]])


def result_text(lesson: dict, score: int) -> str:
    total = len(quiz(lesson))
    percent = round(score / total * 100) if total else 0
    if percent >= 80:
        verdict = "🏆 Отличный результат — материал усвоен."
    elif percent >= 50:
        verdict = "👍 Неплохо, но часть деталей стоит перечитать."
    else:
        verdict = "📚 Похоже, видео стоит пересмотреть."

    filled = round(percent / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return "\n".join(
        [
            f"🏁 <b>Тест пройден: {score} из {total}</b>",
            "",
            f"{bar} {percent}%",
            "",
            verdict,
        ]
    )


def result_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔁 Пройти заново", callback_data=f"qs:{token}:0:0")],
            [InlineKeyboardButton(text="📖 Вернуться к конспекту", callback_data=f"ls:{token}:0")],
        ]
    )
