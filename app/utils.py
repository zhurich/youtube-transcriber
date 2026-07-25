from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Awaitable

logger = logging.getLogger(__name__)


def format_hms(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def split_text(text: str, limit: int) -> list[str]:
    """Режет текст на части не длиннее limit, стараясь не разрывать абзацы и предложения."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
        if cut < limit // 2:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


@contextlib.asynccontextmanager
async def periodic(action: Callable[[], Awaitable[None]], interval: float = 6.0) -> AsyncIterator[None]:
    """Вызывает action раз в interval секунд, пока выполняется тело блока."""

    async def loop() -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await action()
            except Exception:  # noqa: BLE001 - обновление статуса не должно ломать задачу
                logger.debug("Не удалось обновить статус", exc_info=True)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
