from __future__ import annotations

import asyncio
import logging
import sys

from app.audio import AudioError
from app.bot import run
from app.config import load_settings


def main() -> int:
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)

    try:
        asyncio.run(run(settings))
    except AudioError as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("Остановлено пользователем")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
