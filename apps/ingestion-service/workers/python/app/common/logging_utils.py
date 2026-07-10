from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    return logging.getLogger("pilot_worker")


def log_json(logger: logging.Logger, **payload: object) -> None:
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    logger.info(json.dumps(payload, sort_keys=True, ensure_ascii=True))
