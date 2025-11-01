from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

_LEVEL = os.getenv("ROOKEEN_LOG_LEVEL", "INFO").upper()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("trace_id", "run_id"):
            val = getattr(record, key, None)
            if val:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level or _LEVEL)
    logger.propagate = False
    return logger


def new_trace_id() -> str:
    return uuid.uuid4().hex
