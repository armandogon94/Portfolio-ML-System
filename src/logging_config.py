"""Structured logging configuration.

Provides JSON formatter (for Docker/production) and text formatter (for terminal).
Controlled by LOG_FORMAT env var: 'json' or 'text' (default: 'text').
Log level controlled by LOG_LEVEL env var (default: 'INFO').
"""

import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Output keys: timestamp, level, message, logger.
    Exception info and extra fields are included when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Include any extra fields attached to the record
        standard_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_keys and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def setup_logging(
    log_format: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure the root logger with the appropriate formatter.

    Args:
        log_format: 'json' or 'text'. If None, reads LOG_FORMAT env var (default: 'text').
        handler:    If provided, configures this handler instead of creating a new StreamHandler.
                    Useful for testing (pass a StringIO-backed StreamHandler).
    """
    if log_format is None:
        log_format = os.environ.get("LOG_FORMAT", "text").lower()

    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    if handler is not None:
        handler.setFormatter(formatter)
    else:
        # Only add a new StreamHandler if root logger has no handlers yet
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(formatter)
            root.addHandler(h)
        else:
            for h in logging.getLogger().handlers:
                h.setFormatter(formatter)

    logging.getLogger().setLevel(log_level)
