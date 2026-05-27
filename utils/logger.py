"""集中式结构化日志配置"""
import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    if LOG_LEVEL.upper() != "DEBUG":
        for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("Logging configured: level=%s", LOG_LEVEL.upper())
