"""LLM 调用重试：指数退避，自动跳过 4xx 鉴权错误"""
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 2
RETRY_MAX_WAIT = 30


def llm_retry():
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "LLM 调用重试 %d/%d，错误: %s",
            rs.attempt_number, RETRY_ATTEMPTS,
            rs.outcome.exception() if rs.outcome else "unknown",
        ),
        reraise=True,
    )
