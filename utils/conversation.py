"""多轮对话历史管理"""
import logging
from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_CONTEXT_MESSAGES = 20


def trim_history(messages: list[BaseMessage], max_messages: int = MAX_CONTEXT_MESSAGES) -> list[BaseMessage]:
    """保留最近 N 条消息，SystemMessage 始终保留在最前。"""
    if len(messages) <= max_messages:
        return list(messages)

    system_msgs = []
    idx = 0
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_msgs.append(msg)
            idx += 1
        else:
            break

    non_system = messages[idx:]
    if len(system_msgs) + len(non_system) <= max_messages:
        return list(messages)

    keep_count = max_messages - len(system_msgs)
    trimmed = system_msgs + non_system[-keep_count:]
    logger.debug("Trimmed conversation history: %d → %d messages", len(messages), len(trimmed))
    return trimmed
