import logging
import threading
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.knowledge_base import search_faq_tool
from tools.ticket_tools import create_ticket
from config import BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL
from utils.retry import llm_retry
from utils.conversation import trim_history

logger = logging.getLogger(__name__)

AFTERSALES_PROMPT = """你是售后服务专员，负责处理售后相关问题。

职责范围:
- 退换货政策和流程
- 保修查询和维修进度
- 退款处理
- 投诉受理
- 物流异常处理

答案要求:
1. 涉及售后政策、退换货规则时，使用 search_faq_tool 查询知识库
2. 涉及投诉、退换货、保修、质量问题需要正式记录时，使用 create_ticket 创建工单
3. 给出清晰的处理流程和时间预期
4. 告知用户已记录并给出处理时限"""

TOOLS = [search_faq_tool, create_ticket]
TOOL_MAP = {t.name: t for t in TOOLS}

_aftersales_llm = None
_aftersales_lock = threading.Lock()


def get_aftersales_llm():
    global _aftersales_llm
    if _aftersales_llm is None:
        with _aftersales_lock:
            if _aftersales_llm is None:
                _aftersales_llm = ChatOpenAI(
                    model=QWEN_MODEL,
                    api_key=DASHSCOPE_API_KEY,
                    base_url=BASE_URL,
                    temperature=0.3,
                )
    return _aftersales_llm


async def aftersales_node(state):  # type: ignore
    messages = trim_history(state["messages"])
    logger.info("Aftersales worker processing request with %d messages", len(messages))

    llm = get_aftersales_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    msgs = [SystemMessage(content=AFTERSALES_PROMPT), *messages]

    @llm_retry()
    async def _invoke(msgs):
        return await llm_with_tools.ainvoke(msgs)

    response = await _invoke(msgs)

    iteration = 0
    while response.tool_calls and iteration < 5:
        tool_messages = []
        for tc in response.tool_calls:
            logger.info("Aftersales worker tool call: %s(%s)", tc["name"], tc.get("args", {}))
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = await tool_fn.ainvoke(tc["args"])
            else:
                result = f"未知工具: {tc['name']}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        msgs.extend([response, *tool_messages])
        response = await _invoke(msgs)
        iteration += 1

    logger.info("Aftersales worker completed, response length: %d", len(response.content) if response.content else 0)
    return {
        "worker_result": response.content,
        "final_response": response.content,
    }
