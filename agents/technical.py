import logging
import threading
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.knowledge_base import search_faq_tool
from config import BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL
from utils.retry import llm_retry
from utils.conversation import trim_history

logger = logging.getLogger(__name__)

TECHNICAL_PROMPT = """你是技术支持专家，负责解答产品技术问题。

职责范围:
- 产品功能使用指导
- Bug 问题排查
- 技术参数和规格说明
- 配置方法
- 兼容性问题

答案要求:
1. 涉及产品文档、技术规格、已知问题等事实信息时，使用 search_faq_tool 查询知识库
2. 给出具体可操作的建议和步骤
3. 如果问题超出技术支持范围（如退换货、投诉），建议转到售后服务"""

TOOLS = [search_faq_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

_tech_llm = None
_tech_lock = threading.Lock()


def get_tech_llm():
    global _tech_llm
    if _tech_llm is None:
        with _tech_lock:
            if _tech_llm is None:
                _tech_llm = ChatOpenAI(
                    model=QWEN_MODEL,
                    api_key=DASHSCOPE_API_KEY,
                    base_url=BASE_URL,
                    temperature=0.3,
                )
    return _tech_llm


async def technical_node(state):  # type: ignore
    messages = trim_history(state["messages"])
    logger.info("Technical worker processing request with %d messages", len(messages))

    llm = get_tech_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    msgs = [SystemMessage(content=TECHNICAL_PROMPT), *messages]

    @llm_retry()
    async def _invoke(msgs):
        return await llm_with_tools.ainvoke(msgs)

    response = await _invoke(msgs)

    iteration = 0
    while response.tool_calls and iteration < 5:
        tool_messages = []
        for tc in response.tool_calls:
            logger.info("Technical worker tool call: %s(%s)", tc["name"], tc.get("args", {}))
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = await tool_fn.ainvoke(tc["args"])
            else:
                result = f"未知工具: {tc['name']}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        msgs.extend([response, *tool_messages])
        response = await _invoke(msgs)
        iteration += 1

    logger.info("Technical worker completed, response length: %d", len(response.content) if response.content else 0)
    return {
        "worker_result": response.content,
        "final_response": response.content,
    }
