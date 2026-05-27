import logging
import threading
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.knowledge_base import search_faq_tool
from tools.order_tools import query_order, query_logistics
from config import BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL
from utils.retry import llm_retry
from utils.conversation import trim_history

logger = logging.getLogger(__name__)

CUSTOMER_PROMPT = """你是客户服务专员，负责处理客户日常咨询。

职责范围:
- 订单查询和物流跟踪
- 账户注册/登录问题
- 会员权益咨询
- 修改订单/地址/联系方式
- 常规业务咨询

答案要求:
1. 涉及订单详情时，使用 query_order 查询；涉及物流进度时，使用 query_logistics 查询
2. 涉及会员权益、业务流程等事实性问题时，使用 search_faq_tool 查询知识库
3. 友好热情，用精确的信息回答
4. 如果用户提供了订单号（如 DD20240512001），主动查询订单和物流信息"""

TOOLS = [search_faq_tool, query_order, query_logistics]
TOOL_MAP = {t.name: t for t in TOOLS}

_customer_llm = None
_customer_lock = threading.Lock()


def get_customer_llm():
    global _customer_llm
    if _customer_llm is None:
        with _customer_lock:
            if _customer_llm is None:
                _customer_llm = ChatOpenAI(
                    model=QWEN_MODEL,
                    api_key=DASHSCOPE_API_KEY,
                    base_url=BASE_URL,
                    temperature=0.3,
                )
    return _customer_llm


async def customer_node(state):  # type: ignore
    messages = trim_history(state["messages"])
    logger.info("Customer worker processing request with %d messages", len(messages))

    llm = get_customer_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    msgs = [SystemMessage(content=CUSTOMER_PROMPT), *messages]

    @llm_retry()
    async def _invoke(msgs):
        return await llm_with_tools.ainvoke(msgs)

    response = await _invoke(msgs)

    iteration = 0
    while response.tool_calls and iteration < 5:
        tool_messages = []
        for tc in response.tool_calls:
            logger.info("Customer worker tool call: %s(%s)", tc["name"], tc.get("args", {}))
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = await tool_fn.ainvoke(tc["args"])
            else:
                result = f"未知工具: {tc['name']}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        msgs.extend([response, *tool_messages])
        response = await _invoke(msgs)
        iteration += 1

    logger.info("Customer worker completed, response length: %d", len(response.content) if response.content else 0)
    return {
        "worker_result": response.content,
        "final_response": response.content,
    }
