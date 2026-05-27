import json
import logging
import threading
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config import BASE_URL, DASHSCOPE_API_KEY, QWEN_MODEL
from utils.retry import llm_retry

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """你是客服系统的智能调度专家。根据用户消息判断应该转给哪个部门。

分类规则:
- technical: 产品技术问题、Bug报错、功能使用、配置方法、参数规格、兼容性
- aftersales: 退换货、保修、维修、投诉、退款、质量问题、物流异常
- customer: 订单查询、物流跟踪、账户问题、修改信息、会员咨询、常规咨询
- escalate: 用户明确要求人工、投诉升级、情绪激烈、法律威胁、或需求超出客服能力

你必须只输出一个 JSON 对象，不要包含其他内容:
{"intent": "<technical|aftersales|customer|escalate>", "reason": "<简要原因>"}"""

_supervisor_llm = None
_supervisor_lock = threading.Lock()


def get_supervisor_llm():
    global _supervisor_llm
    if _supervisor_llm is None:
        with _supervisor_lock:
            if _supervisor_llm is None:
                _supervisor_llm = ChatOpenAI(
                    model=QWEN_MODEL,
                    api_key=DASHSCOPE_API_KEY,
                    base_url=BASE_URL,
                    temperature=0.1,
                )
    return _supervisor_llm


def _parse_intent(raw: str) -> dict:
    """从 LLM 原始输出中稳健地提取 JSON"""
    content = raw.strip()

    # 去掉 markdown 代码块包裹
    if content.startswith("```"):
        lines = content.split("\n")
        # 去掉首行 ```json 或 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 去掉末行 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取第一个 { ... } 块
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                logger.warning("Supervisor JSON 解析失败，回退到 customer。原始输出: %s", raw[:200])
                return {"intent": "customer", "reason": "fallback: JSON parse error"}
        else:
            logger.warning("Supervisor 输出中未找到 JSON 对象，回退到 customer。原始输出: %s", raw[:200])
            return {"intent": "customer", "reason": "fallback: no JSON found"}

    # 验证 intent 字段有效性
    valid_intents = {"technical", "aftersales", "customer", "escalate"}
    intent = result.get("intent", "customer")
    if intent not in valid_intents:
        logger.warning("Supervisor 返回了无效的 intent: %s，回退到 customer", intent)
        intent = "customer"

    return {"intent": intent, "reason": result.get("reason", "fallback")}


async def supervisor_node(state):  # type: ignore
    messages = state["messages"]
    last_msg = messages[-1].content if messages else ""

    logger.info("Supervisor classifying intent for message: %s", last_msg[:80])

    llm = get_supervisor_llm()

    @llm_retry()
    async def _invoke():
        return await llm.ainvoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"用户消息: {last_msg}")
        ])

    response = await _invoke()

    result = _parse_intent(response.content)
    intent = result["intent"]

    logger.info("Supervisor intent: %s (reason: %s)", intent, result.get("reason", "unknown"))

    return {
        "intent": intent,
        "worker_result": "",
        "needs_escalation": intent == "escalate",
        "final_response": "",
    }
