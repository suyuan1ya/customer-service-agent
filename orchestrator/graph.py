"""LangGraph Supervisor-Worker 状态图"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.state import AgentState
from agents.supervisor import supervisor_node
from agents.technical import technical_node
from agents.aftersales import aftersales_node
from agents.customer import customer_node


def route_by_intent(state: AgentState) -> str:
    """根据 intent 路由到对应 worker"""
    intent = state.get("intent", "customer")
    routing = {
        "technical": "technical_worker",
        "aftersales": "aftersales_worker",
        "customer": "customer_worker",
        "escalate": "finish",
    }
    return routing.get(intent, "customer_worker")


def should_escalate(state: AgentState) -> str:
    if state.get("needs_escalation", False):
        return "finish"
    return "supervisor"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)  # type: ignore
    graph.add_node("technical_worker", technical_node)  # type: ignore
    graph.add_node("aftersales_worker", aftersales_node)  # type: ignore
    graph.add_node("customer_worker", customer_node)  # type: ignore

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_by_intent, {
        "technical_worker": "technical_worker",
        "aftersales_worker": "aftersales_worker",
        "customer_worker": "customer_worker",
        "finish": END,
    })

    # worker 完成后结束（简单模式：worker 直接输出最终回复）
    graph.add_edge("technical_worker", END)
    graph.add_edge("aftersales_worker", END)
    graph.add_edge("customer_worker", END)

    return graph.compile(checkpointer=MemorySaver())


import threading

# 线程安全单例
_agent = None
_lock = threading.Lock()


def get_agent():
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                _agent = build_graph()
    return _agent
