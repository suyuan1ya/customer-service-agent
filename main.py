"""智能客服多 Agent 协作系统 — FastAPI 入口"""

import json
import logging
import time
import uuid
from collections import defaultdict

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from langchain_core.messages import HumanMessage

from config import PORT
from orchestrator.graph import get_agent
from tools.database import init_db, get_connection
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="智能客服多Agent协作系统", lifespan=lifespan)

# ========== 限流 ==========
_rate_window: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 20       # 每个 IP 每分钟最多请求数
RATE_WINDOW_S = 60    # 窗口秒数


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    cutoff = now - RATE_WINDOW_S
    _rate_window[ip] = [t for t in _rate_window[ip] if t > cutoff]
    if len(_rate_window[ip]) >= RATE_LIMIT:
        return False
    _rate_window[ip].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/chat", "/chat/stream"):
        ip = request.client.host if request.client else "unknown"
        if not check_rate_limit(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试（每分钟最多 20 次）"},
            )
    return await call_next(request)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


# ========== SSE 工具函数 ==========
def format_sse(data: dict) -> str:
    """将 dict 格式化为 SSE 规范的字符串：data: <json>\n\n"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    import os
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(request: Request):
    """非流式对话"""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", str(uuid.uuid4()))

    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    response = (
        result.get("final_response") or
        result.get("worker_result") or
        str(result.get("messages", [{}])[-1].content) if result.get("messages") else "抱歉，我无法处理您的问题。"
    )

    return {
        "session_id": session_id,
        "response": response,
        "intent": result.get("intent", "unknown"),
        "needs_escalation": result.get("needs_escalation", False),
    }


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE 流式对话"""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", str(uuid.uuid4()))

    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        # 发送开始事件
        yield format_sse({"event": "start", "data": json.dumps({"session_id": session_id})})

        # 流式执行 agent
        full_response = ""
        intent = ""
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # 捕获 LLM token 流（过滤掉 supervisor 节点的 JSON 输出）
                if kind == "on_chat_model_stream" and node != "supervisor":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield format_sse({
                            "event": "token",
                            "data": json.dumps({"token": chunk.content}, ensure_ascii=False),
                        })

                # 捕获节点完成事件
                elif kind == "on_chain_end":
                    if event.get("name") == "supervisor":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            intent = output.get("intent", "")
                            yield format_sse({
                                "event": "intent",
                                "data": json.dumps({"intent": intent}, ensure_ascii=False),
                            })
        except Exception as e:
            yield format_sse({"event": "error", "data": json.dumps({"error": str(e)})})

        # 非流式场景的保底：直接使用 ainvoke 结果
        if not full_response:
            try:
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=message)]},
                    config=config,
                )
                full_response = (
                    result.get("final_response") or
                    result.get("worker_result") or
                    (result.get("messages", [{}])[-1].content if result.get("messages") else "抱歉，我无法处理您的问题。")
                )
                yield format_sse({
                    "event": "token",
                    "data": json.dumps({"token": full_response}, ensure_ascii=False),
                })
            except Exception as e:
                yield format_sse({"event": "error", "data": json.dumps({"error": str(e)})})

        yield format_sse({
            "event": "done",
            "data": json.dumps({
                "session_id": session_id,
                "intent": intent,
                "response": full_response,
            }, ensure_ascii=False),
        })

    return StarletteStreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
        logger.exception("Health check: database unavailable")
    try:
        from tools.knowledge_base import get_chroma_client
        client = get_chroma_client()
        client.list_collections()
        chroma_ok = True
    except Exception:
        chroma_ok = False
        logger.exception("Health check: ChromaDB unavailable")
    return {
        "status": "ok" if (db_ok and chroma_ok) else "degraded",
        "database": "ok" if db_ok else "unavailable",
        "chromadb": "ok" if chroma_ok else "unavailable",
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("智能客服多Agent协作系统启动: http://localhost:%d", PORT)
    logger.info("Web UI: http://localhost:%d", PORT)
    logger.info("API Docs: http://localhost:%d/docs", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
