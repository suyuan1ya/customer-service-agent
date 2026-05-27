# 智能客服多 Agent 协作系统 — 面试深度剖析

> 本文档逐层拆解整个项目的架构、代码、设计决策和可改进点，帮助你在面试中自信地讨论每一个细节。

---

## 一、项目全景

### 1.1 一句话概括

基于 **LangGraph Supervisor-Worker 多 Agent 架构** 的智能客服系统，集成 **RAG 知识库检索**、**工具调用**、**SSE 流式对话**，通过通义千问大模型自动将用户请求路由到技术/售后/客服三个专业 Agent，并支持订单查询、物流追踪、工单创建等实际操作。

### 1.2 技术栈速览

| 层级 | 技术 | 作用 |
|---|---|---|
| Web 框架 | FastAPI (async) | HTTP 入口、中间件、路由 |
| Agent 编排 | LangGraph | 状态图构建、节点调度、checkpoint |
| LLM | 通义千问 qwen-plus | 意图分类 + 对话生成 |
| LLM 接入 | DashScope (OpenAI 兼容) | 统一的 Chat Completions API |
| 向量数据库 | ChromaDB (Persistent) | 本地持久化的 FAQ 向量存储 |
| Embedding | DashScope text-embedding-v2 | FAQ 文本向量化 |
| 业务数据库 | SQLite 3 (WAL 模式) | 订单、物流、工单存储 |
| 前端 | 原生 HTML/CSS/JS | SSE 流式消费、逐 token 渲染 |
| 部署 | Docker + docker-compose | 容器化一键部署 |

---

## 二、架构深度拆解

### 2.1 整体架构图 (逻辑视角)

```
HTTP Request (POST /chat/stream)
        │
        ▼
┌──────────────────┐
│  Rate Limiter    │  ← 内存滑动窗口，60s/20次
│  Middleware      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Timing          │  ← perf_counter 计时
│  Middleware      │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────┐
│   LangGraph Agent            │
│                              │
│   ┌──────────┐              │
│   │Supervisor├──┐           │
│   │ (意图分类) │  │           │
│   └──────────┘  │           │
│        │        │           │
│   ┌────┼────────┼───────┐  │
│   │    │        │       │  │
│   ▼    ▼        ▼       ▼  │
│ ┌────┐┌────┐┌──────┐┌─────┐│
│ │客服││售后││技术支持││转人工││
│ │Agent│Agent││ Agent ││ESC)││
│ └──┬─┘└──┬─┘└──┬──┘└─────┘│
│    │     │     │           │
│    ▼     ▼     ▼           │
│  ┌──────────────────┐      │
│  │   Tool Calling   │      │
│  │ ┌──────────────┐ │      │
│  │ │ search_faq   │ │← ChromaDB
│  │ │ query_order  │ │← SQLite
│  │ │ query_logistic│ │← SQLite
│  │ │ create_ticket│ │← SQLite
│  │ └──────────────┘ │      │
│  └──────────────────┘      │
└──────────────────────────────┘
       │
       ▼
  SSE Stream → 前端逐 token 渲染
```

### 2.2 Supervisor-Worker 模式详解

这是整个系统最核心的架构决策。为什么选这个模式而不是其他？

**三种常见 Agent 架构对比：**

| 模式 | 代表 | 适用场景 | 本项目契合度 |
|---|---|---|---|
| **Single Agent** | 一个 LLM 干所有事 | 简单问答 | 不够：需要调用不同工具链 |
| **Supervisor-Worker** | 一个调度 + 多个专业 Worker | 任务类型明确可分类 | **最佳匹配**：客服问题天然可分类 |
| **Swarm / Multi-Agent Debate** | 多个平等 Agent 协作 | 复杂推理、需要多视角 | 过度设计：客服不需要 Agent 间辩论 |

**我们为什么这样设计：**
- 客服场景天然存在部门分工（技术部、售后部、客服部），Supervisor-Worker 是对现实组织结构的直接映射
- 每个 Worker 携带不同的工具集和系统 prompt，实现"专人专事"
- Supervisor 作为流量入口统一做意图识别，避免了在每个 Worker 里重复做判断

### 2.3 LangGraph 状态图运转机制

```python
# orchestrator/graph.py
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("technical_worker", technical_node)
graph.add_node("aftersales_worker", aftersales_node)
graph.add_node("customer_worker", customer_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges("supervisor", route_by_intent, {
    "technical_worker": "technical_worker",
    "aftersales_worker": "aftersales_worker",
    "customer_worker": "customer_worker",
    "finish": END,           # escalate → 直接结束
})

graph.add_edge("technical_worker", END)
graph.add_edge("aftersales_worker", END)
graph.add_edge("customer_worker", END)
```

**运转流程（一次完整请求）：**

```
1. entry_point: supervisor
   └→ supervisor_node 执行
      └→ LLM 输出 {"intent": "customer"}
      └→ 返回 {"intent": "customer", ...}

2. conditional_edges: route_by_intent("customer")
   └→ 返回 "customer_worker"

3. customer_node 执行
   └→ LLM + 工具调用
   └→ 返回最终回复

4. add_edge("customer_worker", END)
   └→ 状态机终止
```

**关键问题：为什么 Worker 直接到 END，不返回 Supervisor？**

这是当前架构的一个有意简化。返回 Supervisor 的架构（闭环）适合需要多轮调度的场景（比如 Worker A 发现自己处理不了，请求重新路由）。当前设计的假设是：Supervisor 分类足够准确 + 每个 Worker 能独立处理自己领域的问题。这是一个合理的 MVP 选择，但也是面试中会被追问的点。

### 2.4 AgentState — 状态的流转机制

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ← 关键
    intent: str
    worker_result: str
    needs_escalation: bool
    final_response: str
```

**`add_messages` 的作用（这是 LangGraph 的核心概念）：**

普通的 TypedDict 字段在节点返回时是 **覆盖（replace）**，而 `add_messages` 是 **追加（append）**。

```python
# 假设 state["messages"] = [msg1, msg2]
# 节点返回 {"messages": [msg3]}

# 普通字段 → 替换
# messages with add_messages → msg1, msg2, msg3
```

这意味着：
- 每个节点的 SystemMessage 和工具调用结果会自动累积到历史中
- 不需要手动管理消息列表
- 后续节点能看到完整的对话上下文

---

## 三、逐文件代码剖析

### 3.1 main.py — FastAPI 应用层

#### 限流器（Rate Limiter）— 滑动窗口算法

```python
_rate_window: dict[str, list[float]] = defaultdict(list)  # IP → [timestamps]
RATE_LIMIT = 20       # 每窗口最多请求
RATE_WINDOW_S = 60    # 窗口 60 秒

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    cutoff = now - RATE_WINDOW_S
    _rate_window[ip] = [t for t in _rate_window[ip] if t > cutoff]  # 清理过期
    if len(_rate_window[ip]) >= RATE_LIMIT:
        return False        # 拒绝
    _rate_window[ip].append(now)
    return True             # 放行
```

**面试要点：**
- 这是 **滑动窗口** 算法，不是固定窗口。不会出现窗口边界被双倍放行的问题。
- **缺陷**：`_rate_window` 字典的 key（IP）永远不会被删除。10000 个不同 IP 访问后，字典有 10000 个空 list。应该定期清理空 IP，或者用 `cachetools.TTLCache`。
- **缺陷**：内存存储，多实例部署时无法共享。生产环境需要 Redis 做集中式限流（如 token bucket）。

#### SSE（Server-Sent Events）流式实现

```python
async def event_generator():
    yield format_sse({"event": "start", ...})

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",               # ← 必须 v2，v1 的 event 结构不同
    ):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        if kind == "on_chat_model_stream" and node != "supervisor":
            # ↑ 过滤掉 supervisor 的输出流，只推送 worker 的 token
            # 因为 supervisor 输出的是 JSON {"intent": "..."}，不是给用户看的
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                yield format_sse({"event": "token", "data": json.dumps({"token": chunk.content})})
```

**为什么过滤 `node != "supervisor"`？** Supervisor 输出的是 `{"intent":"customer","reason":"..."}`，这是机器间的协议格式，不应该暴露给用户。如果不过滤，用户会看到一段 JSON 先弹出来。

**流式失败保底：**
```python
if not full_response:
    # 流式失败 → 降级到非流式 ainvoke
    result = await agent.ainvoke(...)
    full_response = result.get("final_response") or ...
```

这是一个务实的容错设计。某些 LLM 部署可能不支持流式，或者流式中途断开。

**SSE 格式细节：**
```python
def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    #                                  ↑ 保证中文不转义  ↑ SSE 规范要求 \n\n 结尾
```

SSE 协议规范：每条消息以 `data: ` 开头，以 `\n\n` 结尾。`ensure_ascii=False` 保证中文不会被转义为 `\uXXXX`。

#### Health Check

```python
@app.get("/health")
async def health():
    conn = get_connection()
    conn.execute("SELECT 1").fetchone()   # DB 探活
    client = get_chroma_client()
    client.list_collections()             # ChromaDB 探活
    return {"status": "ok" if (db_ok and chroma_ok) else "degraded", ...}
```

返回 `"degraded"` 而不是 500，是因为部分依赖挂了系统仍可降级使用。Docker Compose 的 healthcheck 配置会调用这个端点决定是否重启容器。

**请求时间中间件：**
```python
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()  # ← 比 time.time() 精度更高，不受系统时间调整影响
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.1fms)", ...)
```

### 3.2 orchestrator/graph.py — 状态机构建

#### 线程安全单例模式（Double-Checked Locking）

```python
_agent = None
_lock = threading.Lock()

def get_agent():
    global _agent
    if _agent is None:           # 第一次检查（无锁，快速路径）
        with _lock:              # 获取锁
            if _agent is None:   # 第二次检查（有锁，安全路径）
                _agent = build_graph()
    return _agent
```

**为什么需要两次检查？** 考虑两个线程同时到达第一次 `if _agent is None`：
1. 线程 A 和 B 都看到 `_agent is None`，都试图获取锁
2. 线程 A 先拿到锁，创建 agent，释放锁
3. 线程 B 拿到锁，**此时如果没有第二次检查**，会再次创建 agent
4. 有第二次检查 → B 发现 agent 已存在 → 跳过创建

`build_graph()` 内部的 `MemorySaver()` 是 LangGraph 的 checkpointer，用于保存对话状态。这意味着 Agent 是 **有状态的服务**，需要考虑状态的生命周期。

#### 路由函数

```python
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "customer")     # 缺省 → customer
    routing = {
        "technical": "technical_worker",
        "aftersales": "aftersales_worker",
        "customer": "customer_worker",
        "escalate": "finish",                    # ← 注意：escalate 直接 END
    }
    return routing.get(intent, "customer_worker")  # 未知 intent → customer
```

**escalate 为什么不走 Worker？** 系统中没有 "escalate_worker"。当 Supervisor 判定需要转人工时，直接结束流程，由调用方（`main.py`）根据 `needs_escalation=True` 做后续处理。这是一个"外部处理"的边界设计——把人工介入留给系统外部。

### 3.3 agents/supervisor.py — 意图分类 Supervisor

#### 为什么用独立的 LLM 做分类而不是规则匹配？

规则（关键词匹配）的局限性：
- "我买的音箱不响了" → 规则可能被 "买" 误导为订单问题
- "产品有质量问题，我要退货" → 同时涉及技术 + 售后，需要语义理解

LLM 分类的优势：
- 理解上下文和隐含意图
- 容易扩展：加一个新分类只要在 prompt 里加一行

#### 低温度（temperature=0.1）的设计意图

分类任务需要 **确定性输出**。temperature 越低，模型越倾向选择概率最高的 token，输出越稳定。如果是创造性写作任务（如生成回复），才需要较高的 temperature（0.3-0.7）。

#### `_parse_intent()` — 生产级 LLM 输出容错

这是一个面试时非常好的展示点，体现了"不相信 LLM 输出"的工程素养：

```python
def _parse_intent(raw: str) -> dict:
    content = raw.strip()

    # 第 1 层：去 markdown 代码块 ```json ... ```
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]        # 去掉首行 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]       # 去掉末行 ```
        content = "\n".join(lines)

    # 第 2 层：尝试直接解析
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 第 3 层：提取第一个 { ... } 块
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                # 第 4 层：回退到默认值
                return {"intent": "customer", "reason": "fallback: JSON parse error"}
        else:
            # 找不到 JSON → 回退
            return {"intent": "customer", "reason": "fallback: no JSON found"}

    # 验证 intent 值合法性
    valid_intents = {"technical", "aftersales", "customer", "escalate"}
    if result.get("intent") not in valid_intents:
        intent = "customer"  # fallback
    ...
```

**为什么需要这么复杂的解析？** 因为 LLM 的输出不可控：
1. 可能输出纯 JSON: `{"intent":"customer"}`
2. 可能包在 markdown 里: ` ```json\n{"intent":"customer"}\n``` `
3. 可能前后加废话: `根据分析，分类结果是{"intent":"customer"}，因为...`
4. 可能完全跑偏: `今天天气真好` 或空字符串

每一层都是对一类实际出现过的 LLM 行为的防御。

#### 线程安全的 LLM 单例

```python
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
                    base_url=BASE_URL,              # DashScope 兼容端点
                    temperature=0.1,
                )
    return _supervisor_llm
```

**为什么每个 Agent 有独立的 LLM 实例？** 虽然它们调用的是同一个模型（qwen-plus），但 temperature 和绑定的工具不同：
- Supervisor: temperature=0.1，不绑工具，只做 JSON 分类
- Workers: temperature=0.3，绑定了不同的工具集

另外，如果后续想对不同 Agent 用不同模型（如 Supervisor 用 qwen-turbo 省成本，Customer 用 qwen-max 提升体验），架构已经支持。

### 3.4 agents/customer.py — 客服 Worker（三个 Worker 的代表）

三个 Worker（customer、aftersales、technical）结构几乎相同，以 customer 为代表分析：

```
System Prompt（角色定义 + 工具使用指南）
    │
    ▼
LLM.bind_tools(TOOLS)          ← 将 LangChain Tool 绑定到 LLM
    │
    ▼
LLM 生成响应（可能包含 tool_calls 也可能不包含）
    │
    ├── 无 tool_calls → 直接返回回复
    │
    └── 有 tool_calls → 执行工具 → 结果作为 ToolMessage 追加 → 再次调用 LLM
                        ↑___ 循环最多 5 次 ___↑
```

这就是 **ReAct（Reasoning + Acting）模式**，LangChain 的 Agent 标准范式。

#### Tool 调用循环

```python
llm_with_tools = llm.bind_tools(TOOLS)
response = await _invoke(msgs)

iteration = 0
while response.tool_calls and iteration < 5:   # 最多 5 轮工具调用
    tool_messages = []
    for tc in response.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn:
            result = await tool_fn.ainvoke(tc["args"])
        else:
            result = f"未知工具: {tc['name']}"   # 未知工具不崩溃
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    msgs.extend([response, *tool_messages])
    response = await _invoke(msgs)
    iteration += 1
```

**为什么限制 5 次迭代？** 防止无限循环。如果 LLM 陷入反复调用工具的循环（如一直查询但找不到结果就重试），5 次后强制结束。

**ToolMessage 的作用：** 把工具执行的返回值注回对话上下文。LLM 看到这个结果后，可能继续调用工具，也可能总结信息输出最终回复。

#### 对话历史裁剪（trim_history）

```python
MAX_CONTEXT_MESSAGES = 20

def trim_history(messages, max_messages=20):
    # SystemMessage 始终保留在最前
    # 非 System 消息保留最近 N 条
    # 总消息数控制不超过 max_messages
```

**为什么需要？** LLM 有上下文窗口限制，qwen-plus 约 32K tokens。如果不裁剪，长对话会超出窗口导致报错或高昂成本。保留 SystemMessage 是因为它定义了 Agent 的角色和行为规范。

### 3.5 tools/database.py — SQLite 数据层

#### WAL 模式的并发优势

```python
conn.execute("PRAGMA journal_mode=WAL")
```

SQLite 默认使用 rollback journal（DELETE 模式），写操作会阻塞所有读操作。WAL（Write-Ahead Logging）模式下，**读写可以并发**：
- 写操作写入 WAL 文件，不阻塞读
- 读操作读取数据库文件，不阻塞写
- Checkpoint 定期将 WAL 合并回主文件

在客服系统中，多个 Worker 可能同时查询订单（读）和创建工单（写），WAL 模式提升了并发能力。对于单机部署的 MVP，WAL 模式下的 SQLite 完全可以应对。

#### 种子数据设计

6 个订单覆盖了典型状态（已发货、待付款、已完成、已退款），且物流轨迹有时间递进关系，能展示完整的物流流转。这让系统 demo 起来有真实感。

### 3.6 tools/knowledge_base.py — RAG 知识库检索

#### 为什么选 ChromaDB？

| 向量数据库 | 优势 | 劣势 |
|---|---|---|
| **ChromaDB** | 轻量、pip install 即可、嵌入式部署、Python native | 不适合超大规模（百万级+） |
| Pinecone | 全托管、性能好 | 需要联网、要付费 |
| Milvus | 性能最强、分布式 | 部署运维重 |
| FAISS | 纯本地、最快 | 无元数据过滤、需手动持久化 |

对于这个项目（24 条 FAQ、嵌入式部署），ChromaDB 是性价比最高的选择。

#### DashScope 自定义 Embedding 函数

```python
class DashScopeEmbedding(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # 调用 DashScope Embedding API
        # 带指数退避重试（最多 3 次）
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            json={"model": "text-embedding-v2", "input": {"texts": texts}},
        )
        return [e["embedding"] for e in data["output"]["embeddings"]]
```

**为什么用 DashScope 的 Embedding 而不是开源模型？**
- DashScope text-embedding-v2 对中文的语义理解优于大多数开源模型
- 一致性：LLM 和 Embedding 用同一家厂商，减少协议兼容问题
- 部署简单：不需额外启动 Embedding 服务

**为什么需要重试？** Embedding API 也可能有临时故障（限流、网络抖动）。3 次指数退避（2s、4s、6s）足以应对大部分瞬时故障。

#### FAQ 知识库结构

3 个文件，每文件 8 条 FAQ，以 `##` 分隔：
- `technical.txt`：产品使用、bug、规格参数
- `aftersales.txt`：退换货、保修、投诉
- `customer.txt`：订单、物流、账户

加载时按文件名确定 category，存入 ChromaDB 的 metadata 中，查询时可过滤 `where={"category": "technical"}`。

### 3.7 utils/ — 工具层抽象

#### retry.py — LLM 调用重试

```python
def llm_retry():
    return retry(
        stop=stop_after_attempt(3),                               # 最多 3 次
        wait=wait_exponential(multiplier=1, min=2, max=30),      # 2s → 4s → 8s
        retry=retry_if_exception_type(Exception),                 # 捕获所有异常
        reraise=True,                                             # 3 次后抛出
    )
```

使用 tenacity 库实现指数退避重试。生产环境常见 LLM API 故障：Rate Limit、Gateway Timeout、Connection Reset，重试能解决大部分瞬时问题。

**为什么 `retry_if_exception_type(Exception)` 而不区分异常类型？** LLM API 的异常类型因 SDK 版本而异（openai.RateLimitError、openai.APITimeoutError 等），区分过于脆弱。实际项目中的重试应在第 3 次失败时上报告警。

#### cache.py — FAQ 查询缓存

```python
@lru_cache(maxsize=512)
def cached_search_faq(query: str, category: str) -> str:
    results = search_faq(query, category=category, top_k=3)
    return "\n---\n".join(results) if results else ""
```

**LRU 缓存为什么适合 FAQ 场景？** FAQ 查询有明显的重复性——很多用户会问相同的问题（如"如何退货"）。缓存命中后跳过 Embedding API 调用 + ChromaDB 向量搜索，延迟从 ~500ms 降到 ~1μs。

#### logger.py — 第三方库日志静默

```python
if LOG_LEVEL.upper() != "DEBUG":
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
```

这些库在 INFO 级别会产生大量噪音（HTTP 请求日志、连接池状态等），会被抑制到 WARNING，除非开启 DEBUG 模式排查问题。

### 3.8 static/index.html — 前端 SSE 流式渲染

#### SSE 消费实现

```javascript
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';     // 保留最后一个不完整的行

    for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        // 解析 JSON 并处理 event
    }
}
```

**为什么手动做行缓冲？** SSE 的 `data:` 行可能跨越多个 TCP 包的边界。`buffer` 保存最后一个可能不完整的行，下一个 chunk 到来时与前一行拼接。这是 ReadableStream 处理 SSE 的标准做法。

#### 30 秒超时

```javascript
const ctrl = new AbortController();
const timer = setTimeout(function() { ctrl.abort(); }, 30000);
```

防止 LLM 响应时间过长导致前端永久卡住。超时后 AbortError 被 catch 捕获，显示"请求超时"提示，sendBtn 恢复可用。

---

## 四、完整请求链路追踪

以用户输入 "我的订单 DD20240512001 到哪了？" 为例：

```
时间线                    发生的事
─────────────────────────────────────────────────────────
T+0ms     POST /chat/stream {"message": "我的订单DD20240512001到哪了？"}
T+1ms     rate_limit_middleware: IP 检查通过
T+2ms     timing_middleware: 记录开始时间
T+3ms     生成 session_id (if not provided)
T+4ms     get_agent() → 返回已编译的 LangGraph
T+5ms     创建 config {"configurable": {"thread_id": session_id}}
T+6ms     event_generator() 启动
T+7ms     SSE: {"event":"start","data":{"session_id":"xxx"}}

T+10ms    supervisor_node 开始
T+15ms    LLM 请求: System(supervisor_prompt) + Human("我的订单DD20240512001到哪了？")
T+800ms   LLM 响应: {"intent":"customer","reason":"用户查询订单物流"}
T+810ms   _parse_intent 解析 JSON → {"intent":"customer","reason":"..."}
T+815ms   supervisor_node 返回 {"intent":"customer",...}
T+816ms   SSE: {"event":"intent","data":{"intent":"customer"}}

T+820ms   route_by_intent("customer") → "customer_worker"
T+821ms   customer_node 开始
T+825ms   trim_history(最近 20 条消息)
T+830ms   LLM 请求: System(customer_prompt) + conversation_history
T+1500ms  LLM 响应: content="好的，我来帮您查询..." tool_calls=[{"name":"query_order","args":{"order_id":"DD20240512001"}},{"name":"query_logistics","args":{"order_id":"DD20240512001"}}]
T+1501ms  SSE: token="好的" → token="，" → token="我来" → ... (实时流式)

T+1505ms  执行 query_order("DD20240512001")
T+1506ms  SQLite: SELECT * FROM orders WHERE order_id='DD20240512001'
T+1508ms  返回: "订单号: DD20240512001\n状态: 已发货\n..."

T+1510ms  执行 query_logistics("DD20240512001")
T+1511ms  SQLite: SELECT * FROM logistics WHERE order_id='DD20240512001'
T+1513ms  返回: "2024-05-12 快件已揽收 — 深圳分拣中心\n..."

T+1515ms  将 ToolMessage 追加到消息列表
T+1520ms  LLM 第二次调用（带上工具结果）
T+2100ms  LLM 响应: "您的订单DD20240512001（智能音箱X1）目前正在北京配送站，快递员已接单派送中..."
T+2105ms  SSE: token by token 持续推送

T+2110ms  customer_node 返回 {"worker_result":"...","final_response":"..."}
T+2112ms  SSE: {"event":"done","data":{"session_id":"xxx","intent":"customer","response":"..."}}

T+2115ms  timing_middleware: POST /chat/stream → 200 (2112.5ms)
```

**关键观察：**
1. 整个请求约 2 秒，其中 LLM 调用占了约 1.8 秒（85%+ 的延迟来自 LLM）
2. 工具调用（SQLite 查询）只需 ~5ms，几乎可以忽略
3. SSE 流式推送从 T+1500ms 开始，用户约 1.5 秒看到第一个字，而不是 2.1 秒看到全部

---

## 五、设计模式总结

| 模式 | 应用位置 | 作用 |
|---|---|---|
| **Supervisor-Worker** | orchestrator/graph.py | 分布式任务调度，职责分离 |
| **Singleton** | 各 agent 的 LLM、ChromaDB client | 避免重复创建昂贵资源 |
| **Double-Checked Locking** | 所有 get_xxx() 函数 | 线程安全的懒加载单例 |
| **Chain of Responsibility** | _parse_intent 的 4 层 fallback | LLM 输出的鲁棒解析 |
| **ReAct** | Worker 的工具调用循环 | LLM 推理与执行交替 |
| **Strategy** | route_by_intent | 根据意图选择不同的处理策略 |
| **Decorator** | llm_retry()、@lru_cache、@tool | 无侵入地增强函数能力 |
| **Template Method** | 三个 Worker 的共同结构 | 定义处理骨架，子类/实例填充具体行为 |
| **Observer** | SSE streaming | 服务端推送事件，客户端响应式更新 |

---

## 六、面试高频问答

### Q1: 为什么选 LangGraph 而不是直接用 LangChain 的 AgentExecutor？

**参考答案：** LangGraph 提供了显式的状态机和流控能力。在这个项目里，我需要：
1. Supervisor 先分类，再根据分类结果决定走哪个 Worker——这是条件路由，AgentExecutor 不支持
2. 通过 `astream_events` 拿到每个节点的 LLM 流式输出，并且可以按 node 名过滤——AgentExecutor 的流式粒度更粗
3. 如果以后需要 Worker 之间协作（如 Technical 发现处理不了转给 Aftersales），只需要加一条 edge，LangGraph 改起来成本很低

LangGraph 相比原始 AgentExecutor，多了一层 **编排层**，把"LLM 怎么思考"和"系统怎么流转"分开了。

### Q2: 意图分类为什么用 LLM 而不是训练一个分类模型？

**参考答案：**
- **样本问题**：没有标注数据，LLM 可以 zero-shot 完成分类
- **灵活性**：prompt 里加一行规则就能调整分类逻辑，训练模型需要重新采集样本、训练、部署
- **成本合理**：一次分类调用（qwen-plus + 短 prompt）约 0.002 元，还不值得训练专用模型

如果量大到一天百万次请求，可以考虑用 LLM 标注一批数据 → 训练轻量 BERT 分类模型 → 模型做初筛 + LLM 兜底，这样成本能降一个数量级。

### Q3: ChromaDB 的检索效果不好怎么办？

**参考答案：** 向量检索的瓶颈通常不在工具，而在：
1. **分块策略**：当前是整条 FAQ 作为一个文档。如果 FAQ 很长，应该拆成更细的 chunk，提高检索精度
2. **Embedding 模型**：text-embedding-v2 对短文本效果好，如果 FAQ 包含大量专业术语，可能需要微调 embedding 模型或换用针对特定领域的模型
3. **混合检索**：纯语义检索会漏掉精确匹配。加上 BM25 关键词检索做混合（Hybrid Search），对包含订单号、错误码等精确信息的查询效果更好
4. **Reranker**：检索出 top_k 后，用一个 Cross-Encoder 模型做一次重排序，相关性能提升 10-20%

具体排查链路：先看几个 bad case → 确认是没检索到（recall 低）还是检索到了但 LLM 没用对 → 针对性优化。

### Q4: 如果要支持 1000 并发，这个系统哪些地方会先崩？

**参考答案：** 按脆弱度排序：

1. **LLM API（最可能）**：DashScope API 有 TPM/RPM 限制，1000 并发远超免费配额。需要申请提额 + 排队机制（如 `asyncio.Queue`）+ 熔断降级
2. **SQLite 写锁**：WAL 模式支持并发读，但写操作（create_ticket）是串行的。高并发下写锁排队会拖垮响应时间。需要迁移到 PostgreSQL
3. **内存限流器**：多实例部署后，每个实例独立计数，用户轮询不同实例可以绕过限流。需要 Redis 集中式限流
4. **MemorySaver**：所有会话数据在内存中，1000 并发 × 多轮对话 → 内存爆炸。需要持久化到数据库
5. **ChromaDB**：Persistent 模式单机读写，高并发下 IO 瓶颈。需要升级到 ChromaDB 的 Client/Server 模式或用 Milvus

### Q5: 三个 Worker 的代码几乎一样，你怎么重构？

**参考答案：** 三个 Worker 的共同模式是：prompt → LLM → bind_tools → tool_loop → return。可以抽取一个工厂函数：

```python
def create_worker_node(name: str, prompt: str, tools: list, temperature: float = 0.3):
    tool_map = {t.name: t for t in tools}
    llm = ChatOpenAI(model=QWEN_MODEL, temperature=temperature, ...)
    llm_with_tools = llm.bind_tools(tools)

    async def worker_node(state):
        messages = trim_history(state["messages"])
        msgs = [SystemMessage(content=prompt), *messages]
        response = await llm_with_tools.ainvoke(msgs)

        iteration = 0
        while response.tool_calls and iteration < 5:
            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                result = await fn.ainvoke(tc["args"]) if fn else "未知工具"
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            response = await llm_with_tools.ainvoke(msgs)
            iteration += 1

        return {"worker_result": response.content, "final_response": response.content}

    return worker_node

# 使用
customer_node = create_worker_node("customer", CUSTOMER_PROMPT, [search_faq_tool, query_order, query_logistics])
```

这样 3 个文件可以合并为一个，新增 Worker 只需一行配置。

### Q6: MemorySaver 在内存中存了什么？重启会丢什么？

**参考答案：** MemorySaver 是 LangGraph 的内存 checkpoint 存储。每次图执行中遇到节点边界，LangGraph 会自动保存当前的 AgentState 快照。重启后：
- 所有对话历史（messages）丢失
- 所有 checkpoint 丢失
- session_id 对应的上下文丢失

这意味着用户的 session_id 还在前端 sessionStorage 里，但后端已经不认识这个 session 了——会把它当新会话处理。生产环境应该用 `SqliteSaver`（单机）或 `PostgresSaver`（分布式）。

### Q7: SSE 和 WebSocket 有什么区别？为什么用 SSE？

**参考答案：**

| 维度 | SSE | WebSocket |
|---|---|---|
| 方向 | 单向（服务→客户端） | 双向（全双工） |
| 协议 | HTTP（标准） | 升级后的 TCP（ws://） |
| 断线重连 | 浏览器内置自动重连 | 需要手动实现 |
| 穿透代理/防火墙 | 好（就是 HTTP） | 可能被阻断 |
| 复杂度 | 极低 | 较高 |

这个场景是"用户发一条消息 + 服务端流式返回"——天然是请求-响应模式 + 流式推送响应体，SSE 是完美匹配。WebSocket 更适合"双方随时都要发消息"的场景（如在线协作、游戏）。

### Q8: 项目里有哪些你没做但知道应该做的事？

**参考答案（展示工程意识和边界感）：**

1. **会话持久化**：MemorySaver → SqliteSaver/PostgresSaver
2. **API 鉴权**：加 JWT 或 API Key 认证
3. **消息队列**：LLM 调用耗时长，用户请求应该进入队列异步处理，避免 HTTP 连接一直挂着
4. **监控告警**：LLM 调用延迟 P99、错误率、token 消耗量 → 接入 Prometheus + Grafana
5. **A/B 测试框架**：不同 prompt 版本的效果对比
6. **反馈闭环**：用户对回复的满意度评价 → 用于优化 prompt 和 FAQ 覆盖率
7. **多语言支持**：当前只有中文
8. **敏感信息过滤**：用户可能在对话中暴露手机号、身份证号，需要脱敏

---

## 七、简历话术建议

### 项目描述（3-4 行版，适合简历篇幅）

> **智能客服多Agent协作系统** | Python, LangGraph, FastAPI, ChromaDB
>
> - 基于 LangGraph 实现 Supervisor-Worker 多 Agent 架构，将用户请求自动路由至技术/售后/客服三个专业 Agent，各 Agent 携带专属工具链（订单查询、物流追踪、工单创建、知识库检索）
> - 集成 ChromaDB + DashScope Embedding 构建 RAG 检索链路，实现 24 条 FAQ 的语义化匹配；设计 LRU 缓存减少重复 Embedding 调用
> - 实现 SSE 流式对话与前端逐 token 渲染，用户首字延迟约 1.5 秒；设计 LLM 指数退避重试、多层 JSON 容错解析、对话历史裁剪等可靠性机制
> - Docker Compose 一键部署，包含 healthcheck、IP 滑动窗口限流、结构化日志记录

### 如果面试官说"展开讲讲"

立刻抓住三个最有区分度的点：

**1. "多 Agent 是怎么协作的？"**
→ 画出 Supervisor → Worker 的状态图，解释 route_by_intent 的决策逻辑，说明 escalate 不经过 worker 直接结束的设计意图

**2. "LLM 输出不稳定你怎么处理的？"**
→ 讲 `_parse_intent` 的 4 层 fallback：去 markdown → 直接解析 → 提取 JSON 片段 → 默认回退。这是项目的亮点

**3. "流式对话怎么实现的？"**
→ 讲 `astream_events` 的事件过滤（为什么过滤 supervisor 节点）、SSE 格式规范、前端的 ReadableStream + buffer 行缓冲、流式失败的 ainvoke 保底

---

## 八、进阶方向（加分项）

如果面试顺利，可以主动提到的扩展方向：

1. **Human-in-the-loop**：escalate 时不是直接结束，而是发送通知给人工坐席，人工接管后可以注入消息继续图执行
2. **多 Agent 辩论/投票**：对于复杂问题，同时派发给多个 Worker，收集结果后由 Supervisor 综合判断
3. **Tool 即服务**：把每个 tool 独立部署为微服务，支持独立扩缩容和灰度发布
4. **Prompt 版本管理**：在生产环境中，prompt 的变更应该走 CI/CD（类似代码），每次变更记录版本号、评测指标、回滚机制
5. **成本优化**：简单问题用 qwen-turbo，复杂问题用 qwen-max，通过 Supervisor 做模型路由（不仅是意图路由，还包括难度路由）

---

## 九、关键文件速查

| 文件 | 行数 | 核心内容 |
|---|---|---|
| `main.py` | 232 | FastAPI 入口、限流、SSE、health check |
| `orchestrator/graph.py` | 69 | LangGraph 状态图构建、路由、单例 |
| `orchestrator/state.py` | 12 | AgentState 定义 |
| `agents/supervisor.py` | 109 | 意图分类、LLM 单例、JSON 容错解析 |
| `agents/customer.py` | 84 | 客服 Worker（ReAct 模式代表） |
| `agents/aftersales.py` | 84 | 售后 Worker |
| `agents/technical.py` | 82 | 技术支持 Worker |
| `tools/knowledge_base.py` | 102 | ChromaDB 客户端、Embedding 函数、FAQ 检索 |
| `tools/database.py` | 79 | SQLite 初始化、种子数据、WAL 模式 |
| `tools/order_tools.py` | 45 | 订单查询、物流查询 LangChain Tool |
| `tools/ticket_tools.py` | 43 | 工单创建、工单查询 LangChain Tool |
| `utils/retry.py` | 28 | LLM 调用指数退避重试 |
| `utils/cache.py` | 23 | FAQ 查询 LRU 缓存 |
| `utils/conversation.py` | 31 | 对话历史裁剪 |
| `utils/logger.py` | 23 | 结构化日志、第三方库静默 |
| `knowledge_base/loader.py` | 55 | 知识库加载脚本 |
| `static/index.html` | 224 | SSE 流式渲染前端 |
| `tests/test_supervisor.py` | 57 | intent 解析的 8 个边界测试 |
| `tests/test_routing.py` | 23 | 路由逻辑的 6 个测试 |
| `tests/test_api.py` | 62 | API 集成测试 + 限流测试 |
| `Dockerfile` | 28 | 非 root 用户、healthcheck |
| `docker-compose.yml` | 23 | 环境变量注入、数据卷挂载 |

---

> **面试前建议做的事情：**
> 1. 把项目跑起来，实际发几句对话感受流程
> 2. 在 `_parse_intent` 函数里打断点或加日志，观察 LLM 的真实输出格式
> 3. 对着上面的 Q&A 用自己的话练习一遍
> 4. 如果时间充裕，把 SqliteSaver 替换 MemorySaver 的改动实际做一遍（只需要改 graph.py 一行代码 + 安装 langgraph-checkpoint-sqlite）
