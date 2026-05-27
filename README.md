# 智能客服 Multi-Agent 协作系统

基于 [LangGraph](https://github.com/langchain-ai/langgraph) Supervisor-Worker 架构的智能客服系统，使用通义千问 (DashScope) 作为大模型，通过多 Agent 协作处理不同类型客户咨询。

## 架构

```
用户 → Supervisor（意图识别）→
├── Customer Worker  → 售前咨询
├── Technical Worker → 技术问题
└── Aftersales Worker → 售后/退换货
```

- **Supervisor**：分类用户意图，路由到对应 Worker
- **Customer Worker**：产品信息、价格、活动咨询
- **Technical Worker**：故障排查、技术参数解答
- **Aftersales Worker**：退换货流程、物流跟踪、工单处理

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/suyuan1ya/customer-service-agent.git
cd customer-service-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python main.py
```

访问 http://localhost:8000 打开聊天界面，http://localhost:8000/docs 查看 API 文档。

### Docker 部署

```bash
docker compose up -d
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 聊天界面 |
| `/chat` | POST | 非流式对话 |
| `/chat/stream` | POST | SSE 流式对话 |
| `/health` | GET | 健康检查 |

### 请求示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我要退货怎么操作？", "session_id": "user-001"}'
```

## 技术栈

- **框架**: FastAPI + LangGraph
- **模型**: 通义千问 (DashScope OpenAI-Compatible API)
- **向量数据库**: ChromaDB
- **数据库**: SQLite
- **部署**: Docker / Docker Compose

## 项目结构

```
├── main.py              # FastAPI 入口
├── config.py            # 配置
├── orchestrator/        # LangGraph 状态图
│   ├── graph.py         # 图定义与编译
│   └── state.py         # 状态定义
├── agents/              # Agent 节点
│   ├── supervisor.py    # 意图识别
│   ├── customer.py      # 售前
│   ├── technical.py     # 技术
│   └── aftersales.py    # 售后
├── tools/               # 工具函数
│   ├── database.py      # SQLite 操作
│   ├── knowledge_base.py # ChromaDB 检索
│   ├── order_tools.py   # 订单操作
│   └── ticket_tools.py  # 工单操作
├── utils/               # 工具库
├── data/faq/            # FAQ 知识库文件
├── static/              # 前端页面
└── tests/               # 测试
```

## License

MIT
