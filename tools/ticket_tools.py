"""工单工具 — 基于 SQLite 持久化存储"""
import uuid
import datetime
from langchain_core.tools import tool
from tools.database import get_connection


@tool
def create_ticket(description: str, category: str, priority: str = "normal") -> str:
    """创建售后工单。当用户需要正式记录问题、投诉、退换货、保修维修时调用。
    参数: description（问题描述）、category（aftersales/technical/customer）、
    priority（low/normal/high/urgent，默认normal）。
    用户在涉及投诉、涉及退款退货、明确要求开具工单、问题需要跟踪处理时必须调用。"""
    ticket_id = f"TK-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    created = datetime.datetime.now().isoformat()
    conn = get_connection()
    conn.execute(
        "INSERT INTO tickets (ticket_id, description, category, priority, status, created) VALUES (?, ?, ?, ?, '已创建', ?)",
        (ticket_id, description[:200], category, priority, created),
    )
    conn.commit()
    conn.close()
    return f"工单已创建 — 工单号: {ticket_id}, 状态: 已创建, 分类: {category}, 优先级: {priority}"


@tool
def query_ticket(ticket_id: str) -> str:
    """查询已有工单的处理状态。输入工单号（如 TK-20240512-ABC123），
    返回工单的当前状态、分类、描述和创建时间。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    if not row:
        conn.close()
        return f"未找到工单 {ticket_id}，请确认工单号是否正确。"
    result = (
        f"工单号: {row['ticket_id']}\n"
        f"状态: {row['status']}\n"
        f"分类: {row['category']}\n"
        f"描述: {row['description']}\n"
        f"创建时间: {row['created']}"
    )
    conn.close()
    return result
