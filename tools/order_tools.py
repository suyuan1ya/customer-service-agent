"""订单&物流工具 — 基于 SQLite 数据库"""
from langchain_core.tools import tool
from tools.database import get_connection


@tool
def query_order(order_id: str) -> str:
    """查询订单详情。输入订单号（如 DD20240512001），返回订单状态、商品名、
    价格、创建时间、物流预计送达时间等信息。
    当用户询问"我的订单""查一下订单""订单状态"时调用此工具。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if not row:
        # 列出可用订单号
        ids = [r["order_id"] for r in conn.execute("SELECT order_id FROM orders LIMIT 10")]
        conn.close()
        return f"未找到订单 {order_id}，请确认订单号是否正确。可查询的订单: {', '.join(ids)}"
    result = (
        f"订单号: {row['order_id']}\n"
        f"状态: {row['status']}\n"
        f"商品: {row['product']}\n"
        f"金额: ¥{row['price']}\n"
        f"创建时间: {row['created']}\n"
        f"物流: {row['delivery']}"
    )
    conn.close()
    return result


@tool
def query_logistics(order_id: str) -> str:
    """查询物流轨迹详情。输入订单号，返回包裹在每个节点的状态、时间和地点。
    当用户询问"物流到哪了""快递进度""什么时候到"时调用此工具。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT time, status, location FROM logistics WHERE order_id = ? ORDER BY id",
        (order_id,),
    ).fetchall()
    if not rows:
        conn.close()
        return f"未找到订单 {order_id} 的物流信息，可能尚未发货或订单号有误。"
    lines = [f"{r['time']}  {r['status']} — {r['location']}" for r in rows]
    conn.close()
    return "\n".join(lines)
