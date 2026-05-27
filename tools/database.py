"""SQLite 数据库——替代 mock 数据"""
import sqlite3
import os
import threading

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "orders.db")
_lock = threading.Lock()

SEED_ORDERS = [
    ("DD20240512001", "已发货", "智能音箱 X1", 299.00, "2024-05-12 10:30", "预计 2024-05-15 送达"),
    ("DD20240510002", "待付款", "无线耳机 Pro", 599.00, "2024-05-10 14:20", "付款后 48 小时内发货"),
    ("DD20240508003", "已完成", "机械键盘 K8", 449.00, "2024-05-08 09:15", "已于 2024-05-11 签收"),
    ("DD20240505004", "已退款", "智能手表 S3", 899.00, "2024-05-05 16:00", "退款已到账"),
    ("DD20240520001", "已发货", "蓝牙耳机 Air", 199.00, "2024-05-20 11:00", "预计 2024-05-23 送达"),
    ("DD20240518002", "已完成", "移动电源 P10", 129.00, "2024-05-18 15:30", "已于 2024-05-20 签收"),
]

SEED_LOGISTICS = [
    ("DD20240512001", "2024-05-12 20:00", "快件已揽收", "深圳分拣中心"),
    ("DD20240512001", "2024-05-13 08:00", "运输中", "广州中转站"),
    ("DD20240512001", "2024-05-14 06:00", "到达目的地", "北京配送站"),
    ("DD20240512001", "2024-05-15 09:00", "派送中", "快递员已接单"),
    ("DD20240508003", "2024-05-08 17:00", "快件已揽收", "上海分拣中心"),
    ("DD20240508003", "2024-05-09 12:00", "运输中", "杭州中转站"),
    ("DD20240508003", "2024-05-10 10:00", "派送中", "快递员已接单"),
    ("DD20240508003", "2024-05-11 14:30", "已签收", "本人签收"),
    ("DD20240520001", "2024-05-20 20:00", "快件已揽收", "广州分拣中心"),
    ("DD20240520001", "2024-05-21 06:00", "运输中", "长沙中转站"),
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    with _lock:
        conn = get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id   TEXT PRIMARY KEY,
                status     TEXT NOT NULL,
                product    TEXT NOT NULL,
                price      REAL NOT NULL,
                created    TEXT NOT NULL,
                delivery   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS logistics (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL REFERENCES orders(order_id),
                time     TEXT NOT NULL,
                status   TEXT NOT NULL,
                location TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id   TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                category    TEXT NOT NULL,
                priority    TEXT NOT NULL DEFAULT 'normal',
                status      TEXT NOT NULL DEFAULT '已创建',
                created     TEXT NOT NULL
            );
        """)
        if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", SEED_ORDERS
            )
            conn.executemany(
                "INSERT INTO logistics (order_id, time, status, location) VALUES (?, ?, ?, ?)",
                SEED_LOGISTICS,
            )
        conn.commit()
        conn.close()
