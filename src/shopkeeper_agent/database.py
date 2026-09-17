"""SQLite 样例数仓。生产中可替换为只读 MySQL / 数据仓库连接。"""

from __future__ import annotations

import sqlite3
from pathlib import Path


DEMO_ORDERS = (
    (1, "2025-01-03", "华北", "电器", "金卡", 2000.0),
    (2, "2025-01-05", "华东", "家居", "银卡", 1500.0),
    (3, "2025-01-12", "华北", "图书", "普通", 500.0),
    (4, "2025-01-18", "华南", "电器", "金卡", 3000.0),
    (5, "2025-01-24", "华北", "家居", "银卡", 800.0),
    (6, "2025-01-30", "华东", "图书", "普通", 700.0),
)


def create_demo_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                order_date TEXT NOT NULL,
                region TEXT NOT NULL,
                category TEXT NOT NULL,
                customer_level TEXT NOT NULL,
                payment_amount REAL NOT NULL
            )
            """
        )
        existing_count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if existing_count == 0:
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", DEMO_ORDERS)
        connection.commit()
    finally:
        # sqlite3 的 connection 上下文只负责 commit/rollback，不负责 close；Windows 会因此锁住文件。
        connection.close()


def execute_readonly_query(database_path: Path, sql: str) -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(sql)
        columns = tuple(column[0] for column in cursor.description or ())
        rows = [tuple(row) for row in cursor.fetchall()]
    finally:
        connection.close()
    return columns, rows
