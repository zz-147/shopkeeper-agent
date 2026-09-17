"""SQL 执行前的最小安全策略：只读、单语句、表和字段白名单。"""

from __future__ import annotations

import re


class SQLSafetyValidator:
    forbidden_keywords = re.compile(
        r"\b(attach|alter|create|delete|detach|drop|insert|pragma|replace|truncate|update|vacuum)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, allowed_table: str, allowed_columns: frozenset[str]) -> None:
        self.allowed_table = allowed_table.lower()
        self.allowed_columns = frozenset(column.lower() for column in allowed_columns)

    def validate(self, raw_sql: str) -> str:
        sql = raw_sql.strip().rstrip(";").strip()
        if not sql:
            raise ValueError("模型没有生成 SQL。")
        if ";" in sql:
            raise ValueError("只允许一条 SQL 语句。")
        if not re.match(r"^select\b", sql, flags=re.IGNORECASE):
            raise ValueError("只允许 SELECT 查询。")
        if self.forbidden_keywords.search(sql):
            raise ValueError("SQL 含有禁止的写入或管理操作。")

        table_matches = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", sql, flags=re.IGNORECASE)
        if table_matches != [self.allowed_table]:
            raise ValueError(f"只允许查询表：{self.allowed_table}。")

        identifiers = {
            identifier.lower()
            for identifier in re.findall(r"\b[a-zA-Z_][\w]*\b", sql)
        }
        permitted_sql_words = {
            "as", "asc", "avg", "by", "count", "desc", "from", "group", "limit", "order", "result", "select", "sum", "where", "and",
        }
        unknown = identifiers - self.allowed_columns - {self.allowed_table} - permitted_sql_words
        # 中文字面量不会被这个 ASCII 标识符规则识别；它们只会作为带引号的过滤值传入 SQLite。
        if unknown:
            raise ValueError(f"SQL 使用了未声明标识符：{', '.join(sorted(unknown))}。")
        return sql

