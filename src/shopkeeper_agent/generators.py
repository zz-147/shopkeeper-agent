"""SQL 生成器。规则版用于离线、可重复测试；DeepSeek 版保留真实模型接入。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from .catalog import MetadataCatalog


class SQLGenerator(Protocol):
    def generate(self, question: str, metadata_context: str) -> str:
        """只返回 SQL 文本，不执行 SQL。"""


class RuleBasedSQLGenerator:
    """用确定性规则模拟“根据已召回元数据生成 SQL”的阶段。"""

    def __init__(self, catalog: MetadataCatalog | None = None) -> None:
        self.catalog = catalog or MetadataCatalog()

    def generate(self, question: str, metadata_context: str) -> str:
        del metadata_context
        metric = self.catalog.find_metric(question)
        group_dimension = self.catalog.find_group_dimension(question)
        filters = self.catalog.find_filters(question)

        select_parts = []
        if group_dimension:
            select_parts.append(group_dimension.column)
        select_parts.append(f"{metric.expression} AS result")
        sql = f"SELECT {', '.join(select_parts)} FROM {self.catalog.table_name}"

        if filters:
            conditions = [f"{dimension.column} = '{value}'" for dimension, value in filters]
            sql += f" WHERE {' AND '.join(conditions)}"
        if group_dimension:
            sql += f" GROUP BY {group_dimension.column} ORDER BY result DESC"
        return f"{sql} LIMIT 100"


class DeepSeekSQLGenerator:
    """使用 DeepSeek 的 OpenAI 兼容接口生成 SQL；未配置密钥时拒绝执行。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @classmethod
    def from_environment(cls) -> "DeepSeekSQLGenerator":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 DEEPSEEK_API_KEY。请在本机环境变量或 .env 文件中配置，且不要把密钥发给我。")
        return cls(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    def generate(self, question: str, metadata_context: str) -> str:
        prompt = (
            "你是电商数据分析 SQL 生成器。仅可依据给定元数据回答。"
            "仅输出一条 SQLite SELECT SQL，不要 Markdown，不要解释，不要使用未声明的表或字段。\n\n"
            f"元数据：\n{metadata_context}\n\n用户问题：{question}"
        )
        payload = json.dumps(
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"模型请求失败：{error.reason}") from error
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("模型返回格式异常，未取得 SQL。") from error
        return re.sub(r"^```(?:sql)?|```$", "", content.strip(), flags=re.IGNORECASE).strip()

