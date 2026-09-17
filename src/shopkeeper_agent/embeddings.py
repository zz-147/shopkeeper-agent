"""Embedding 提供者。DashScope 使用 OpenAI 兼容的 embeddings 接口。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为一批文档生成向量。"""

    def embed_query(self, text: str) -> list[float]:
        """为查询生成向量。"""


class DashScopeEmbeddingProvider:
    """DashScope text-embedding-v4 的最小、无框架依赖客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str, dimensions: int) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions

    @classmethod
    def from_environment(cls) -> "DashScopeEmbeddingProvider":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 DASHSCOPE_API_KEY。请在本项目 .env 中配置后再构建向量索引。")
        dimensions_text = os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024")
        try:
            dimensions = int(dimensions_text)
        except ValueError as error:
            raise RuntimeError("DASHSCOPE_EMBEDDING_DIMENSIONS 必须是正整数。") from error
        if dimensions <= 0:
            raise RuntimeError("DASHSCOPE_EMBEDDING_DIMENSIONS 必须是正整数。")
        return cls(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            dimensions=dimensions,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding 查询文本不能为空。")
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {"model": self.model, "input": texts, "dimensions": self.dimensions, "encoding_format": "float"},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Embedding 服务返回 HTTP {error.code}。") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Embedding 服务请求失败：{error.reason}") from error

        try:
            records = sorted(body["data"], key=lambda record: record["index"])
            vectors = [list(record["embedding"]) for record in records]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Embedding 服务返回格式异常。") from error
        if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("Embedding 返回数量或向量维度与配置不一致。")
        return vectors
