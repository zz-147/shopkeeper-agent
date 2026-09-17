"""DashScope text-embedding-v4 提供者。"""

from __future__ import annotations

import os
from typing import Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

import dashscope


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为一批文档生成向量。"""

    def embed_query(self, text: str) -> list[float]:
        """为查询生成向量。"""


class DashScopeEmbeddingProvider:
    """基于 DashScope 官方 SDK 的 text-embedding-v4 客户端。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_address: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_address = base_address

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
        configured_base_url = os.getenv("DASHSCOPE_EMBEDDING_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
        return cls(
            api_key=api_key,
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            dimensions=dimensions,
            base_address=_native_api_base_url(configured_base_url),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(list(texts), text_type="document")

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding 查询文本不能为空。")
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            text_type=text_type,
            dimension=self.dimensions,
            base_address=self.base_address,
        )
        if response.status_code != 200:
            error_code = response.code or "unknown"
            raise RuntimeError(
                f"Embedding 服务请求失败（HTTP {response.status_code}，错误代码：{error_code}）。"
            )
        try:
            records = sorted(response.output["embeddings"], key=lambda record: record["text_index"])
            vectors = [list(record["embedding"]) for record in records]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Embedding 服务返回格式异常。") from error
        if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("Embedding 返回数量或向量维度与配置不一致。")
        return vectors


def _native_api_base_url(base_url: str | None) -> str | None:
    """将兼容模式地址转换成供 DashScope SDK 使用的原生 API 地址。"""

    if not base_url:
        return None
    parsed = urlsplit(base_url)
    normalized_path = parsed.path.rstrip("/")
    if normalized_path == "/compatible-mode/v1":
        normalized_path = "/api/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))
