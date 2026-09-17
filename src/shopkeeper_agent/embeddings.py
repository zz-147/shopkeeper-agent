"""DashScope text-embedding-v4 提供者。"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import dashscope
from requests import RequestException


DASHSCOPE_ENVIRONMENT_KEYS = frozenset(
    {
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_EMBEDDING_MODEL",
        "DASHSCOPE_EMBEDDING_DIMENSIONS",
    }
)


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为一批文档生成向量。"""

    def embed_query(self, text: str) -> list[float]:
        """为查询生成向量。"""


class DashScopeEmbeddingProvider:
    """使用 DashScope 官方 SDK 调用 text-embedding-v4。"""

    max_batch_size = 10

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self.api_key = api_key
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
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            dimensions=dimensions,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        text_list = list(texts)
        vectors: list[list[float]] = []
        # 当前服务端已验证单次最多接受 10 条文本；在此处分批，调用方无需关心限制。
        for start in range(0, len(text_list), self.max_batch_size):
            vectors.extend(self._embed(text_list[start : start + self.max_batch_size], text_type="document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding 查询文本不能为空。")
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        try:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=texts,
                api_key=self.api_key,
                text_type=text_type,
                dimension=self.dimensions,
            )
        except RequestException as error:
            raise RuntimeError("无法连接 Embedding 服务。请检查网络和代理配置。") from error
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
