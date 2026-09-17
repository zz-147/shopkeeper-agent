"""Qdrant 本地持久化索引的薄封装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class VectorDocument:
    id: int
    text: str
    payload: dict[str, str]


@dataclass(frozen=True)
class VectorMatch:
    score: float
    payload: dict[str, str]


class QdrantMetadataIndex:
    collection_name = "shopkeeper_metadata"

    def __init__(self, path: Path | str, vector_size: int) -> None:
        self.vector_size = vector_size
        self.client = QdrantClient(":memory:") if str(path) == ":memory:" else QdrantClient(path=str(path))

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def rebuild(self, documents: Sequence[VectorDocument], vectors: Sequence[Sequence[float]]) -> None:
        if not documents or len(documents) != len(vectors):
            raise ValueError("向量文档不能为空，且文档数量必须与向量数量一致。")
        if any(len(vector) != self.vector_size for vector in vectors):
            raise ValueError("存在与索引维度不一致的向量。")
        if self.exists():
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                models.PointStruct(id=document.id, vector=list(vector), payload=document.payload)
                for document, vector in zip(documents, vectors, strict=True)
            ],
        )

    def search(self, query_vector: Sequence[float], limit: int, kind: str | None = None) -> tuple[VectorMatch, ...]:
        if not self.exists():
            raise RuntimeError("未找到本地向量索引。请先运行 scripts/build_vector_index.py。")
        if len(query_vector) != self.vector_size:
            raise ValueError("查询向量维度与索引维度不一致。")
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=limit,
            query_filter=(
                models.Filter(
                    must=[models.FieldCondition(key="kind", match=models.MatchValue(value=kind))]
                )
                if kind
                else None
            ),
        ).points
        return tuple(VectorMatch(score=float(point.score), payload=dict(point.payload or {})) for point in points)

    def close(self) -> None:
        self.client.close()
