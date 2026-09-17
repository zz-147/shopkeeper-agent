"""将元数据写入 Qdrant，并用 Embedding 进行语义召回。"""

from __future__ import annotations

from pathlib import Path

from .catalog import Dimension, MetadataCatalog, Metric
from .embeddings import DashScopeEmbeddingProvider, EmbeddingProvider
from .retrieval import RetrievedMetadata
from .vector_store import QdrantMetadataIndex, VectorDocument


def build_metadata_documents(catalog: MetadataCatalog) -> tuple[VectorDocument, ...]:
    documents: list[VectorDocument] = []
    next_id = 1
    for metric in catalog.metrics:
        documents.append(
            VectorDocument(
                id=next_id,
                text=(
                    f"指标：{metric.name}。别名：{', '.join(metric.aliases)}。"
                    f"计算公式：{metric.expression}。含义：{metric.description}。"
                ),
                payload={"kind": "metric", "metric_name": metric.name},
            )
        )
        next_id += 1
    for dimension in catalog.dimensions:
        documents.append(
            VectorDocument(
                id=next_id,
                text=f"字段：{dimension.name}，数据库列：{dimension.column}。别名：{', '.join(dimension.aliases)}。",
                payload={"kind": "dimension", "dimension_name": dimension.name, "column": dimension.column},
            )
        )
        next_id += 1
        for value in dimension.values:
            documents.append(
                VectorDocument(
                    id=next_id,
                    text=f"字段取值：{dimension.name}的{dimension.column}等于{value}。",
                    payload={
                        "kind": "value",
                        "dimension_name": dimension.name,
                        "column": dimension.column,
                        "value": value,
                    },
                )
            )
            next_id += 1
    return tuple(documents)


class VectorMetadataRetriever:
    """基于 Qdrant 结果构造与词法检索相同的最小元数据上下文。"""

    def __init__(
        self,
        catalog: MetadataCatalog,
        embeddings: EmbeddingProvider,
        index: QdrantMetadataIndex,
        min_score: float = 0.45,
    ) -> None:
        self.catalog = catalog
        self.embeddings = embeddings
        self.index = index
        self.min_score = min_score

    def retrieve(self, question: str) -> RetrievedMetadata:
        matches = self.index.search(self.embeddings.embed_query(question), limit=12)
        relevant_matches = [match for match in matches if match.score >= self.min_score]
        metric = self._first_metric(relevant_matches)
        if not metric:
            raise ValueError("向量检索未找到可信的业务指标，请换一种表达或补充指标元数据。")

        filters: list[tuple[Dimension, str]] = []
        dimensions: list[Dimension] = []
        for match in relevant_matches:
            if match.payload.get("kind") != "value":
                continue
            dimension = self._dimension_by_column(match.payload["column"])
            value = match.payload["value"]
            if (dimension, value) not in filters:
                filters.append((dimension, value))
            if dimension not in dimensions:
                dimensions.append(dimension)

        group_dimension = self._group_dimension(question, relevant_matches)
        if group_dimension and group_dimension not in dimensions:
            dimensions.append(group_dimension)
        return RetrievedMetadata(
            metric=metric,
            dimensions=tuple(dimensions),
            filters=tuple(filters),
            group_dimension=group_dimension,
        )

    def close(self) -> None:
        self.index.close()

    def _first_metric(self, matches: list) -> Metric | None:
        for match in matches:
            if match.payload.get("kind") == "metric":
                return self._metric_by_name(match.payload["metric_name"])
        return None

    def _group_dimension(self, question: str, matches: list) -> Dimension | None:
        if not any(marker in question for marker in ("各", "按", "分别", "每个")):
            return None
        for match in matches:
            if match.payload.get("kind") == "dimension":
                return self._dimension_by_column(match.payload["column"])
        # 对“各地区”等明确表达保留词法兜底，避免向量召回遗漏字段时丢失分组语义。
        return self.catalog.find_group_dimension(question)

    def _metric_by_name(self, name: str) -> Metric:
        return next(metric for metric in self.catalog.metrics if metric.name == name)

    def _dimension_by_column(self, column: str) -> Dimension:
        return next(dimension for dimension in self.catalog.dimensions if dimension.column == column)


def create_vector_retriever(catalog: MetadataCatalog, index_path: Path) -> VectorMetadataRetriever:
    embeddings = DashScopeEmbeddingProvider.from_environment()
    index = QdrantMetadataIndex(index_path, vector_size=embeddings.dimensions)
    if not index.exists():
        index.close()
        raise RuntimeError("未找到本地向量索引。请先运行 scripts/build_vector_index.py。")
    return VectorMetadataRetriever(catalog=catalog, embeddings=embeddings, index=index)
