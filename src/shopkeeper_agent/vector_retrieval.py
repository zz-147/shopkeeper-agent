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
        min_metric_score: float = 0.25,
        min_metric_margin: float = 0.05,
    ) -> None:
        self.catalog = catalog
        self.embeddings = embeddings
        self.index = index
        self.min_metric_score = min_metric_score
        self.min_metric_margin = min_metric_margin

    def retrieve(self, question: str) -> RetrievedMetadata:
        query_vector = self.embeddings.embed_query(question)
        metric_matches = self.index.search(query_vector, limit=2, kind="metric")
        metric = self._trusted_metric(metric_matches)

        # SQL 的筛选值必须精确命中白名单，不能仅因语义接近就生成 WHERE 条件。
        filters = list(self.catalog.find_filters(question))
        dimensions: list[Dimension] = []
        for dimension, _ in filters:
            if dimension not in dimensions:
                dimensions.append(dimension)

        group_dimension = self.catalog.find_group_dimension(question)
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

    def _trusted_metric(self, matches: tuple) -> Metric:
        if not matches or matches[0].score < self.min_metric_score:
            raise ValueError("向量检索未找到可信的业务指标，请换一种表达或补充指标元数据。")
        if len(matches) > 1 and matches[0].score - matches[1].score < self.min_metric_margin:
            raise ValueError("向量检索到多个接近的业务指标，请补充更明确的指标描述。")
        return self._metric_by_name(matches[0].payload["metric_name"])

    def _metric_by_name(self, name: str) -> Metric:
        return next(metric for metric in self.catalog.metrics if metric.name == name)

def create_vector_retriever(catalog: MetadataCatalog, index_path: Path) -> VectorMetadataRetriever:
    embeddings = DashScopeEmbeddingProvider.from_environment()
    index = QdrantMetadataIndex(index_path, vector_size=embeddings.dimensions)
    if not index.exists():
        index.close()
        raise RuntimeError("未找到本地向量索引。请先运行 scripts/build_vector_index.py。")
    return VectorMetadataRetriever(catalog=catalog, embeddings=embeddings, index=index)
