"""元数据三路召回的本地教学实现。

当前采用别名和值的精确匹配，确保可离线测试；生产环境可以保持同一接口，替换为向量检索和全文检索。
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Dimension, MetadataCatalog, Metric


@dataclass(frozen=True)
class RetrievedMetadata:
    metric: Metric
    dimensions: tuple[Dimension, ...]
    filters: tuple[tuple[Dimension, str], ...]
    group_dimension: Dimension | None

    @property
    def context(self) -> str:
        fields = list(self.metric.columns)
        for dimension in self.dimensions:
            if dimension.column not in fields:
                fields.append(dimension.column)
        fields_text = ", ".join(fields)
        filters_text = "；".join(f"{dimension.column} = '{value}'" for dimension, value in self.filters) or "无"
        group_text = self.group_dimension.column if self.group_dimension else "无"
        return (
            "表：orders\n"
            f"已召回字段：{fields_text}\n"
            f"已召回指标：{self.metric.name} = {self.metric.expression}，含义：{self.metric.description}\n"
            f"已召回字段取值：{filters_text}\n"
            f"分组字段：{group_text}"
        )

    @property
    def sources(self) -> tuple[str, ...]:
        sources = [f"指标检索：{self.metric.name}"]
        sources.extend(f"字段检索：{column}" for column in self.metric.columns)
        sources.extend(f"字段检索：{dimension.name}（{dimension.column}）" for dimension in self.dimensions)
        sources.extend(f"字段取值检索：{dimension.name}={value}" for dimension, value in self.filters)
        return tuple(dict.fromkeys(sources))


class MetadataRetriever:
    """按指标、字段、字段取值三类元数据构造最小上下文。"""

    def __init__(self, catalog: MetadataCatalog | None = None) -> None:
        self.catalog = catalog or MetadataCatalog()

    def retrieve(self, question: str) -> RetrievedMetadata:
        metric = self.catalog.find_metric(question)
        filters = tuple(self.catalog.find_filters(question))
        group_dimension = self.catalog.find_group_dimension(question)

        dimensions: list[Dimension] = []
        for dimension, _ in filters:
            if dimension not in dimensions:
                dimensions.append(dimension)
        if group_dimension and group_dimension not in dimensions:
            dimensions.append(group_dimension)

        return RetrievedMetadata(
            metric=metric,
            dimensions=tuple(dimensions),
            filters=filters,
            group_dimension=group_dimension,
        )
