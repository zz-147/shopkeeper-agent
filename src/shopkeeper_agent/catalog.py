"""可替换的元数据目录：当前为内存版本，后续可接 Qdrant / Elasticsearch。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    expression: str
    aliases: tuple[str, ...]
    description: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Dimension:
    name: str
    column: str
    aliases: tuple[str, ...]
    values: tuple[str, ...]


class MetadataCatalog:
    """提供“问题相关的指标、维度与表结构”，而不是将全部 schema 交给模型。"""

    table_name = "orders"
    metrics = (
        Metric("销售额", "SUM(payment_amount)", ("销售额", "成交额", "GMV"), "订单实付金额之和", ("payment_amount",)),
        Metric("订单量", "COUNT(*)", ("订单量", "订单数", "订单数量"), "订单记录数", ("order_id",)),
        Metric("客单价", "AVG(payment_amount)", ("客单价", "平均订单金额"), "每笔订单的平均实付金额", ("payment_amount",)),
    )
    dimensions = (
        Dimension("地区", "region", ("地区", "区域"), ("华北", "华东", "华南")),
        Dimension("品类", "category", ("品类", "类目", "分类"), ("电器", "家居", "图书")),
        Dimension("会员等级", "customer_level", ("会员等级", "用户等级"), ("普通", "银卡", "金卡")),
    )

    @property
    def allowed_columns(self) -> frozenset[str]:
        return frozenset({"order_id", "order_date", "region", "category", "customer_level", "payment_amount"})

    def find_metric(self, question: str) -> Metric:
        for metric in self.metrics:
            if any(alias.lower() in question.lower() for alias in metric.aliases):
                return metric
        raise ValueError("未识别到受支持指标。目前支持：销售额、订单量、客单价。")

    def find_filters(self, question: str) -> list[tuple[Dimension, str]]:
        filters: list[tuple[Dimension, str]] = []
        for dimension in self.dimensions:
            for value in dimension.values:
                if value in question:
                    filters.append((dimension, value))
        return filters

    def find_group_dimension(self, question: str) -> Dimension | None:
        for dimension in self.dimensions:
            asks_for_group = any(alias in question for alias in dimension.aliases)
            if asks_for_group and any(marker in question for marker in ("各", "按", "分别", "每个")):
                return dimension
        return None

