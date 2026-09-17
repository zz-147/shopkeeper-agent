"""将检索、生成、校验、执行组合成可测试的工作流。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import MetadataCatalog
from .database import execute_readonly_query
from .generators import SQLGenerator
from .retrieval import MetadataRetriever
from .safety import SQLSafetyValidator


@dataclass(frozen=True)
class QueryResult:
    question: str
    sql: str
    columns: tuple[str, ...]
    rows: list[tuple[object, ...]]
    sources: tuple[str, ...]


class ShopkeeperService:
    def __init__(
        self,
        database_path: Path,
        generator: SQLGenerator,
        catalog: MetadataCatalog | None = None,
        retriever: MetadataRetriever | None = None,
    ) -> None:
        self.database_path = database_path
        self.catalog = catalog or MetadataCatalog()
        self.generator = generator
        self.retriever = retriever or MetadataRetriever(self.catalog)
        self.validator = SQLSafetyValidator(self.catalog.table_name, self.catalog.allowed_columns)

    def ask(self, question: str) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        retrieved_metadata = self.retriever.retrieve(question)
        raw_sql = self.generator.generate(question, retrieved_metadata.context)
        safe_sql = self.validator.validate(raw_sql)
        columns, rows = execute_readonly_query(self.database_path, safe_sql)
        return QueryResult(
            question=question,
            sql=safe_sql,
            columns=columns,
            rows=rows,
            sources=(*retrieved_metadata.sources, "SQLite：orders"),
        )
