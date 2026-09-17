"""供前端调用的 FastAPI 接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .database import create_demo_database
from .generators import DeepSeekSQLGenerator, RuleBasedSQLGenerator, load_dotenv_file
from .service import ShopkeeperService
from .vector_retrieval import create_vector_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="例如：华北地区销售额")
    model: Literal["rule", "deepseek"] = Field(default="rule", description="rule 离线可测试；deepseek 需要本机密钥")
    retrieval: Literal["lexical", "vector"] = Field(default="lexical", description="lexical 为别名匹配；vector 需要本地 Qdrant 索引")


class QueryResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    sources: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: str


def create_app(database_path: Path | None = None) -> FastAPI:
    """创建应用；测试时可传入临时数据库，避免测试依赖真实本地文件。"""
    app = FastAPI(
        title="Shopkeeper Agent API",
        version="0.2.0",
        description="将中文电商问题转换为受控的只读 SQL，并返回可追溯查询结果。",
    )
    resolved_database_path = database_path or PROJECT_ROOT / "data" / "shopkeeper.db"
    create_demo_database(resolved_database_path)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        return HealthResponse(status="ok", database=resolved_database_path.name)

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    def query(request: QueryRequest) -> QueryResponse:
        # `.env` 仅在本机加载；环境变量优先级更高，且 .env 已被 Git 忽略。
        load_dotenv_file(PROJECT_ROOT / ".env")
        vector_retriever = None
        try:
            generator = RuleBasedSQLGenerator() if request.model == "rule" else DeepSeekSQLGenerator.from_environment()
            service = ShopkeeperService(database_path=resolved_database_path, generator=generator)
            if request.retrieval == "vector":
                vector_retriever = create_vector_retriever(service.catalog, PROJECT_ROOT / "data" / "qdrant")
                service.retriever = vector_retriever
            result = service.ask(request.question)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if vector_retriever:
                vector_retriever.close()
        return QueryResponse(
            question=result.question,
            sql=result.sql,
            columns=list(result.columns),
            rows=[list(row) for row in result.rows],
            sources=list(result.sources),
        )

    return app


app = create_app()
