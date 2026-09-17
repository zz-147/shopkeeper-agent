from __future__ import annotations

import sys
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopkeeper_agent.database import create_demo_database
from shopkeeper_agent.generators import RuleBasedSQLGenerator, load_dotenv_file
from shopkeeper_agent.retrieval import MetadataRetriever
from shopkeeper_agent.safety import SQLSafetyValidator
from shopkeeper_agent.service import ShopkeeperService
from shopkeeper_agent.api import create_app
from shopkeeper_agent.vector_retrieval import VectorMetadataRetriever, build_metadata_documents
from shopkeeper_agent.vector_store import QdrantMetadataIndex
from fastapi.testclient import TestClient


class ShopkeeperWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "shopkeeper.db"
        create_demo_database(self.database_path)
        self.service = ShopkeeperService(self.database_path, RuleBasedSQLGenerator())

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_region_sales_question_returns_expected_amount(self) -> None:
        result = self.service.ask("华北地区销售额")

        self.assertEqual(result.columns, ("result",))
        self.assertEqual(result.rows, [(3300.0,)])
        self.assertIn("WHERE region = '华北'", result.sql)

    def test_grouped_order_count_returns_all_regions(self) -> None:
        result = self.service.ask("各地区订单量")

        self.assertEqual(result.columns, ("region", "result"))
        self.assertEqual(result.rows, [("华北", 3), ("华东", 2), ("华南", 1)])
        self.assertIn("GROUP BY region", result.sql)

    def test_unknown_metric_is_rejected_instead_of_guessing(self) -> None:
        with self.assertRaisesRegex(ValueError, "未识别到受支持指标"):
            self.service.ask("本月退款率")


class SQLSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = SQLSafetyValidator(
            allowed_table="orders",
            allowed_columns=frozenset({"order_id", "order_date", "region", "category", "customer_level", "payment_amount"}),
        )

    def test_rejects_write_statement(self) -> None:
        with self.assertRaisesRegex(ValueError, "只允许 SELECT"):
            self.validator.validate("DELETE FROM orders")

    def test_rejects_multiple_statements(self) -> None:
        with self.assertRaisesRegex(ValueError, "一条 SQL"):
            self.validator.validate("SELECT order_id FROM orders; DELETE FROM orders")

    def test_accepts_whitelisted_read_query(self) -> None:
        sql = self.validator.validate("SELECT SUM(payment_amount) AS result FROM orders LIMIT 100")
        self.assertEqual(sql, "SELECT SUM(payment_amount) AS result FROM orders LIMIT 100")


class MetadataRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = MetadataRetriever()

    def test_retrieves_only_metadata_needed_for_filtered_sales_question(self) -> None:
        retrieved = self.retriever.retrieve("华北地区销售额")

        self.assertEqual(retrieved.metric.name, "销售额")
        self.assertEqual(retrieved.context.splitlines()[1], "已召回字段：payment_amount, region")
        self.assertIn("region = '华北'", retrieved.context)
        self.assertNotIn("客单价", retrieved.context)
        self.assertNotIn("customer_level", retrieved.context)
        self.assertIn("字段取值检索：地区=华北", retrieved.sources)

    def test_group_question_retrieves_group_dimension_without_value_filter(self) -> None:
        retrieved = self.retriever.retrieve("各品类订单量")

        self.assertEqual(retrieved.metric.name, "订单量")
        self.assertEqual(retrieved.group_dimension.column, "category")
        self.assertIn("已召回字段：order_id, category", retrieved.context)
        self.assertIn("分组字段：category", retrieved.context)
        self.assertNotIn("字段取值检索", " ".join(retrieved.sources))


class StaticEmbeddingProvider:
    dimensions = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "指标：销售额" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [1.0, 0.0]


class VectorMetadataRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = QdrantMetadataIndex(":memory:", vector_size=2)
        self.embeddings = StaticEmbeddingProvider()
        documents = build_metadata_documents(MetadataRetriever().catalog)
        self.index.rebuild(documents, self.embeddings.embed_documents([document.text for document in documents]))
        self.retriever = VectorMetadataRetriever(MetadataRetriever().catalog, self.embeddings, self.index, min_score=0.9)

    def tearDown(self) -> None:
        self.retriever.close()

    def test_semantic_retriever_uses_qdrant_result_for_unlisted_expression(self) -> None:
        retrieved = self.retriever.retrieve("北方营收表现")

        self.assertEqual(self.embeddings.last_query, "北方营收表现")
        self.assertEqual(retrieved.metric.name, "销售额")
        self.assertIn("SUM(payment_amount)", retrieved.context)


class EnvironmentTests(unittest.TestCase):
    def test_dotenv_loader_adds_missing_value_without_overwriting_existing_one(self) -> None:
        previous = os.environ.get("SHOPKEEPER_TEST_KEY")
        try:
            os.environ["SHOPKEEPER_TEST_KEY"] = "from-system"
            with tempfile.TemporaryDirectory() as directory:
                env_file = Path(directory) / ".env"
                env_file.write_text("SHOPKEEPER_TEST_KEY=from-file\nSHOPKEEPER_NEW_KEY=loaded\n", encoding="utf-8")
                load_dotenv_file(env_file)
            self.assertEqual(os.environ["SHOPKEEPER_TEST_KEY"], "from-system")
            self.assertEqual(os.environ["SHOPKEEPER_NEW_KEY"], "loaded")
        finally:
            os.environ.pop("SHOPKEEPER_NEW_KEY", None)
            if previous is None:
                os.environ.pop("SHOPKEEPER_TEST_KEY", None)
            else:
                os.environ["SHOPKEEPER_TEST_KEY"] = previous


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "api.db"
        self.client = TestClient(create_app(database_path))

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_health_check_reports_ready(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "database": "api.db"})

    def test_query_returns_sql_results_and_sources(self) -> None:
        response = self.client.post("/query", json={"question": "华北地区销售额"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["columns"], ["result"])
        self.assertEqual(payload["rows"], [[3300.0]])
        self.assertIn("WHERE region = '华北'", payload["sql"])
        self.assertIn("指标检索：销售额", payload["sources"])
        self.assertIn("SQLite：orders", payload["sources"])

    def test_unknown_metric_returns_explicit_client_error(self) -> None:
        response = self.client.post("/query", json={"question": "退款率"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("未识别到受支持指标", response.json()["detail"])

    def test_blank_question_is_rejected_by_request_schema(self) -> None:
        response = self.client.post("/query", json={"question": ""})

        self.assertEqual(response.status_code, 422)

    def test_deepseek_configuration_error_returns_client_error(self) -> None:
        with patch(
            "shopkeeper_agent.api.DeepSeekSQLGenerator.from_environment",
            side_effect=RuntimeError("未检测到 DEEPSEEK_API_KEY"),
        ):
            response = self.client.post("/query", json={"question": "华北地区销售额", "model": "deepseek"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("未检测到 DEEPSEEK_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
