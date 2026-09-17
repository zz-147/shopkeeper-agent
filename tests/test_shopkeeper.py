from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopkeeper_agent.database import create_demo_database
from shopkeeper_agent.generators import RuleBasedSQLGenerator, load_dotenv_file
from shopkeeper_agent.safety import SQLSafetyValidator
from shopkeeper_agent.service import ShopkeeperService


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


if __name__ == "__main__":
    unittest.main()
