"""电商智能问数的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from shopkeeper_agent.database import create_demo_database
from shopkeeper_agent.generators import DeepSeekSQLGenerator, RuleBasedSQLGenerator, load_dotenv_file
from shopkeeper_agent.service import ShopkeeperService
from shopkeeper_agent.vector_retrieval import create_vector_retriever


def main() -> int:
    parser = argparse.ArgumentParser(description="将中文电商问题转换为安全 SQL 并执行")
    parser.add_argument("question", help="例如：华北地区销售额")
    parser.add_argument(
        "--model",
        choices=("rule", "deepseek"),
        default="rule",
        help="rule 为离线教学模式；deepseek 需要本机环境变量中的 API Key",
    )
    parser.add_argument(
        "--retrieval",
        choices=("lexical", "vector"),
        default="lexical",
        help="lexical 为本地别名匹配；vector 使用 DashScope Embedding 与本地 Qdrant 索引",
    )
    args = parser.parse_args()

    # .env 只保存在本机，且已被 .gitignore 排除；环境变量优先级更高。
    load_dotenv_file(Path(__file__).parent / ".env")
    database_path = Path(__file__).parent / "data" / "shopkeeper.db"
    create_demo_database(database_path)
    vector_retriever = None
    try:
        generator = RuleBasedSQLGenerator() if args.model == "rule" else DeepSeekSQLGenerator.from_environment()
        service = ShopkeeperService(database_path=database_path, generator=generator)
        if args.retrieval == "vector":
            vector_retriever = create_vector_retriever(service.catalog, Path(__file__).parent / "data" / "qdrant")
            service.retriever = vector_retriever
        result = service.ask(args.question)
    except (ValueError, RuntimeError) as error:
        print(f"无法安全回答：{error}")
        return 1
    finally:
        if vector_retriever:
            vector_retriever.close()

    print(f"问题：{result.question}")
    print(f"元数据来源：{', '.join(result.sources)}")
    print(f"已执行 SQL：{result.sql}")
    print("查询结果：")
    if not result.rows:
        print("未查询到数据")
        return 0

    print(" | ".join(result.columns))
    for row in result.rows:
        print(" | ".join(str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
