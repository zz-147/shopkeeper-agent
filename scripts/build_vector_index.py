"""为元数据构建本地 Qdrant 向量索引。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shopkeeper_agent.catalog import MetadataCatalog
from shopkeeper_agent.embeddings import DASHSCOPE_ENVIRONMENT_KEYS, DashScopeEmbeddingProvider
from shopkeeper_agent.generators import load_dotenv_file
from shopkeeper_agent.vector_retrieval import build_metadata_documents
from shopkeeper_agent.vector_store import QdrantMetadataIndex


def main() -> int:
    load_dotenv_file(PROJECT_ROOT / ".env", override_keys=DASHSCOPE_ENVIRONMENT_KEYS)
    catalog = MetadataCatalog()
    embedding_provider = DashScopeEmbeddingProvider.from_environment()
    documents = build_metadata_documents(catalog)
    vectors = embedding_provider.embed_documents([document.text for document in documents])
    index = QdrantMetadataIndex(PROJECT_ROOT / "data" / "qdrant", embedding_provider.dimensions)
    try:
        index.rebuild(documents, vectors)
    finally:
        index.close()
    print(f"向量索引构建完成：{len(documents)} 条元数据，维度 {embedding_provider.dimensions}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
