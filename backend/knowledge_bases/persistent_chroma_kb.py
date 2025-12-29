#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Chroma 构建可持久化向量知识库的示例脚本。

流程概览：
1. 将原始文本切分为文档片段；
2. 借助嵌入模型生成向量并写入 Chroma；
3. 将向量库持久化到磁盘目录；
4. 再次加载后执行相似度检索。

运行前需要安装：

    pip install "langchain-community>=0.0.30" "langchain-text-splitters>=0.0.1" "sentence-transformers>=2.3.0" "chromadb>=0.4.22"

建议在 GPU/CPU 均可使用的环境中运行，离线部署时可提前下载嵌入模型。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 Chroma / LangChain 相关依赖，请先安装：\n"
        "  pip install \"langchain-community>=0.0.30\" \"langchain-text-splitters>=0.0.1\" \"sentence-transformers>=2.3.0\" \"chromadb>=0.4.22\"\n"
    ) from exc

# 兼容包内导入和脚本独立运行
try:  # pragma: no cover - 脚本执行路径不同
    from .faiss_in_memory_kb import DEFAULT_CONFIG
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from faiss_in_memory_kb import DEFAULT_CONFIG  # type: ignore


@dataclass
class PersistentKBConfig:
    """持久化向量库配置"""

    persist_directory: Path = Path("analysis_results/chroma_demo_store")
    collection_name: str = "ai_agent_collection"
    embedding_model: str = DEFAULT_CONFIG.embedding_model
    chunk_size: int = DEFAULT_CONFIG.chunk_size
    chunk_overlap: int = DEFAULT_CONFIG.chunk_overlap


class PersistentKnowledgeBase:
    """封装 Chroma 向量库的构建、持久化与加载流程。"""

    def __init__(self, config: Optional[PersistentKBConfig] = None):
        self.config = config or PersistentKBConfig()
        # 使用严格的本地嵌入加载逻辑（来自 faiss_store.ensure_embeddings）
        from .faiss_store import ensure_embeddings
        self.embeddings = ensure_embeddings(self.config.embedding_model)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " "],
        )
        self.store: Optional[Chroma] = None

    def build(self, texts: Iterable[str], *, source: str = "manual") -> Chroma:
        """从原始文本构建新的 Chroma 向量库。"""

        documents: List[Document] = []
        for idx, text in enumerate(texts):
            metadata = {"source": source, "doc_id": idx}
            chunks = self._splitter.create_documents([text], metadatas=[metadata])
            documents.extend(chunks)

        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        self.store = Chroma.from_documents(
            documents,
            embedding=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_directory),
        )
        self.store.persist()
        return self.store

    def load(self) -> Chroma:
        """从磁盘目录加载已有向量库。"""

        self.store = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_directory),
        )
        return self.store

    def similarity_search(self, query: str, top_k: int = 3) -> List[dict]:
        if not self.store:
            raise RuntimeError("向量库尚未构建或加载，请先调用 build() 或 load().")

        results = self.store.similarity_search_with_score(query, k=top_k)
        payload = []
        for doc, score in results:
            payload.append(
                {
                    "score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return payload


def run_demo() -> None:
    manager = PersistentKnowledgeBase()

    sample_texts = [
        "Chroma 提供轻量级的本地持久化机制，适合在开发环境中快速测试向量检索。",
        "LangChain 可以与 Chroma 搭配，实现端到端的 Retrieval-Augmented Generation (RAG) 工作流。",
        "在生产部署中，可以将模型输出和性能报告写入向量库，支持快速检索历史案例。",
    ]

    print("\n【开始构建持久化知识库】")
    store = manager.build(sample_texts, source="demo")
    print(f"持久化目录: {manager.config.persist_directory.resolve()}")
    print(f"Collection 名称: {manager.config.collection_name}")
    try:
        vector_count = store._collection.count()
    except AttributeError:  # pragma: no cover - 不同版本返回结构不同
        vector_count = len(store.get()["documents"])
    print(f"当前向量数量: {vector_count}")

    print("\n【重新加载并执行检索】")
    manager.load()
    question = "如何把 LangChain 和 Chroma 结合?"
    answers = manager.similarity_search(question)
    print(json.dumps(answers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()
