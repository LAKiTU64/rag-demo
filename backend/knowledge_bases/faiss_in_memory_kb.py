#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 LangChain + FAISS 构建内存向量知识库的示例脚本。

该脚本展示了以下流程：
1. 准备原始文档并进行切分；
2. 通过嵌入模型生成向量；
3. 基于 FAISS 建立内存向量库；
4. 执行相似度检索与问答。

运行前请确保已安装以下依赖：

    pip install "langchain-community>=0.0.30" "langchain-text-splitters>=0.0.1" "sentence-transformers>=2.3.0"

默认使用 `sentence-transformers/all-MiniLM-L6-v2` 模型生成向量，如需离线部署可提前下载模型文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover - 明确错误提示
    raise ImportError(
        "缺少 LangChain/FAISS 相关依赖，请先安装：\n"
        "  pip install \"langchain-community>=0.0.30\" \"langchain-text-splitters>=0.0.1\" \"sentence-transformers>=2.3.0\"\n"
    ) from exc


@dataclass
class KBConfig:
    """知识库构建配置"""

    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


DEFAULT_CONFIG = KBConfig()


def build_documents(raw_texts: Iterable[str], *, source: str = "manual") -> List[Document]:
    """将原始文本转换为 LangChain Document 列表并进行切分。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CONFIG.chunk_size,
        chunk_overlap=DEFAULT_CONFIG.chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "],
    )

    documents: List[Document] = []
    for idx, text in enumerate(raw_texts):
        metadata = {"source": source, "doc_id": idx}
        chunks = splitter.create_documents([text], metadatas=[metadata])
        documents.extend(chunks)
    return documents


def create_faiss_vector_store(documents: List[Document], *, model_name: Optional[str] = None) -> FAISS:
    """基于文档构建 FAISS 向量存储（严格本地模型）。"""

    model = model_name or DEFAULT_CONFIG.embedding_model
    from .faiss_store import ensure_embeddings
    embeddings = ensure_embeddings(model)
    return FAISS.from_documents(documents, embeddings)


def query_vector_store(store: FAISS, question: str, top_k: int = 3) -> List[dict]:
    """执行相似度检索并返回结构化结果。"""

    results = store.similarity_search_with_score(question, k=top_k)
    formatted = []
    for doc, score in results:
        formatted.append(
            {
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    return formatted


def export_store(store: FAISS, output_dir: Path) -> None:
    """将内存 FAISS 向量库保存到本地目录，便于后续加载调试。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(output_dir))


def run_demo() -> None:
    sample_texts = [
        "LangChain 是一个用于构建基于大语言模型应用的框架，支持多种向量数据库集成。",
        "FAISS 是 Facebook AI 提出的高效相似度搜索库，适合在内存中存储中等规模的向量。",
        "NCU 可用于对 GPU Kernel 进行逐条指令级的性能分析，常与 NSys 输出结合使用。",
    ]

    documents = build_documents(sample_texts, source="demo")
    store = create_faiss_vector_store(documents)

    question = "如何在 LangChain 中使用 FAISS?"
    results = query_vector_store(store, question)

    print("\n【相似度检索结果】")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    export_dir = Path("analysis_results/faiss_demo_store")
    export_store(store, export_dir)
    print(f"\n向量库已保存到: {export_dir.resolve()}")


if __name__ == "__main__":
    run_demo()
