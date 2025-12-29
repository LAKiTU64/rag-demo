#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAISS 向量知识库封装

提供统一的构建 / 加载 / 查询接口，便于在 LangChain Agent 中启用检索增强 (RAG)。

功能:
1. build_index(texts) -> 创建并返回内存 FAISS 向量库
2. save_index(store, path) -> 保存到本地目录
3. load_index(path, embedding_model) -> 加载已保存的向量库
4. query(store, question, top_k) -> 返回结构化检索结果
5. ensure_embeddings(model_name) -> 延迟加载嵌入模型 (复用实例，降低开销)

依赖:
    pip install "langchain-community>=0.0.30" "langchain-text-splitters>=0.0.1" "sentence-transformers>=2.3.0" "faiss-cpu"

注意:
    - 如果后续要换成其它向量库 (Chroma / Milvus / PGVector)，保持统一接口即可。
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import math, hashlib
from dataclasses import dataclass
import os
import logging
import torch

logger = logging.getLogger(__name__)

try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:
    raise ImportError(
        "缺少向量检索相关依赖，请安装:\n"
        "  pip install langchain-community langchain-text-splitters sentence-transformers faiss-cpu\n"
    ) from exc

@dataclass
class FAISSConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

_DEFAULT_CFG = FAISSConfig()
_embeddings_cache: Dict[str, Any] = {}

class LocalHashEmbeddings:
    """极简本地嵌入实现 (无下载): 通过 token hash 分布生成固定维度向量.

    模型名使用:
      - local-simple  -> 维度256
      - local:<dim>   -> 指定维度 (例如 local:384)

    仅用于离线粗粒度相似度, 不具备真实语义能力, 但可避免外部依赖报错。
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        import re
        return [t for t in re.split(r"[\s,.;:!?，。；：！?]+", text) if t]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._tokens(text.lower())
        for i, tok in enumerate(tokens):
            h = hashlib.sha256(tok.encode('utf-8')).hexdigest()
            pos = (int(h[:8], 16) ^ int(h[8:16], 16)) % self.dim
            weight = 1.0 / (math.sqrt(i + 1))
            vec[pos] += weight
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

# ---------------------- 构建与加载 ----------------------

def ensure_embeddings(model_name: Optional[str] = None):
    """延迟加载 embeddings（严格本地模式）：
    - 支持 'local-simple' 或 'local:<dim>'（内置哈希嵌入，便于离线/测试）
    - 支持传入本地 HuggingFace 模型目录路径（必须存在），将以 local_files_only 模式加载
    - 不再尝试从 ModelScope 或 HuggingFace 在线下载，若未找到会抛出错误
    """
    name = model_name or _DEFAULT_CFG.embedding_model
    if name in _embeddings_cache:
        return _embeddings_cache[name]

    # 本地 embedding 分支（内置哈希）
    if name.startswith('local:') or name == 'local-simple':
        dim = 256
        if name.startswith('local:'):
            try:
                dim = int(name.split(':', 1)[1])
            except Exception:
                dim = 256
        emb = LocalHashEmbeddings(dim=dim)
        _embeddings_cache[name] = emb
        logger.info(f"[LocalEmb] 使用本地哈希嵌入 dim={dim}")
        return emb

    # 如果是本地路径，直接以 local_files_only 加载
    if os.path.exists(name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb = HuggingFaceEmbeddings(model_name=name, model_kwargs={"local_files_only": True, "device": device})
        _embeddings_cache[name] = emb
        logger.info(f"[LocalHF] 使用本地 HuggingFace 嵌入: {name} (device={device})")
        return emb

    # 其它情况统一报错（不允许在线下载）
    raise RuntimeError(
        f"嵌入模型 '{name}' 未找到或在线下载已被禁用。请提供本地模型路径，或使用 'local-simple' / 'local:<dim>'。"
    )

def _split_texts(raw_texts: Iterable[str], cfg: FAISSConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    docs: List[Document] = []
    for idx, text in enumerate(raw_texts):
        metadata = {"source": "user", "orig_index": idx}
        chunks = splitter.create_documents([text], metadatas=[metadata])
        docs.extend(chunks)
    return docs

def build_index(texts: Iterable[str], model_name: Optional[str] = None, cfg: Optional[FAISSConfig] = None) -> FAISS:
    cfg = cfg or _DEFAULT_CFG
    docs = _split_texts(texts, cfg)
    embeddings = ensure_embeddings(model_name or cfg.embedding_model)
    store = FAISS.from_documents(docs, embeddings)
    return store

def save_index(store: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

def load_index(path: Path, model_name: Optional[str] = None) -> FAISS:
    embeddings = ensure_embeddings(model_name or _DEFAULT_CFG.embedding_model)
    store = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True  # 本地可信环境允许
    )
    return store

# ---------------------- 查询 ----------------------

def query(store: FAISS, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    results = store.similarity_search_with_score(question, k=top_k)
    out = []
    for doc, score in results:
        out.append({
            "score": float(score),
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return out

# ---------------------- 简易 CLI ----------------------

def _demo():
    texts = [
        "LangChain 支持多种向量数据库，包括 FAISS 与 Chroma。",
        "FAISS 适合中等规模的向量检索，支持快速相似度搜索。",
        "RAG (检索增强生成) 通过检索相关片段提升回答的准确性。"
    ]
    store = build_index(texts)
    ans = query(store, "什么是 RAG?")
    from pprint import pprint
    pprint(ans)

if __name__ == "__main__":
    _demo()
