# FaissManager.py

import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import torch

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 严格使用本地嵌入模型（不允许联网下载）
# 请将本地模型放在 EMBEDDING_MODEL 指定的路径，或修改 EMBEDDING_MODEL 值

# --- Config ---
EMBEDDING_MODEL = "./.models/BAAI/bge-small-zh-v1.5"
INDEX_PATH = "./faiss_index"
CHUNK_SIZE = 300  # 如果回答总是“断章取义”，需要把这个值调大；如果发现 LLM 总是找不到重点，可能需要调小。
CHUNK_OVERLAP = 30  # 如果切分后的句子经常出现“前因后果”不连贯，需要调小这个值。
DEFAULT_SEARCH_K = 3
SIMILARITY_THRESHOLD = 0.6  # 如果搜索结果总是“不相关”，需要调小这个值；如果总是“重复”或“完全不对”，需要调大这个值。
SYSTEM_DOC_ID = "system"  # 默认初始化的文档ID
BEIJING_TZ = timezone(timedelta(hours=8))  # 定义东八区时区


class VectorKBManager:
    """
    向量知识库管理类：支持基于文档维度的增、删、查。
    删除策略：采用“软删除标记 + 硬删除重构”的方案。
    """

    def __init__(self, index_path="./faiss_index") -> None:
        self.index_path = index_path
        # 初始化 Embedding：严格使用本地模型（不允许联网下载）
        if not os.path.exists(EMBEDDING_MODEL):
            raise FileNotFoundError(
                f"本地嵌入模型未找到: {EMBEDDING_MODEL}. 请将模型放置在该路径或修改 EMBEDDING_MODEL 配置。"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device, "local_files_only": True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None

        # 【逻辑优化】记录每个文档的“有效起始时间”
        # 键：doc_id, 值：datetime 对象。检索时仅匹配 add_time >= 该时间的切片。
        self.doc_valid_from: Dict[str, datetime] = {}

        # 软删除名单：存储在内存中的文件名集合。即使向量还在索引里，只要在这里面的文件，搜索时都会被过滤掉。
        # 注意：在新的时间戳逻辑下，此集合主要用于标识哪些文档处于完全屏蔽状态。
        self.soft_deleted_sources = set()

        self._load_or_create()

    def _load_or_create(self, is_reset: bool = False) -> None:
        """
        初始化加载。如果本地有索引则读取，否则创建一个空库。
        """

        if os.path.exists(self.index_path) and not is_reset:
            # 加载本地 FAISS 索引
            self.vectorstore = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            # 加载后扫描一遍全库，同步文档的有效时间戳（默认为各个文档现存的最早时间）
            self._sync_valid_times()
            print(f"📦 已从本地加载索引: {self.index_path}")
        else:
            # FAISS 不允许完全空的库存在，所以初始化一个系统级别的占位文档
            # 同时记录库的初始创建时间
            create_time = datetime.now(BEIJING_TZ)
            initial_doc = [
                Document(
                    page_content="init_system_placeholder",
                    metadata={"doc_id": SYSTEM_DOC_ID, "add_time": create_time},
                )
            ]
            self.vectorstore = FAISS.from_documents(initial_doc, self.embeddings)
            print(
                f"🆕 已初始化全新的向量库 (创建时间: {create_time.strftime('%Y-%m-%d %H:%M:%S')})"
            )

    def _sync_valid_times(self) -> None:
        """内部方法：扫描库中所有元数据，初始化有效时间映射"""
        all_docs = self.vectorstore.docstore._dict.values()
        for d in all_docs:
            did = d.metadata.get("doc_id")
            atime = d.metadata.get("add_time")
            if did and did != SYSTEM_DOC_ID:
                # 初始加载时，默认有效起始时间为该文档的最早切片时间
                if did not in self.doc_valid_from or atime < self.doc_valid_from[did]:
                    self.doc_valid_from[did] = atime

    def _get_loader(
        self, file_path: str
    ) -> TextLoader | UnstructuredMarkdownLoader | Docx2txtLoader | PyPDFLoader:
        """
        根据文件后缀返回对应的 LangChain 加载器
        """

        ext = file_path.split(".")[-1].lower()
        if ext == "txt":
            return TextLoader(file_path, encoding="utf-8")
        elif ext == "md":
            # Markdown 建议使用非结构化加载器，能更好处理标题逻辑
            return UnstructuredMarkdownLoader(file_path)
        elif ext == "docx":
            return Docx2txtLoader(file_path)
        elif ext == "pdf":
            return PyPDFLoader(file_path)
        else:
            raise ValueError(f"❌ 不支持的文件格式: {ext}")

    def add_document(self, file_path: str) -> None:
        """
        增加文档：将文件读取、分割并存入向量库。
        这个函数有问题，如果一定要使用FAISS，需要优化逻辑。
        :param file_path: 本地文档路径
        """

        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            return

        file_name = os.path.basename(file_path)
        add_time = datetime.now(BEIJING_TZ)

        # 逻辑保护：如果之前软删除了，现在恢复
        if file_name in self.soft_deleted_sources:
            self.soft_deleted_sources.remove(file_name)

        try:
            # 1. 自动选择加载器并解析
            loader = self._get_loader(file_path)
            docs = loader.load()

            # 2. 文本切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)

            # 3. 统一注入元数据，包括文档ID和添加时间
            for split in splits:
                split.metadata["doc_id"] = file_name
                split.metadata["add_time"] = add_time

            # 4. 入库 (追加模式，O(1) 效率)
            self.vectorstore.add_documents(documents=splits)
            self.vectorstore.save_local(self.index_path)

            # 更新该文档的有效起始时间为当前添加时间，旧版本的切片将自动在检索时被逻辑屏蔽
            self.doc_valid_from[file_name] = add_time

            print(
                f"✅ 文档 '{file_name}' 已入库，旧版已逻辑屏蔽。时间: {add_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except Exception as e:
            print(f"❌ 解析文件 {file_name} 出错: {e}")

    def soft_delete(self, file_name: str) -> None:
        """
        软删除：仅在向量库中记录该文件已“失效”，搜索时会自动跳过。
        """
        # 通过将有效起始时间设为“现在”，逻辑上屏蔽掉之前入库的所有同名切片
        self.doc_valid_from[file_name] = datetime.now(BEIJING_TZ)
        self.soft_deleted_sources.add(file_name)
        print(f"🟡 已软删除（标记屏蔽）: {file_name}，物理数据仍保留，查询已不可见。")

    def hard_delete(self) -> None:
        """
        硬删除：耗时操作，建议定期执行。
        原理：从 docstore 中提取所有未被软删的文档，彻底丢弃已删除数据并重构索引。
        """
        if (
            not self.soft_deleted_sources
            and len(list(self.vectorstore.docstore._dict.values())) > 1
        ):
            # 如果没有软删标记，可以考虑跳过，除非是为了清理历史版本
            print("💡 暂无明确的软删除标记需要物理清理。")

        all_docs = list(self.vectorstore.docstore._dict.values())

        # 过滤逻辑：仅保留 1. 系统文档 2. 没被软删且时间戳符合最新有效时间的切片
        remaining_docs = [
            doc
            for doc in all_docs
            if doc.metadata.get("doc_id") == SYSTEM_DOC_ID
            or (
                doc.metadata.get("doc_id") not in self.soft_deleted_sources
                and doc.metadata.get("doc_id") in self.doc_valid_from
                and doc.metadata.get("add_time")
                >= self.doc_valid_from[doc.metadata.get("doc_id")]
            )
        ]

        if remaining_docs:
            # 彻底重建 FAISS 索引（释放物理空间）
            self.vectorstore = FAISS.from_documents(remaining_docs, self.embeddings)
            self.vectorstore.save_local(self.index_path)
        else:
            # 如果文档被删光了，则重置库
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            self._load_or_create()

        # 清空软删除名单，因为数据已经从物理上抹除了
        self.soft_deleted_sources.clear()
        print("🔥 硬删除完成：索引已重构，物理空间已释放，仅保留最新版本。")

    def search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        t: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        查询：在相似度搜索的基础上增加实时过滤逻辑。
        返回包含内容、来源ID和添加时间的字典列表。
        """

        # 时间戳逻辑过滤函数
        def time_filter(meta):
            did = meta.get("doc_id")
            atime = meta.get("add_time")
            if did == SYSTEM_DOC_ID:
                return False  # 永远不返回占位文档
            if did in self.soft_deleted_sources:
                return False
            # 核心：切片时间必须 >= 该文档要求的有效起始时间
            if did in self.doc_valid_from:
                return atime >= self.doc_valid_from[did]
            return True

        # 使用相似度分值搜索并应用过滤器
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=time_filter
        )

        formatted_results = []
        for doc, score in docs_and_scores:
            if score < t:
                add_time = doc.metadata.get("add_time")
                time_str = (
                    add_time.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(add_time, datetime)
                    else "Unknown"
                )
                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "doc_id": doc.metadata.get("doc_id"),
                        "add_time": time_str,
                        "score": round(float(score), 4),
                    }
                )
        return formatted_results

    def get_overview(self) -> Dict[str, Any]:
        """
        概览：显示当前向量库的状态，包括文档列表和更新统计。
        """
        all_docs = list(self.vectorstore.docstore._dict.values())
        system_doc = next(
            (d for d in all_docs if d.metadata.get("doc_id") == SYSTEM_DOC_ID), None
        )
        create_time_raw = system_doc.metadata.get("add_time") if system_doc else None
        create_time_str = (
            create_time_raw.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(create_time_raw, datetime)
            else "Unknown"
        )

        doc_stats = {}
        for doc in all_docs:
            did = doc.metadata.get("doc_id")
            atime = doc.metadata.get("add_time")
            if did and did != SYSTEM_DOC_ID:
                if did not in doc_stats or (
                    isinstance(atime, datetime) and atime > doc_stats[did]
                ):
                    doc_stats[did] = atime

        sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1], reverse=True)
        latest_update = (
            sorted_docs[0][1].strftime("%Y-%m-%d %H:%M:%S")
            if sorted_docs
            else create_time_str
        )

        print("\n" + "=" * 25 + " 向量库实时概览 " + "=" * 25)
        print(
            f"📁 路径: {self.index_path} | 📅 创建: {create_time_str} | 🕒 更新: {latest_update}"
        )
        print(f"📊 规模: {len(all_docs)-1} 切片 | {len(doc_stats)} 文档")
        for name, time in sorted_docs:
            status = (
                "[正常]"
                if (
                    name not in self.soft_deleted_sources
                    and time >= self.doc_valid_from.get(name, time)
                )
                else "[已屏蔽/过期]"
            )
            print(
                f"  - {name:<20} | 最终版本: {time.strftime('%Y-%m-%d %H:%M:%S')} | {status}"
            )
        print("=" * 66 + "\n")
        return {"documents": sorted_docs}

    def reset_index(self) -> None:
        """
        一键初始化/重置向量库：
        彻底删除磁盘上的索引文件并清空内存状态，恢复到初始空库状态。
        """

        # 1. 物理删除本地索引目录
        if os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
                print(f"🧹 已物理删除本地索引目录: {self.index_path}")
            except Exception as e:
                print(f"⚠️ 删除索引目录失败: {e}")

        # 2. 清空内存中的软删除记录
        self.soft_deleted_sources.clear()
        self.doc_valid_from.clear()

        # 3. 调用初始化方法重新创建空库
        self._load_or_create(is_reset=True)
        print("✨ 向量库已完成一键重置。")
