import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
EMBEDDING_MODEL = "./.models/BAAI/bge-small-zh-v1.5"
CHROMA_PATH = "./.chroma_db"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
DEFAULT_SEARCH_K = 3
SIMILARITY_THRESHOLD = 0.5
BEIJING_TZ = timezone(timedelta(hours=8))


class VectorKBManager:
    """
    向量知识库管理类（ChromaDB）：使用本地 Embedding 模型。
    """

    def __init__(self, persist_directory=CHROMA_PATH) -> None:
        self.persist_directory = persist_directory

        # 严格检查本地模型路径
        if not os.path.exists(EMBEDDING_MODEL):
            raise FileNotFoundError(
                f"❌ 找不到本地模型目录: {EMBEDDING_MODEL}。请确保模型已下载到该位置。"
            )

        print(f"🔁 正在从本地加载嵌入模型: {EMBEDDING_MODEL}")

        # 初始化 Embedding：强制开启 local_files_only，禁止联网下载
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,  # 核心改动：禁止任何线上拉取逻辑
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self._load_or_create()

    def _load_or_create(self, is_reset: bool = False) -> None:
        """
        初始化加载。如果本地有数据则读取，否则创建一个空库。
        """
        if is_reset and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"},
        )

        if (
            not os.path.exists(self.persist_directory)
            or self.vectorstore._collection.count() == 0
        ):
            print("🆕 已就绪全新的空向量库")
        else:
            print(f"📦 已从本地加载 ChromaDB: {self.persist_directory}")

    def _get_loader(
        self, file_path: str
    ) -> TextLoader | UnstructuredMarkdownLoader | Docx2txtLoader | PyPDFLoader:
        ext = file_path.split(".")[-1].lower()
        if ext == "txt":
            return TextLoader(file_path, encoding="utf-8")
        elif ext == "md":
            return UnstructuredMarkdownLoader(file_path)
        elif ext == "docx":
            return Docx2txtLoader(file_path)
        elif ext == "pdf":
            return PyPDFLoader(file_path)
        else:
            raise ValueError(f"❌ 不支持的文件格式: {ext}")

    def add_document(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            return

        file_name = os.path.basename(file_path)
        add_time = datetime.now(BEIJING_TZ)

        self.vectorstore.delete(where={"doc_id": file_name})

        try:
            loader = self._get_loader(file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
                separators=[
                    "\n### ",  # 优先按 Kernel / 子模块
                    "\n## ",  # 次级结构
                    "\n\n",  # 段落
                    "\n",  # 行
                    " ",  # 词
                    "",  # 最兜底
                ],
            )
            splits = text_splitter.split_documents(docs)

            for split in splits:
                split.metadata["doc_id"] = file_name
                split.metadata["add_time"] = add_time.isoformat()

            self.vectorstore.add_documents(documents=splits)
            print(f"✅ 文档 '{file_name}' ({len(splits)} 个切片) 已成功入库")

        except Exception as e:
            print(f"❌ 解析文件 {file_name} 出错: {e}")

    def delete_document(self, file_name: str) -> None:
        self.vectorstore.delete(where={"doc_id": file_name})
        print(f"🔥 已物理删除文档: {file_name}")

    def search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        t: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in docs_and_scores:
            if score <= t:
                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "doc_id": doc.metadata.get("doc_id"),
                        "add_time": doc.metadata.get("add_time"),
                        "score": round(float(score), 4),
                    }
                )
        return formatted_results

    def get_overview(self) -> Dict[str, Any]:
        all_data = self.vectorstore.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        if os.path.exists(self.persist_directory):
            ctime = os.path.getctime(self.persist_directory)
            create_time_str = datetime.fromtimestamp(ctime, BEIJING_TZ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            create_time_str = "Unknown"

        doc_stats = {}
        for meta in metadatas:
            did = meta.get("doc_id")
            atime = meta.get("add_time")
            if did:
                if did not in doc_stats or atime > doc_stats[did]:
                    doc_stats[did] = atime

        sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1], reverse=True)
        latest_update = sorted_docs[0][1] if sorted_docs else "N/A"

        print("\n" + "=" * 25 + " 向量库实时概览 " + "=" * 25)
        print(f"📁 路径: {self.persist_directory} | 📅 创建: {create_time_str}")
        print(f"🕒 更新: {latest_update}")
        print(f"📊 规模: {len(metadatas)} 切片 | {len(doc_stats)} 文档")

        if sorted_docs:
            print("📜 文档清单:")
            for name, time in sorted_docs:
                display_time = datetime.fromisoformat(time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"  - {name:<20} | 导入时间: {display_time}")
        else:
            print("📜 文档清单: (空)")
        print("=" * 66 + "\n")

        return {"total_chunks": len(metadatas)}

    def reset_index(self) -> None:
        self._load_or_create(is_reset=True)
        print("✨ 向量库已完成一键重置。")

    def as_retriever(self, **kwargs):
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import PrivateAttr

        class KBRetriever(BaseRetriever):
            _kb_manager: VectorKBManager = PrivateAttr()
            k: int = DEFAULT_SEARCH_K
            t: float = SIMILARITY_THRESHOLD

            def __init__(self, kb_manager, k, t, **data):
                super().__init__(**data)
                self._kb_manager = kb_manager
                self.k = k
                self.t = t

            def _get_relevant_documents(self, query: str) -> List[Document]:
                search_results = self._kb_manager.search(query, k=self.k, t=self.t)
                return [
                    Document(
                        page_content=res["content"],
                        metadata={
                            "doc_id": res["doc_id"],
                            "add_time": res["add_time"],
                            "score": res["score"],
                        },
                    )
                    for res in search_results
                ]

        k = kwargs.get("k", DEFAULT_SEARCH_K)
        t = kwargs.get("t", SIMILARITY_THRESHOLD)
        return KBRetriever(kb_manager=self, k=k, t=t)


if __name__ == "__main__":
    # 1. 初始化（确保模型路径正确）
    try:
        kb = VectorKBManager()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        exit(1)

    # 2. 指定测试目录
    DOCS_DIR = "./documents"
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        with open(os.path.join(DOCS_DIR, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("这是一个本地测试文档。")

    # --- 3. 核心测试：直接遍历并调用 add_document ---
    print(f"\n🚀 开始遍历目录: {DOCS_DIR}")

    for filename in os.listdir(DOCS_DIR):
        full_path = os.path.join(DOCS_DIR, filename)

        # 排除文件夹，只处理文件
        if os.path.isfile(full_path):
            # 直接调用，内部 _get_loader 会处理它不认识的文件格式
            kb.add_document(full_path)

    # 4. 统计与查询
    kb.get_overview()

    print("\n🔍 正在进行检索测试...")
    test_query = "L2缓存命中率低"  # 根据你的实际文档内容调整
    results = kb.search(test_query)

    for res in results:
        print(
            f"📄 来源: {res['doc_id']} | 评分: {res['score']} | 内容: {res['content'][:50]}..."
        )

    print("\n✅ 批量测试流程结束。")
