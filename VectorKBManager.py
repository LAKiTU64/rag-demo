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

# 使用国内镜像源下载 HuggingFace 模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Config ---
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 300  # 如果回答总是“断章取义”，需要把这个值调大；如果发现 LLM 总是找不到重点，可能需要调小。
CHUNK_OVERLAP = 50  # 如果切分后的句子经常出现“前因后果”不连贯，需要调小这个值。
DEFAULT_SEARCH_K = 3
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值（0-1之间，越小越严苛）。
BEIJING_TZ = timezone(timedelta(hours=8))  # 定义东八区时区


class VectorKBManager:
    """
    向量知识库管理类（ChromaDB）：支持基于文档维度的增、删、查。
    """

    def __init__(self, persist_directory=CHROMA_PATH) -> None:
        self.persist_directory = persist_directory
        # 初始化 Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None

        self._load_or_create()

    def _load_or_create(self, is_reset: bool = False) -> None:
        """
        初始化加载。如果本地有数据则读取，否则创建一个空库。
        """
        if is_reset and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        # 显式指定使用余弦距离 (Cosine Similarity)
        # 注意：Chroma 返回的是距离 distance = 1 - similarity，所以依然是越小越相关
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"},
        )

        # 根据是否存在目录显示状态
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
        """
        根据文件后缀返回对应的 LangChain 加载器
        """
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
        """
        增加文档：如果存在同名文档，则直接覆写。
        :param file_path: 本地文档路径
        """
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            return

        file_name = os.path.basename(file_path)
        add_time = datetime.now(BEIJING_TZ)

        # 覆写：先删除该文档的所有旧切片
        self.vectorstore.delete(where={"doc_id": file_name})

        try:
            # 自动选择加载器并解析
            loader = self._get_loader(file_path)
            docs = loader.load()

            # 文本切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
            )
            splits = text_splitter.split_documents(docs)

            for split in splits:
                split.metadata["doc_id"] = file_name
                split.metadata["add_time"] = add_time.isoformat()

            # 添加新切片
            self.vectorstore.add_documents(documents=splits)
            print(f"✅ 文档 '{file_name}' ({len(splits)} 个切片) 已成功入库/覆盖")

        except Exception as e:
            print(f"❌ 解析文件 {file_name} 出错: {e}")

    def delete_document(self, file_name: str) -> None:
        """
        删除文档：直接从数据库中物理删除该文档的所有切片。
        """
        self.vectorstore.delete(where={"doc_id": file_name})
        print(f"🔥 已物理删除文档: {file_name}")

    def search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        t: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        查询：返回包含内容、来源ID和添加时间的字典列表。
        """
        # 直接搜索，Chroma 内部会处理空库情况
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in docs_and_scores:
            # 应用相似度阈值过滤，在 Cosine Distance 下，score 越小代表越相关
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
        """
        概览：显示当前向量库的状态，包括文档列表和更新统计。
        """
        # 仅获取元数据，避免在大规模库中加载所有文本导致 OOM
        all_data = self.vectorstore.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        # 获取目录创建时间作为“库创建时间”
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
                # 保留该文档最新的时间记录
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
                # 将 iso 格式转回易读格式
                display_time = datetime.fromisoformat(time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"  - {name:<20} | 导入时间: {display_time}")
        else:
            print("📜 文档清单: (空)")
        print("=" * 66 + "\n")

        return {"total_chunks": len(metadatas)}

    def reset_index(self) -> None:
        """
        一键初始化/重置向量库：
        彻底删除磁盘上的索引文件并恢复到初始空库状态。
        """
        self._load_or_create(is_reset=True)
        print("✨ 向量库已完成一键重置。")

    def as_retriever(self, search_kwargs: dict = None):
        """
        返回一个兼容 LangChain 的 Retriever 对象。
        """
        
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        class ChromaRetriever(BaseRetriever):
            def __init__(self, kb_manager, k=DEFAULT_SEARCH_K, t=SIMILARITY_THRESHOLD):
                self.kb_manager = kb_manager
                self.k = k
                self.t = t

            def _get_relevant_documents(self, query: str):
                results = self.kb_manager.search(query, k=self.k, t=self.t)
                docs = [
                    Document(
                        page_content=r["content"],
                        metadata={"doc_id": r["doc_id"], "score": r["score"]},
                    )
                    for r in results
                ]
                return docs

        return ChromaRetriever(self, **(search_kwargs or {}))


if __name__ == "__main__":
    # --- 测试流程 ---
    manager = VectorKBManager()

    # 1. 创建多个测试文档
    files_to_test = {
        "test_f1.txt": "华为是全球领先的 ICT（信息与通信）基础设施和智能终端提供商。",
        "test_f2.md": "# Python简介\nPython 是一种广泛运用于人工智能开发的高级编程语言。",
    }

    print("--- 开始测试：添加文档 ---")
    for filename, content in files_to_test.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        manager.add_document(filename)

    # 2. 测试覆写逻辑（再次添加同名文件）
    print("\n--- 开始测试：覆写文档 ---")
    manager.add_document("test_f1.txt")

    # 3. 搜索展示（测试有效搜索和无效搜索）
    print("\n--- 开始测试：搜索功能 ---")
    test_queries = ["华为", "人工智能", "西瓜"]
    for q in test_queries:
        print(f">>> 搜索关键词: [{q}]")
        res = manager.search(q)
        if not res:
            print("    (无结果)")
        for r in res:
            print(
                f"    内容: {r['content']} | 评分: {r['score']} | 来源: {r['doc_id']}"
            )

    # 4. 查看概览
    manager.get_overview()

    # 5. 删除测试
    print("--- 开始测试：删除文档 ---")
    manager.delete_document("test_f1.txt")
    manager.get_overview()

    # 6. 重置测试
    print("--- 开始测试：重置向量库 ---")
    manager.reset_index()
    manager.get_overview()

    # 清理测试产生的本地文件
    for filename in files_to_test.keys():
        if os.path.exists(filename):
            os.remove(filename)
    # 如果希望测试完彻底删除数据库目录，可以取消下面注释
    # if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
    print("✅ 测试流程结束，临时文件已清理。")
