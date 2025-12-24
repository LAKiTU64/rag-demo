import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用国内镜像源下载 HuggingFace 模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Config
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
INDEX_PATH = "./faiss_index"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


class VectorKBManager:
    """
    向量知识库管理类：支持基于文档维度的增、删、查。
    删除策略：采用“软删除标记 + 硬删除重构”的方案。
    """

    def __init__(self, index_path="./faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None

        # 软删除名单：存储在内存中的文件名集合。即使向量还在索引里，只要在这里面的文件，搜索时都会被过滤掉。
        self.soft_deleted_sources = set()

        self._load_or_create()

    def _load_or_create(self) -> None:
        """
        初始化加载。如果本地有索引则读取，否则创建一个空库。
        """

        if os.path.exists(self.index_path):
            # 加载本地 FAISS 索引
            self.vectorstore = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            print(f"📦 已从本地加载索引: {self.index_path}")
        else:
            # FAISS 不允许完全空的库存在，所以初始化一个系统级别的占位文档
            initial_doc = [
                Document(
                    page_content="init_system_placeholder",
                    metadata={"doc_id": "system"},
                )
            ]
            self.vectorstore = FAISS.from_documents(initial_doc, self.embeddings)
            print("🆕 已初始化全新的向量库")

    def add_document(self, file_path: str) -> None:
        """
        增加文档：将文件读取、分割并存入向量库。
        :param file_path: 本地文档路径
        """
        file_name = os.path.basename(file_path)

        # 逻辑保护：如果该文件之前被软删除了，现在重新添加时应从名单中移除
        if file_name in self.soft_deleted_sources:
            self.soft_deleted_sources.remove(file_name)

        # 1. 加载文本
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        # 2. 文本切分：设置 chunk 块大小和重叠度，确保语义不因切分而丢失
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        # 3. 注入元数据：为每个分片打上doc_id（即文件名），可按doc_id进行管理
        for split in splits:
            split.metadata["doc_id"] = file_name

        # 4. 添加到向量库并持久化
        self.vectorstore.add_documents(documents=splits)
        self.vectorstore.save_local(self.index_path)
        print(f"✅ 文档 '{file_name}' 已入库 (共 {len(splits)} 个切片)")

    def soft_delete(self, file_name: str) -> None:
        """
        软删除：仅在向量库中记录该文件已“失效”，搜索时会自动跳过。
        """
        self.soft_deleted_sources.add(file_name)
        print(f"🟡 已软删除（标记屏蔽）: {file_name}，物理数据仍保留，查询已不可见。")

    def hard_delete(self) -> None:
        """
        硬删除：耗时操作，建议定期执行。
        原理：从 docstore 中提取所有未被软删的文档，彻底丢弃已删除数据并重构索引。
        """
        if not self.soft_deleted_sources:
            print("💡 暂无软删除标记，无需清理。")
            return

        # self.vectorstore.docstore._dict 存储了 ID 到 Document 对象的映射
        all_docs = self.vectorstore.docstore._dict.values()

        # 过滤出需要保留的文档
        remaining_docs = [
            doc
            for doc in all_docs
            if doc.metadata.get("doc_id") not in self.soft_deleted_sources
            and doc.metadata.get("doc_id") != "system"
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
        print("🔥 硬删除完成：索引已重构，过时数据已被物理清除。")

    def search(self, query: str, k: int = 3) -> list[Document]:
        """
        查询：在相似度搜索的基础上增加实时过滤逻辑。
        :param query: 用户提出的问题
        :param k: 返回最相关的结果数量，默认为3
        """

        # 定义过滤函数：检查该文档是否在软删除黑名单中
        def filter_func(metadata):
            return metadata.get("doc_id") not in self.soft_deleted_sources

        # 使用 filter 参数进行后置过滤（Post-filtering）
        results = self.vectorstore.similarity_search(query, k=k, filter=filter_func)
        return results


if __name__ == "__main__":
    # 1. 模拟生成两个测试文件
    with open("doc_recipe_1.txt", "w", encoding="utf-8") as f:
        f.write("红烧肉的秘诀是五花肉要切成3厘米见方的块，加冰糖小火慢炖。")
    with open("doc_recipe_2.txt", "w", encoding="utf-8") as f:
        f.write("回锅肉的关键是先将肉煮至六七成熟，起锅后再切薄片回锅。")

    manager = VectorKBManager()

    # 测试添加
    manager.add_document("doc_recipe_1.txt")
    manager.add_document("doc_recipe_2.txt")

    # 2. 软删除测试：删除“红烧肉”
    print("\n>>> 执行软删除: doc_recipe_1.txt")
    manager.soft_delete("doc_recipe_1.txt")

    # 查询验证：搜红烧肉应该搜不到（或搜到无关内容），搜回锅肉正常
    print("\n>>> 软删除后查询 '红烧肉'：")
    res = manager.search("红烧肉")
    if not res:
        print("（符合预期：未找到相关结果）")
    for doc in res:
        print(f"找到内容: {doc.page_content} | 来源: {doc.metadata['doc_id']}")

    # 3. 硬删除测试：清理存储空间
    print("\n>>> 执行硬删除清理物理空间...")
    manager.hard_delete()

    # 4. 再次查询
    print("\n>>> 最终查询 '回锅肉'：")
    res_final = manager.search("回锅肉")
    for doc in res_final:
        print(f"找到内容: {doc.page_content} | 来源: {doc.metadata['doc_id']}")

    # 现场清理：删除测试用的 txt 文件
    for f in ["doc_recipe_1.txt", "doc_recipe_2.txt"]:
        if os.path.exists(f):
            os.remove(f)
