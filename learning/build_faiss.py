import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 建议从专用包导入

# 0. 配置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def build_vector_store(file_path: str, index_save_path: str):
    # 1. 加载文档
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 2. 分割文本
    # 注意：最新的库建议从 langchain_text_splitters 导入
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 中文建议稍微大一点点，保证语境完整
        chunk_overlap=30,
        add_start_index=True,  # 保留原始位置信息，方便溯源
    )
    splits = text_splitter.split_documents(docs)
    print(f"📦 已将文档分割为 {len(splits)} 个代码块")

    # 3. 初始化 Embedding 模型
    # 推荐使用 BGE 系列，对中文支持极好且体积适中
    model_name = "BAAI/bge-small-zh-v1.5"
    encode_kwargs = {"normalize_embeddings": True}  # 归一化，提升检索精度

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # 如果有GPU可改为 'cuda'
        encode_kwargs=encode_kwargs,
    )

    # 4. 构建 FAISS 向量库
    print("🚀 正在构建向量索引，请稍候...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 5. 持久化存储
    vectorstore.save_local(index_save_path)
    print(f"✅ FAISS 向量库已成功保存到: {index_save_path}")


if __name__ == "__main__":
    DATA_FILE = "./learning/test_data.txt"
    SAVE_PATH = "./learning/faiss_index"
    build_vector_store(DATA_FILE, SAVE_PATH)
