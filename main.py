# 在文件最顶部添加，确保在导入 Chroma 之前执行
import os
import sys

# 使用国内镜像源下载 HuggingFace 模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 尝试使用 pysqlite3 替代系统 sqlite3
try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except ImportError:
    pass


from VectorKBManager import VectorKBManager

kb = VectorKBManager()
kb.add_document("./projects/rag-demo/test_doc.txt")
print(kb.get_overview())
print(kb.search("什么是制约GPU性能的关键因素"))
