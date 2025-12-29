# LangChain + FAISS 向量知识库集成说明

本说明文档解释如何在当前项目中使用 FAISS 构建与查询向量知识库，并在 `LangChainAgent` 中开启检索增强 (RAG)。

## 1. 依赖安装
确保已安装以下依赖（已写入根目录 `requirements.txt`）:

```text
langchain-community>=0.0.30
langchain-text-splitters>=0.0.1
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4
```

如果尚未安装：
```bash
pip install -r requirements.txt
```

## 2. 快速构建向量索引
使用封装模块 `faiss_store.py`：

```python
from pathlib import Path
from backend.knowledge_bases.faiss_store import build_index, save_index

texts = [
    "LangChain 支持多种向量数据库，包括 FAISS 与 Chroma。",
    "FAISS 适合中等规模的向量检索，支持快速相似度搜索。",
    "RAG (检索增强生成) 通过检索相关片段提升回答的准确性。"
]

store = build_index(texts)
save_index(store, Path("/workspace/Agent/AI_Agent_Complete/faiss_index"))
```

## 3. 在 Agent 中启用检索增强
`langchain_agent.py` 已增加以下初始化参数:

| 参数 | 说明 |
|------|------|
| enable_faiss | 是否启用向量检索功能 (布尔) |
| faiss_embedding_model | 嵌入模型名称，可覆盖默认 `sentence-transformers/all-MiniLM-L6-v2` |
| faiss_index_dir | 已保存索引目录路径，若为空可后续通过 `add_texts_to_faiss` 构建 |

示例：
```python
from langchain_agent import LangChainAgent

agent = LangChainAgent(
    use_openai=False,
    enable_faiss=True,
    faiss_index_dir="/workspace/Agent/AI_Agent_Complete/faiss_index"
)
```

如果没有预构建索引，可以动态添加文本：
```python
resp = agent.add_texts_to_faiss([
    "性能分析可以结合 nsys 与 ncu",
    "FAISS 使用倒排向量结构进行近似最近邻搜索"
])
print(resp)
```

## 4. 查询与检索
在消息中包含“检索 / 查询 / search / retrieve / 向量 / 知识库”任一关键词时，Agent 会对当前向量库执行查询：

```python
result = agent.query_faiss("什么是 FAISS?", top_k=3)
for item in result:
    print(item["score"], item["content"])
```

或通过对话：
```python
await agent.process_message("检索 FAISS 是什么")
```

返回结果示例：
```json
[
  {
    "score": 0.1234,
    "content": "FAISS 适合中等规模的向量检索，支持快速相似度搜索。",
    "metadata": {"source": "user", "orig_index": 1}
  }
]
```

## 5. 嵌入模型替换
如果需要使用更高质量嵌入 (如 `sentence-transformers/all-mpnet-base-v2`):
```python
agent = LangChainAgent(enable_faiss=True, faiss_embedding_model="sentence-transformers/all-mpnet-base-v2")
```
首次使用会自动下载模型，建议提前在构建阶段预拉取。

## 6. 常见问题
| 问题 | 说明 | 解决 |
|------|------|------|
| ImportError | 缺少相关依赖 | 安装 requirements.txt 中依赖 |
| 检索结果为空 | 向量库未构建或语义距离过大 | 添加更多领域文本或调整 chunk_size |
| 模型下载慢 | 外网访问限制 | 预先下载模型并指向本地缓存目录 |

## 7. 后续扩展建议
- 增加多向量源合并 (文档 + 配置 + 代码片段)
- 增加对检索结果的基于 LLM 的总结 (RAG answer)
- 引入持久化 Chroma / Milvus 替代内存 FAISS
- 为检索加入过滤条件 (metadata filter)

## 8. 最简端到端示例
```python
from langchain_agent import LangChainAgent

agent = LangChainAgent(enable_faiss=True)
agent.add_texts_to_faiss([
    "LangChain 可以与 FAISS 集成。",
    "FAISS 用于快速向量相似度检索。"
])
print(agent.query_faiss("FAISS 是什么", top_k=2))
```

---
如需将检索结果接入后端 WebSocket 或前端展示，可直接读取保存的索引目录或封装一个 `/api/knowledge/search` HTTP 接口。欢迎继续提出需要的扩展功能。
