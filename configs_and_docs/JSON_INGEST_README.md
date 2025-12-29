# JSON 理论知识库摄取指南

本指南说明如何将结构化的“理论上限”或性能公式等 JSON 内容上传到系统并构建 FAISS 向量索引用于 RAG 检索增强。

## 功能概述
上传包含层级结构的 JSON 文件 (例如: 计算公式、硬件参数、优化策略、复杂度结论)。系统将:
1. 递归展开为扁平片段 (`键路径::子键: 值`)
2. 过滤重复与过短内容
3. 二次按窗口切分形成嵌入块
4. 构建或重建 FAISS 索引并持久化到 `faiss_index/`
5. LangChain Agent 可通过检索指令使用这些向量信息辅助回答

## 上传方式与嵌入来源
前端页面侧边栏 "📚 理论上限知识 (JSON) 上传":
1. 选择 / 拖拽单个 `.json`
2. 选择嵌入模型 (默认 MiniLM-L6-v2，可改为 ModelScope 前缀)
3. 查看摄取统计 (新增碎片数 / 总碎片 / provider)

后端接口 (`POST /knowledge/upload`):
```json
{
  "raw_json": { "section": "Intro", "text": "测试内容" },
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### 嵌入模型来源选择
支持三类：
- HuggingFace (默认): 例如 `sentence-transformers/all-MiniLM-L6-v2`
- ModelScope: 使用 `ms:<model-id>` 前缀，例如 `ms:damo/nlp_gte-base-zh` (需 `pip install modelscope`，离线环境推荐)
- TF-IDF fallback: 当嵌入模型加载失败或后续开放的 `force_tfidf=true` 时自动启用，返回 `embedding_provider=tfidf_fallback`

### 强制 TF-IDF 模式
无需任何外部模型，仅基于关键词统计：
1. 接口参数：`{"force_tfidf": true}`
2. 或设置环境变量：`OFFLINE_FORCE_TFIDF=1` (容器 / 进程启动前导出)，即可使所有未显式指定的上传走 TF-IDF。
3. 返回字段包含：`force_tfidf: true` 与 `embedding_provider: tfidf_fallback`
适用于完全离线且不希望出现模型加载报错的场景。

返回示例 (ModelScope):
```json
{
  "status": "success",
  "mode": "create",
  "fragments_added": 24,
  "embedding_model": "ms:damo/nlp_gte-base-zh",
  "embedding_provider": "modelscope"
}
```
返回示例 (TF-IDF 回退):
```json
{
  "status": "success",
  "mode": "tfidf_fallback",
  "fragments_added": 24,
  "embedding_provider": "tfidf_fallback"
}
```

调试模式：加入 `"debug": true` 可在失败时返回诊断信息。

## 目录与持久化
索引目录默认: `/workspace/Agent/AI_Agent_Complete/faiss_index`
包含:
- `index.faiss` 向量数据
- `index.pkl` Document 与元数据

## 触发检索的关键词
在聊天中包含以下任意关键词将触发向量检索辅助 (前后可组合):
```
检索, 查询, search, retrieve, 向量, 知识库
```

## 索引刷新
上传完成后调用 Agent 的 `reload_faiss_index()` (若在运行时需要即时生效)。目前 WebSocket 层可在特定指令中加入刷新逻辑，或人工在后端触发。

## 设计细节
- 重建策略: 追加新 JSON 时读取旧索引所有内容 + 新块合并重建 (后续可优化为增量添加)
- 扁平化路径分隔符: `::` 便于保留层级语义
- 文本窗口切分: chunk_size=600, overlap=80 (可按实际嵌入模型上下文窗口调整)
- 去重: 基于完整字符串匹配

## 异常处理与常见 400 原因
400 错误通常由以下原因之一触发：
1. JSON 解析失败 (`JSON解析失败`)：raw_json 字段字符串非法或内容不是有效 JSON。
2. 缺少内容：未提供 `file`、`raw_json` 或主体 JSON 对象 (`缺少 JSON 内容`)。
3. 片段为空：所有值过短 (<4) 或经清洗后没有可用文本 (`未生成任何文本片段`)。
4. 嵌入模型路径合法但本地 / 离线无法下载且 TF-IDF 构建又失败 (极少出现，需安装 sklearn)。

排查步骤：
```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -H 'Content-Type: application/json' \
  -d '{"raw_json": {"a": "短"}, "debug": true}'
```
查看响应中的 `diagnostics`、`root_keys`、`embedding_model`。

建议：提供更长的描述性文本，提高碎片生成有效性；若离线请优先使用 `ms:` 前缀模型。

## 示例 JSON
```json
{
  "theoretical_limits": {
    "memory_bandwidth": { "GPU_A": "1200 GB/s", "GPU_B": "900 GB/s" },
    "latency_models": { "attention": "O(n^2)", "flash_attention": "降低常数项" },
    "scaling": { "tokens_vs_params": "Chinchilla scaling", "compute_opt": "FLOPs 平衡" }
  }
}
```

上传后将生成类似扁平片段:
```
theoretical_limits::memory_bandwidth::GPU_A: 1200 GB/s
theoretical_limits::latency_models::attention: O(n^2)
...
```

## 后续扩展建议
- 支持多文件合并上传
- 增加增量添加 API 避免全量重建
- 片段来源元数据存储 (文件名、时间戳、类别标签)
- 添加语义过滤与分类标签
- 为检索结果在前端提供高亮展示区域

## 快速测试
```bash
python backend/tests/test_kb_ingest.py
```
应看到 `status=success` 且目录生成。若在离线环境测试：
```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -H 'Content-Type: application/json' \
  -d '{"raw_json": {"section": "Intro", "text": "这是用于测试的离线环境摄取"}, "embedding_model": "ms:damo/nlp_gte-base-zh"}'
```
返回应包含 `embedding_provider: modelscope`。

---
如需进一步增强，请查看 `knowledge_bases/kb_ingest.py` 与 `langchain_version/langchain_agent.py` 的相关方法。
