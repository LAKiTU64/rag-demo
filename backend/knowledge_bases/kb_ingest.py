#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识库摄取 (Ingestion) 模块

提供将用户上传的 JSON 理论“上限知识”内容转换为可检索的向量块并更新 FAISS 索引的能力。

核心函数:
    flatten_json(obj) -> List[str]
        递归提取 JSON 所有叶子键值对为自然语言片段
    chunk_texts(texts, chunk_size, overlap) -> List[str]
        使用简单窗口切分长文本
    ingest_json_to_faiss(json_str, index_dir, embedding_model) -> dict
        将 JSON 内容添加进现有或新建的 FAISS 索引

索引路径规范:
    默认路径: /workspace/Agent/AI_Agent_Complete/faiss_index

注意:
    - 若已有索引, 将追加新文本 (需要重新构建合并后的向量库)
    - 可以根据需要未来改为增量添加 (目前直接重建)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Dict, Optional

from .faiss_store import build_index, load_index, save_index, ensure_embeddings
import traceback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # fallback
except Exception:
    TfidfVectorizer = None  # type: ignore

DEFAULT_INDEX_DIR = Path("/workspace/Agent/AI_Agent_Complete/faiss_index")

# ------------------ JSON 解析 ------------------

def flatten_json(obj: Any, prefix: str = "") -> List[str]:
    """递归展开 JSON, 将叶子节点转换为文本片段。

    规则:
        - 键路径用 '::' 连接
        - 值为标量(str/int/float/bool)直接拼接
        - 列表: 逐项展开
        - 对象: 递归
    """
    lines: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}{k}" if prefix == "" else f"{prefix}::{k}"
            lines.extend(flatten_json(v, key_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            key_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            lines.extend(flatten_json(item, key_path))
    else:
        # 标量
        scalar = str(obj).strip()
        if scalar:
            lines.append(f"{prefix}: {scalar}")
    return lines

# ------------------ 文本切分 ------------------

def chunk_texts(texts: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """简单窗口切分: 将拼接后的大文本再二次切分，避免过长影响嵌入质量"""
    joined = "\n".join(texts)
    chunks: List[str] = []
    start = 0
    while start < len(joined):
        end = start + chunk_size
        chunk = joined[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap  # 回退 overlap 形成重叠
    return chunks

# ------------------ 摄取逻辑 ------------------

def _is_model_config(obj: Any) -> bool:
    """启发式判断是否为模型配置 JSON: 包含 prompt/gen/hidden/heads 等典型 key 且无长自然语言段。"""
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    score = 0
    for k in keys:
        kl = k.lower()
        if any(x in kl for x in ['prompt', 'gen', 'generate', 'hidden', 'head', 'layer', 'block', 'vocab', 'max', 'batch']):
            score += 1
    # 判定：匹配关键词 ≥3 且没有明显的说明性长文本字段
    long_text_fields = [v for v in obj.values() if isinstance(v, str) and len(v) > 120]
    return score >= 3 and len(long_text_fields) == 0

def _serialize_config(obj: Dict[str, Any]) -> List[str]:
    lines = ["模型配置参数汇总："]
    for k, v in obj.items():
        lines.append(f"- {k}: {v}")
    # 额外生成推理上下文若存在 prompt/gen 字段
    prompt_len = obj.get('prompt_len') or obj.get('prompt_length') or obj.get('prompt')
    gen_len = obj.get('gen_len') or obj.get('generate_len') or obj.get('generation_length') or obj.get('gen')
    if prompt_len and gen_len:
        lines.append(f"该配置针对 prompt 长度 {prompt_len} 与生成长度 {gen_len} 的推理场景，可用于分析注意力开销与 KV 缓存带宽压力。")
    return lines

def _build_tfidf_index(chunks: List[str], index_dir: Path) -> Dict[str, Any]:
    """构建 TF-IDF 备用索引 (无向量召回，仅关键词近似)。"""
    if TfidfVectorizer is None:
        return {"status": "error", "message": "TF-IDF fallback 不可用，未安装 sklearn"}
    try:
        vec = TfidfVectorizer(max_features=4096)
        _ = vec.fit_transform(chunks)
        vocab_path = index_dir / 'tfidf_vocab.json'
        data_path = index_dir / 'tfidf_texts.json'
        vocab_path.write_text(json.dumps(vec.vocabulary_, ensure_ascii=False), encoding='utf-8')
        data_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
        return {
            "status": "success",
            "mode": "tfidf_fallback",
            "fragments_added": len(chunks),
            "index_dir": str(index_dir),
            "embedding_provider": "tfidf_fallback"
        }
    except Exception as e:
        return {"status": "error", "message": f"TF-IDF 构建失败: {e}"}

def ingest_json_to_faiss(json_str: str, index_dir: Optional[Path] = None,
                         embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                         force_tfidf: bool = False,
                         segmentation_mode: str = "window") -> Dict[str, Any]:
    index_dir = index_dir or DEFAULT_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON解析失败: {e}"}
    is_config = _is_model_config(data)
    raw_fragments = _serialize_config(data) if is_config else flatten_json(data)
    # 去重 & 过滤太短文本
    cleaned = []
    seen = set()
    for frag in raw_fragments:
        frag_norm = frag.strip()
        if len(frag_norm) < 4:
            continue
        if frag_norm in seen:
            continue
        seen.add(frag_norm)
        cleaned.append(frag_norm)

    # 二次切分 (可选)
    final_chunks = chunk_texts(cleaned, chunk_size=600, overlap=80)  # segmentation_mode currently window only

    if not final_chunks:
        return {"status": "error", "message": "未生成任何文本片段", "is_config": is_config}

    # 尝试加载嵌入模型，失败则 fallback
    embeddings_ready = True
    if force_tfidf:
        embeddings_ready = False
    else:
        try:
            ensure_embeddings(embedding_model)
        except Exception as emb_err:
            print(f"[WARN] 嵌入模型加载失败，切换 TF-IDF fallback: {emb_err}")
            embeddings_ready = False

    # 如果目录已存在向量索引, 尝试加载并追加 (重建)
    existing = None
    if embeddings_ready:
        try:
            if any(index_dir.iterdir()):
                existing = load_index(index_dir, model_name=embedding_model)
        except Exception:
            existing = None
    else:
        # TF-IDF 模式不做增量合并 (简单覆盖)；也可扩展读取旧文本再合并
        return _build_tfidf_index(final_chunks, index_dir)

    if existing is not None and embeddings_ready:
        # 取出现有文档内容 + 新块重建
        try:
            from langchain_community.vectorstores import FAISS
            # existing.docstore._dict 存储 Document 对象
            old_texts = [doc.page_content for doc in existing.docstore._dict.values()]
            merged = old_texts + final_chunks
            new_store = build_index(merged, model_name=embedding_model)
            save_index(new_store, index_dir)
            return {
                "status": "success",
                "mode": "append-rebuild",
                "fragments_added": len(final_chunks),
                "total_fragments": len(merged),
                "index_dir": str(index_dir),
                "is_config": is_config,
                "embedding_provider": 'modelscope' if embedding_model.startswith('ms:') else 'huggingface'
            }
        except Exception as e:
            return {"status": "error", "message": f"重建索引失败: {e}"}
    else:
        # 新建索引
        try:
            if embeddings_ready:
                store = build_index(final_chunks, model_name=embedding_model)
                save_index(store, index_dir)
                return {
                    "status": "success",
                    "mode": "create",
                    "fragments_added": len(final_chunks),
                    "total_fragments": len(final_chunks),
                    "index_dir": str(index_dir),
                    "is_config": is_config,
                    "embedding_provider": 'modelscope' if embedding_model.startswith('ms:') else 'huggingface'
                }
            else:
                return _build_tfidf_index(final_chunks, index_dir)
        except Exception as e:
            return {"status": "error", "message": f"创建索引失败: {e}"}

def ingest_model_config(json_str: str, index_dir: Optional[Path] = None,
                        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                        force_tfidf: bool = False) -> Dict[str, Any]:
    """专门处理模型配置 JSON (如 h800_prompt1024_gen1024.json) 的摄取。

    行为:
        1. 解析 JSON -> 判定为配置 -> 序列化为自然语言行 (_serialize_config)
        2. 写出原始 config 到 index_dir/model_config_raw.json
        3. 写出序列化文本到 index_dir/model_config_text.txt
        4. 调用 ingest_json_to_faiss 的内部逻辑 (复用) 但跳过 flatten_json 原路径

    返回字段新增: serialized_path, raw_path
    """
    index_dir = index_dir or DEFAULT_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON解析失败: {e}"}
    if not isinstance(data, dict):
        return {"status": "error", "message": "模型配置必须为 JSON 对象 (dict)"}
    # 序列化
    serialized_lines = _serialize_config(data)
    raw_path = index_dir / 'model_config_raw.json'
    text_path = index_dir / 'model_config_text.txt'
    try:
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        text_path.write_text("\n".join(serialized_lines), encoding='utf-8')
    except Exception as e:
        return {"status": "error", "message": f"写入文件失败: {e}"}
    # 用序列化结果组装伪 JSON 以复用 ingest_json_to_faiss 逻辑 (避免重复代码)
    wrapped = json.dumps({"config_text": serialized_lines}, ensure_ascii=False)
    result = ingest_json_to_faiss(wrapped, index_dir=index_dir, embedding_model=embedding_model, force_tfidf=force_tfidf)
    result["raw_path"] = str(raw_path)
    result["serialized_path"] = str(text_path)
    result["is_config"] = True
    result["source"] = "model_config"
    return result

__all__ = [
    'flatten_json', 'chunk_texts', 'ingest_json_to_faiss', 'ingest_model_config', 'ingest_plain_text_to_faiss', 'DEFAULT_INDEX_DIR'
]

def ingest_plain_text_to_faiss(text: str, index_dir: Optional[Path] = None,
                               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    """将一段较长的 Markdown/纯文本写入向量索引 (作为报告或总结回写)。"""
    index_dir = index_dir or DEFAULT_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    # 简单按段落切分
    raw_parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    # 去重
    parts = []
    seen = set()
    for p in raw_parts:
        if p not in seen and len(p) > 10:
            seen.add(p); parts.append(p)
    if not parts:
        return {"status": "error", "message": "无有效文本片段"}
    # 载入旧索引 -> 重建
    existing_texts = []
    try:
        if any(index_dir.iterdir()):
            from .faiss_store import load_index
            store_prev = load_index(index_dir, model_name=embedding_model)
            existing_texts = [d.page_content for d in store_prev.docstore._dict.values()]
    except Exception:
        existing_texts = []
    merged = existing_texts + parts
    try:
        from .faiss_store import build_index, save_index
        new_store = build_index(merged, model_name=embedding_model)
        save_index(new_store, index_dir)
        return {"status": "success", "added": len(parts), "total": len(merged), "index_dir": str(index_dir)}
    except Exception as e:
        return {"status": "error", "message": f"构建失败: {e}"}
