#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model Intelligence Utilities

解析用户输入中的模型路径/名称 (Qwen, Llama 等), 抽取模型家族与参数规模并生成预取理论检索片段。

暴露函数:
    extract_model_info(text: str) -> dict
    build_theory_queries(model_info: dict) -> list[str]
    prefetch_theory_snippets(model_info: dict, top_k: int = 4) -> dict

策略:
1. 使用正则匹配常见模型族 + 尺寸 (7B, 13B, 32B, 70B, 110B ...)
2. 标准化 family (小写) 与 size (数字B)
3. 构造检索 query 模板并调用 FAISS（如果存在）
"""
from __future__ import annotations
import re
from typing import List, Dict, Any
from pathlib import Path

FAMILY_PATTERNS = [
    r"qwen[-_ ]?(\d+\w*)", r"llama[-_ ]?(\d+\w*)", r"chatglm[-_ ]?(\d+\w*)",
    r"baichuan[-_ ]?(\d+\w*)", r"internlm[-_ ]?(\d+\w*)", r"mistral[-_ ]?(\d+\w*)", r"yi[-_ ]?(\d+\w*)"
]

SIZE_EXTRACT = re.compile(r"(\d+)(b|B|k|K|m|M)?")

def _norm_size(raw: str) -> float:
    m = SIZE_EXTRACT.search(raw)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2) or 'B'
    if unit.lower() == 'b':
        return val
    if unit.lower() == 'm':
        return val / 1000.0
    if unit.lower() == 'k':
        return val / 1_000_000.0
    return val

def extract_model_info(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    matches = []
    for pat in FAMILY_PATTERNS:
        for m in re.finditer(pat, lowered):
            matches.append(m)
    family = None; size_b = None; raw_match = None
    if matches:
        # 取最长匹配
        m = sorted(matches, key=lambda x: len(x.group(0)), reverse=True)[0]
        raw_match = m.group(0)
        # 家族: 去除尺寸部分
        fam = re.split(r"[-_ ]", raw_match)[0]
        family = re.sub(r"\d+.*", "", fam)  # 去掉后续数字
        size_fragment = m.group(1)
        size_b = _norm_size(size_fragment)
    # 参数量估计 (B)
    param_est = size_b * 1e9 if size_b else None
    suggestions = []
    if family and size_b:
        if size_b <= 8:
            suggestions.append("可尝试更高 batch_size (>=16) 观察吞吐扩展")
        elif size_b <= 32:
            suggestions.append("建议开启 FlashAttention / fused kernels 降低注意力开销")
        else:
            suggestions.append("优先确认张量并行/流水并行策略, 避免单 GPU 过度溢出")
    return {
        "detected": bool(family),
        "raw_match": raw_match,
        "family": family,
        "size_b": size_b,
        "param_estimate": param_est,
        "heuristic_suggestions": suggestions
    }

def build_theory_queries(model_info: Dict[str, Any]) -> List[str]:
    if not model_info.get('detected'):
        return ["transformer performance optimization", "gpu kernel bottleneck memory bandwidth"]
    family = model_info['family']
    size_b = model_info['size_b']
    return [
        f"{family} {int(size_b)}B architecture bottlenecks",
        f"{family} {int(size_b)}B memory bandwidth optimization",
        f"{family} decoder attention optimization",
        f"{family} {int(size_b)}B throughput scaling limits",
    ]

def _faiss_ready() -> bool:
    try:
        from knowledge_bases.faiss_store import load_index
    except Exception:
        return False
    return Path('/workspace/Agent/AI_Agent_Complete/faiss_index').exists()

def prefetch_theory_snippets(model_info: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
    queries = build_theory_queries(model_info)
    results_map: Dict[str, List[str]] = {}
    if not _faiss_ready():
        return {"queries": queries, "snippets": results_map, "available": False}
    try:
        from knowledge_bases.faiss_store import load_index, query as faiss_query
        store = load_index(Path('/workspace/Agent/AI_Agent_Complete/faiss_index'), model_name='sentence-transformers/all-MiniLM-L6-v2')
        for q in queries:
            hits = faiss_query(store, q, top_k=top_k)
            results_map[q] = [h['text'][:300] for h in hits]
    except Exception:
        pass
    return {"queries": queries, "snippets": results_map, "available": True}

__all__ = [
    'extract_model_info', 'build_theory_queries', 'prefetch_theory_snippets'
]