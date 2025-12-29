#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangChain Synthesis Utilities

将性能分析结果 (comprehensive_analysis.json + advanced_performance_report.md) 与知识库检索片段融合，
通过 LangChain (若可用) 生成终极综合报告摘要与行动建议。

主要入口:
    synthesize_final_report(perf_dir: Path, queries: List[str]|None) -> Dict[str, Any]

返回结构:
{
  'markdown_path': str,
  'summary': str,
  'kb_hits': Dict[str, List[str]],
  'model_info': Dict[str, Any]
}

容错策略:
1. 若 LangChain 不可用 => 使用模板拼接 fallback 总结
2. 若 FAISS 不可用 => kb_hits 为空
3. 若 comprehensive/advanced 报告缺失 => 附加警告
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.prompts import ChatPromptTemplate  # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    from knowledge_bases.faiss_store import load_index, query as kb_query  # type: ignore
except Exception:
    load_index = None
    kb_query = None

try:
    from backend.model_intel import extract_model_info, build_theory_queries
except Exception:
    from model_intel import extract_model_info, build_theory_queries  # type: ignore

DEFAULT_FAISS_DIR = Path('/workspace/Agent/AI_Agent_Complete/faiss_index')

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

def _load_text(path: Path) -> str:
    if not path.exists():
        return ''
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return ''

def _faiss_ready() -> bool:
    return load_index is not None and kb_query is not None and DEFAULT_FAISS_DIR.exists()

def _collect_kb_hits(queries: List[str], top_k: int = 3) -> Dict[str, List[str]]:
    if not _faiss_ready():
        return {}
    hits: Dict[str, List[str]] = {}
    try:
        store = load_index(DEFAULT_FAISS_DIR, model_name='sentence-transformers/all-MiniLM-L6-v2')
        for q in queries:
            try:
                res = kb_query(store, q, top_k=top_k)
                hits[q] = [r['text'][:400] for r in res]
            except Exception:
                hits[q] = []
    except Exception:
        pass
    return hits

def _build_prompt(perf_summary: str, kb_hits: Dict[str, List[str]], advanced_excerpt: str) -> str:
    kb_part = '\n'.join([
        f'Query: {q}\n' + '\n'.join([f'- {t}' for t in texts])
        for q, texts in kb_hits.items()
    ]) or '(无知识库检索结果)'
    return (
        "你是一名 GPU / LLM 性能优化专家。请基于以下性能分析摘要与知识库片段，生成结构化精炼总结：\n\n"
        + "[性能摘要]\n" + perf_summary[:4000] + "\n\n"
        + "[高级报告片段]\n" + advanced_excerpt[:3000] + "\n\n"
        + "[知识库检索]\n" + kb_part + "\n\n"
        + "输出格式: \n1. 关键瓶颈概述 (≤5条)\n2. 优先优化行动 (T1/T2等, 每条一句)\n3. 理论支撑要点 (引用知识库摘要)\n4. 预估收益与风险一句话总结\n"
        + "请使用中文。"
    )

def _fallback_summary(perf_summary: str, kb_hits: Dict[str, List[str]], advanced_excerpt: str) -> str:
    top_q = list(kb_hits.keys())[:3]
    return (
        "## 综合摘要 (Fallback)\n\n" +
        f"性能概览: {perf_summary[:300] or '缺失'}...\n\n" +
        "任务优先级: 参考 T1 GEMM 优化 / T2 CUDA Graph / T3 注意力与内存模式 / Fusion。\n\n" +
        "知识库关键词: " + (', '.join(top_q) if top_q else '无') + "\n\n" +
        "建议: 先聚焦最大时间占比 compute kernels, 并并行规划图捕获与内存访问优化, 最后进行融合与 KB 写回。"
    )

def synthesize_final_report(perf_dir: Path, queries: Optional[List[str]] = None, extra_query_text: Optional[str] = None) -> Dict[str, Any]:
    comp = _load_json(perf_dir / 'comprehensive_analysis.json')
    adv_text = _load_text(perf_dir / 'advanced_performance_report.md')
    enriched_text = _load_text(perf_dir / 'integrated_performance_report_enriched.md')
    basic_text = _load_text(perf_dir / 'integrated_performance_report.md')

    kernel_overview = comp.get('nsys_overview', {}).get('kernel_analysis', {})
    hot_list = comp.get('hot_kernels', [])
    perf_parts = []
    if kernel_overview:
        perf_parts.append(
            f"总kernels {kernel_overview.get('total_kernels','?')} | 总时间 {kernel_overview.get('total_kernel_time','?')} ms | 平均 {kernel_overview.get('avg_kernel_time','?')} ms"
        )
    if hot_list:
        perf_parts.append('热点: ' + ', '.join([k.get('name','')[:60] for k in hot_list[:5]]))
    perf_summary = '\n'.join(perf_parts) or '(缺失基础性能数据)'

    model_info = extract_model_info(basic_text + adv_text + (extra_query_text or ''))
    theory_queries = build_theory_queries(model_info)
    if queries:
        theory_queries.extend([q for q in queries if isinstance(q, str)])
    if extra_query_text:
        theory_queries.append(extra_query_text.strip()[:100])

    kb_hits = _collect_kb_hits(theory_queries)

    adv_excerpt = ''
    if adv_text:
        lines = adv_text.splitlines()
        for ln in lines:
            if any(h in ln for h in ['## 1. 热点 Kernel 分类', '## 3. 任务列表', '## 6. 总结']):
                adv_excerpt += ln + '\n'
        adv_excerpt += '\n'.join(lines[-20:])

    if LANGCHAIN_AVAILABLE:
        try:
            prompt_text = _build_prompt(perf_summary, kb_hits, adv_excerpt)
            template = ChatPromptTemplate.from_messages([
                ("system", "你是专业的 GPU / LLM 性能优化顾问"),
                ("human", "{input}")
            ])
            chain = template | ChatOpenAI(temperature=0.2, model='gpt-3.5-turbo')
            resp = chain.invoke({"input": prompt_text})
            summary = resp.content
        except Exception:
            summary = _fallback_summary(perf_summary, kb_hits, adv_excerpt)
    else:
        summary = _fallback_summary(perf_summary, kb_hits, adv_excerpt)

    final_path = perf_dir / 'final_langchain_integrated_report.md'
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write('# 🌐 终极综合性能报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        if enriched_text:
            f.write('> 综合分析增强报告片段\n\n')
            f.write(enriched_text[:15000])
            f.write('\n\n---\n\n')
        f.write('## LangChain 综合摘要\n\n')
        f.write(summary + '\n\n')
        if kb_hits:
            f.write('## 知识库检索片段\n\n')
            for q, texts in kb_hits.items():
                f.write(f'### Query: {q}\n')
                for t in texts:
                    f.write(f'- {t}\n')
                f.write('\n')
        else:
            f.write('## 知识库检索片段\n\n- (无，可能未构建向量索引或离线)\n')
        f.write('\n## 后续建议\n\n')
        f.write('- 扩展 batch/input 参数扫验证瓶颈稳定性\n')
        f.write('- Autotune 主 compute kernels (GEMM)\n')
        f.write('- 引入持续写回管线将报告摄取到向量库\n')
    return {
        'markdown_path': str(final_path),
        'summary': summary,
        'kb_hits': kb_hits,
        'model_info': model_info
    }

__all__ = ['synthesize_final_report']
