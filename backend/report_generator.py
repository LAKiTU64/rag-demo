#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enriched Performance Report Generator

将现有的综合分析结果 (nsys + ncu) 与上传的理论知识库 (FAISS) 结合，生成更完整的 Markdown 报告。

功能:
1. 装载 comprehensive_analysis.json 或传入的 dict
2. 读取热点 kernel、SM 效率、内存带宽等指标
3. 针对每类瓶颈查询 FAISS 向量库获取相关理论支撑 (若可用)
4. 输出结构化的带解释报告

用法示例:
    from report_generator import generate_enriched_report
    md_path = generate_enriched_report(output_dir=Path('/workspace/Agent/AI_Agent_Complete/sglang_analysis_b8_i512_o64'))

触发检索关键词映射:
    - SM效率低 -> "SM utilization optimization"
    - 内存带宽低 -> "memory bandwidth optimization"
    - kernel执行时间长 -> "kernel latency reduction"
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 可选加载 FAISS 检索
try:
    from knowledge_bases.faiss_store import load_index, query
except Exception:
    load_index = None
    query = None

DEFAULT_FAISS_DIR = Path('/workspace/Agent/AI_Agent_Complete/faiss_index')

def _load_comprehensive_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / 'comprehensive_analysis.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def _faiss_available() -> bool:
    return load_index is not None and query is not None and DEFAULT_FAISS_DIR.exists()

def _retrieve_theory(frase: str, top_k: int = 3) -> List[str]:
    if not _faiss_available():
        return []
    try:
        store = load_index(DEFAULT_FAISS_DIR, model_name='sentence-transformers/all-MiniLM-L6-v2')
        results = query(store, frase, top_k=top_k)
        return [r['text'] for r in results]
    except Exception:
        return []

def _format_list_block(items: List[str]) -> str:
    return '\n'.join([f'- {i}' for i in items]) if items else '- (无相关理论检索结果)'

def _kernel_bottleneck_theory(analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """针对单个 kernel 分析结果构造理论检索。"""
    theory_map: Dict[str, List[str]] = {}
    # SM 效率
    gpu_util = analysis.get('gpu_utilization', {})
    sm_eff = gpu_util.get('average_sm_efficiency', 0)
    if sm_eff and sm_eff < 40:  # 低 SM 利用
        theory_map['SM效率偏低'] = _retrieve_theory('low SM occupancy reasons and optimization')
    # 内存带宽
    memory_analysis = analysis.get('memory_analysis', {})
    bw_stats = memory_analysis.get('bandwidth_stats', {})
    avg_bw = bw_stats.get('average_bandwidth', 0)
    if avg_bw and avg_bw < 200:  # 阈值可后续动态调整
        theory_map['内存带宽偏低'] = _retrieve_theory('improve memory bandwidth GPU kernel coalesced access')
    # 通用瓶颈描述
    bottlenecks = analysis.get('bottleneck_summary', [])
    for b in bottlenecks:
        desc = b.get('description','')
        if 'latency' in desc.lower():
            theory_map.setdefault('延迟优化', []).extend(_retrieve_theory('reduce kernel latency GPU optimization techniques'))
        if 'memory' in desc.lower():
            theory_map.setdefault('内存访问模式', []).extend(_retrieve_theory('optimize global memory access patterns'))
    # 去重
    for k, v in theory_map.items():
        seen = set(); uniq = []
        for t in v:
            if t not in seen:
                seen.add(t); uniq.append(t)
        theory_map[k] = uniq
    return theory_map

def generate_enriched_report(output_dir: Path, comprehensive: Optional[Dict[str, Any]] = None) -> str:
    """生成增强版报告.
    返回生成的 markdown 文件路径字符串
    """
    if comprehensive is None:
        comprehensive = _load_comprehensive_results(output_dir)
    if comprehensive is None:
        raise FileNotFoundError('未找到综合分析结果 JSON')

    report_path = output_dir / 'integrated_performance_report_enriched.md'

    nsys_overview = comprehensive.get('nsys_overview', {})
    hot_count = comprehensive.get('hot_kernels_count', 0)
    ncu_analysis = comprehensive.get('ncu_detailed_analysis', {})

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 📘 集成性能分析增强报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('## 1. 全局概览 (Nsys)\n\n')
        if 'kernel_analysis' in nsys_overview:
            ka = nsys_overview['kernel_analysis']
            f.write(f'- 总 kernels 数量: {ka.get("total_kernels",0)}\n')
            f.write(f'- 总 kernel 执行时间: {ka.get("total_kernel_time",0):.2f} ms\n')
            f.write(f'- 平均 kernel 执行时间: {ka.get("avg_kernel_time",0):.3f} ms\n')
        f.write('\n')
        f.write('## 2. 热点 Kernels 概览\n\n')
        f.write(f'识别的热点数量: {hot_count}\n\n')
        hot_list = comprehensive.get('hot_kernels', [])
        for i, k in enumerate(hot_list[:15], 1):
            f.write(f'{i}. {k.get("name","")[:100]} | 总时间 {k.get("total_time_ms",0):.2f} ms | 调用次数 {k.get("count",0)} | 平均 {k.get("avg_time_ms",0):.3f} ms\n')
        f.write('\n')
        f.write('## 3. 深度分析 (NCU) + 理论支撑\n\n')
        for kernel_name, analysis in ncu_analysis.items():
            f.write(f'### Kernel: {kernel_name}\n\n')
            gpu_util = analysis.get('gpu_utilization', {})
            if gpu_util:
                f.write(f'- 平均 SM 效率: {gpu_util.get("average_sm_efficiency",0):.1f}%\n')
            memory_analysis = analysis.get('memory_analysis', {})
            bw_stats = memory_analysis.get('bandwidth_stats', {})
            if bw_stats:
                f.write(f'- 平均内存带宽: {bw_stats.get("average_bandwidth",0):.1f} GB/s\n')
            bottlenecks = analysis.get('bottleneck_summary', [])
            if bottlenecks:
                f.write('- 初步瓶颈:')
                for b in bottlenecks:
                    f.write(f' {b.get("description","")}[{b.get("severity","")}] ;')
                f.write('\n')
            theory = _kernel_bottleneck_theory(analysis)
            if theory:
                f.write('\n#### 理论检索建议\n')
                for theme, texts in theory.items():
                    f.write(f'- {theme}:\n')
                    for t in texts[:5]:
                        f.write(f'  * {t[:180]}\n')
            f.write('\n')
        f.write('## 4. 交叉指标与潜在瓶颈分类\n\n')
        # 简单分类: 低SM / 低带宽 / 延迟类
        low_sm = [k for k,a in ncu_analysis.items() if a.get('gpu_utilization',{}).get('average_sm_efficiency',0) < 40]
        low_bw = []
        for k,a in ncu_analysis.items():
            bw = a.get('memory_analysis',{}).get('bandwidth_stats',{}).get('average_bandwidth',0)
            if bw and bw < 200:
                low_bw.append(k)
        f.write(f'- 低SM效率 Kernels: {low_sm if low_sm else "无"}\n')
        f.write(f'- 低内存带宽 Kernels: {low_bw if low_bw else "无"}\n')
        f.write('\n')
        f.write('## 5. 综合优化建议\n\n')
        f.write('- 针对低 SM 效率: 分析线程块维度、occupancy、是否存在 warp divergence\n')
        f.write('- 针对低内存带宽: 检查访问是否未对齐、是否可以使用共享内存、提高并发度\n')
        f.write('- 针对高延迟 kernel: 考虑算子融合、算法替换 (例如 FlashAttention) 或减少同步屏障\n')
        f.write('- 考虑利用理论上限数据对比实际指标，评估是否接近硬件瓶颈\n\n')
        f.write('## 6. 理论上限匹配总结\n\n')
        if _faiss_available():
            # 聚合一次整体检索
            global_theory = _retrieve_theory('GPU performance optimization theoretical limits memory bandwidth latency occupancy')
            f.write(_format_list_block(global_theory) + '\n')
        else:
            f.write('- (未加载理论知识库或索引缺失)\n')
        f.write('\n')
        f.write('## 7. 后续行动与数据采集建议\n\n')
        f.write('- 进行参数扫 (batch/input length) 观察热点 kernel 时间变化趋势\n')
        f.write('- 针对识别瓶颈补充 Nsight Compute section level 采集 (例如 memory_workload)\n')
        f.write('- 若部分 kernel 已接近理论内存带宽，可转向算子级别优化或模型结构更改\n')
        f.write('- 将报告写入知识库便于后续检索迭代\n')
    return str(report_path)

__all__ = ['generate_enriched_report']
