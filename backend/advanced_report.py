#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced Performance Report Generator

生成更高阶的性能优化建议报告，包含：
1. 分层瓶颈排序与类型归类 (Compute / Memory / Launch / Fusion)
2. 优化策略列表 (短期 / 中期 / 长期)
3. 任务清单 (T1/T2/... 含预计耗时与角色)
4. 预计收益与风险评估
5. 知识库写回候选片段 (可用于后续 ingestion)

若 Agent / 深度分析数据缺失，将生成占位骨架。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from backend.agent_core import AIAgent  # optional
except Exception:
    AIAgent = None  # type: ignore

DEFAULT_ANALYSIS_FILE = 'integrated_performance_report.md'

# ----------------- Helpers -----------------

def _load_basic_report(dir_path: Path) -> Optional[str]:
    fp = dir_path / DEFAULT_ANALYSIS_FILE
    if not fp.exists():
        return None
    try:
        return fp.read_text(encoding='utf-8')
    except Exception:
        return None

def _extract_hot_kernels(report_text: str) -> List[Dict[str, Any]]:
    hot_list: List[Dict[str, Any]] = []
    lines = report_text.splitlines()
    in_hot = False
    for line in lines:
        if '识别的热点Kernels' in line:
            in_hot = True
            continue
        if in_hot:
            if line.startswith('##'):
                break
            if line.strip().startswith(tuple(str(i)+'.' for i in range(1,10))):
                # example: '1. **kernel_name**...'
                try:
                    idx_dot = line.index('.')
                except ValueError:
                    continue
                rank_part = line[:idx_dot].strip()
                rest = line[idx_dot+1:].strip()
                name = ''
                if '**' in rest:
                    # between first pair of **
                    parts = rest.split('**')
                    if len(parts) >= 3:
                        name = parts[1]
                hot_list.append({'rank': rank_part, 'name': name, 'raw': rest})
    return hot_list

def _classify_bottlenecks(hot_kernels: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    classes = {'compute': [], 'memory': [], 'launch': [], 'fusion': []}
    for k in hot_kernels:
        name = k['name'].lower()
        if any(x in name for x in ['gemm','matmul','mm','cublas']):
            classes['compute'].append(k['name'])
        elif any(x in name for x in ['mem','ld','st','copy']):
            classes['memory'].append(k['name'])
        elif any(x in name for x in ['cudart','runtime','cudaLaunch']):
            classes['launch'].append(k['name'])
        else:
            # default to compute or fusion candidate
            classes['fusion'].append(k['name'])
    return classes

def _generate_tasks(classes: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    tasks: Dict[str, List[Dict[str, Any]]] = {'high': [], 'medium': [], 'low': []}
    # High priority examples
    if classes['compute']:
        tasks['high'].append({
            'id': 'T1',
            'title': '替换/优化主 MatMul/GEMM 内核',
            'estimate_days': '1-2',
            'kernels': classes['compute'][:5],
            'action': '基于 CUTLASS/cublasGemmEx 做基准，选择最优配置或自定义 kernel'
        })
    if classes['launch']:
        tasks['high'].append({
            'id': 'T2',
            'title': '启用 CUDA Graph 捕获 decode 执行路径',
            'estimate_days': '0.5-1',
            'kernels': classes['launch'][:5],
            'action': '减少小核 launch 开销与 host-side 调度空洞'
        })
    # Medium priority examples
    if classes['memory']:
        tasks['medium'].append({
            'id': 'T3',
            'title': '优化内存访问模式 / FlashAttention',
            'estimate_days': '2-3',
            'kernels': classes['memory'][:5],
            'action': 'KV 布局优化 + 启用 FlashAttention 分块减少 DRAM 带宽压力'
        })
    tasks['medium'].append({
        'id': 'T4',
        'title': '实现 LayerNorm+激活 Fusion',
        'estimate_days': '2-3',
        'kernels': [],
        'action': 'Triton/nvFuser 融合降低 launch & memory 开销'
    })
    # Low priority examples
    tasks['low'].append({
        'id': 'T5',
        'title': '建立验证与知识库写回流水线',
        'estimate_days': '1-2',
        'kernels': [],
        'action': 'A/B 验证 (latency, throughput, perplexity) 自动写回向量库'
    })
    return tasks

def _expected_gains(tasks: Dict[str, List[Dict[str, Any]]]) -> str:
    return (
        '若高/中优先级任务全部落地：吞吐提升预估 1.3×–2.0×， token 延迟降低 20%–40% (取决于算子融合与图捕获效果)。'
    )

def _generate_granular_kernel_tasks(hot_kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = []
    for idx, k in enumerate(hot_kernels[:20], 1):
        cls = k.get('classification','other')
        base_id = f"KT{idx}"
        if cls == 'compute':
            action = '验证 Tensor Core 使用率，运行 CUTLASS 基准，尝试 autotune GEMM 配置'
        elif cls == 'memory':
            action = '检查全局内存访问步幅与对齐，评估是否可用共享内存 / FlashAttention 分块'
        elif cls == 'launch':
            action = '整合到 CUDA Graph 或合并小核，减少 host-side 间隙'
        else:
            action = '尝试算子融合 (LayerNorm+激活) 或剖析指令级瓶颈'
        tasks.append({
            'id': base_id,
            'kernel': k.get('name'),
            'classification': cls,
            'time_pct': round(k.get('time_pct',0),2),
            'avg_time_ms': round(k.get('avg_time_ms',0),4),
            'action': action
        })
    return tasks

def generate_advanced_report(output_dir: Path, detailed: bool = False) -> str:
    report_path = output_dir / 'advanced_performance_report.md'
    base_text = _load_basic_report(output_dir)
    hot_kernels = _extract_hot_kernels(base_text) if base_text else []
    classes = _classify_bottlenecks(hot_kernels) if hot_kernels else {'compute': [], 'memory': [], 'launch': [], 'fusion': []}
    tasks = _generate_tasks(classes)
    gains = _expected_gains(tasks)
    metrics_block = ''
    granular_tasks: List[Dict[str, Any]] = []
    if detailed:
        from backend.perf_data_parser import load_comprehensive, aggregate_metrics
        comp = load_comprehensive(output_dir / 'comprehensive_analysis.json')
        if comp:
            agg = aggregate_metrics(comp)
            idle_pct = f"{agg['idle_fraction']*100:.1f}%" if agg.get('idle_fraction') is not None else 'N/A'
            bw = agg.get('bandwidth', {})
            metrics_block = ("## 0. 关键指标快照\n\n" +
                f"- Idle Fraction (估算): {idle_pct}\n" +
                f"- 平均带宽 (GB/s): {bw.get('avg_bandwidth_gb_s','N/A')}\n" +
                f"- 总数据传输 (MB): {bw.get('total_data_mb','N/A')}\n" +
                f"- 热点 kernel 数: {len(agg.get('hot_kernels', []))}\n\n")
            # 生成每 kernel 任务
            granular_tasks = _generate_granular_kernel_tasks(agg.get('hot_kernels', []))
        else:
            metrics_block = '## 0. 关键指标快照\n\n- (未找到 comprehensive_analysis.json)\n\n'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 🧠 高阶性能优化报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        if metrics_block:
            f.write(metrics_block)
        if not base_text:
            f.write('> ⚠️ 基础分析报告缺失，仅生成骨架。请先运行集成分析以获得热点 kernel 数据。\n\n')
        f.write('## 1. 热点 Kernel 分类\n\n')
        for cat, items in classes.items():
            f.write(f'- {cat}: {items if items else "(无)"}\n')
        f.write('\n')
        f.write('## 2. 优化策略概览\n\n')
        f.write('- Compute: 使用 CUTLASS / Tensor Core 配置，减少非最优 GEMM\n')
        f.write('- Memory: FlashAttention / KV 重布局 / 减少不必要的全量访问\n')
        f.write('- Launch: CUDA Graph 捕获减少 host-side idle 与 launch overhead\n')
        f.write('- Fusion: Triton/nvFuser 进行算子融合减少中间写回\n\n')
        f.write('## 3. 任务列表（按优先级）\n\n')
        for prio in ['high','medium','low']:
            f.write(f'### {prio.upper()}\n')
            for t in tasks[prio]:
                f.write(f"- {t['id']}: {t['title']} (预计 {t['estimate_days']} 天)\n  核心: {t['action']}\n  涉及 Kernels: {t['kernels'] if t['kernels'] else '(通用)'}\n")
            f.write('\n')
        f.write('## 4. 验证计划\n\n')
        f.write('- A/B: 对比优化前后 throughput / 单 token latency / perplexity\n')
        f.write('- Profiling: 使用 nsys + ncu 验证热点是否重新排序\n')
        f.write('- KB 回写: 自动摄取报告结论与指标到向量库\n\n')
        if granular_tasks:
            f.write('## 4.1 细粒度 Kernel 任务 (Granular)\n\n')
            for gt in granular_tasks:
                f.write(f"- {gt['id']}: {gt['kernel']} [{gt['classification']}] 占比 {gt['time_pct']}% 平均 {gt['avg_time_ms']} ms\n  行动: {gt['action']}\n")
            f.write('\n')
        f.write('## 5. 预计收益与风险\n\n')
        f.write(gains + '\n')
        f.write('- 风险: 可能精度下降 / 需要额外显存 / 开发时间不确定\n\n')
        f.write('## 6. 总结 (Summary)\n\n')
        f.write('当前瓶颈排序 (估计): Compute MatMul/GEMM > Memory-bound Attention > Kernel launch/fusion gaps.\n')
        f.write('执行顺序建议: MatMul 内核 / 精度调整 → CUDA Graph & Fusion → Attention KV 优化 → 全面算子融合与自动调参与 KB 写回。\n')
    return str(report_path)

__all__ = ['generate_advanced_report']
