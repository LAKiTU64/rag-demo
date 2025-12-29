#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多节点性能分析结果合并工具

用途:
  当你在多个机器 / GPU 节点上分别运行了 nsys_to_ncu_analyzer.py (或 SGlang 集成分析工作流)
  会在各自的输出目录生成:
    - comprehensive_analysis.json  综合汇总 (不含热点 kernel 详细列表)
    - hot_kernels.json            热点 kernel 列表 (step2 产物)
    - integrated_performance_report.md  单节点 Markdown 报告

本脚本接受多个分析输出目录, 合并为统一的:
    - multi_node_comprehensive_analysis.json
    - multi_node_integrated_report.md
并可选择写入知识库 (FAISS 或 TF-IDF fallback) 用于后续 RAG 检索。

合并策略(简化启发式):
 1. kernel_analysis: 逐字段求和 (时间、数量), 重新计算平均值; unique_kernels 合并去重求数;
 2. memory_analysis: total_transfers/total_data_mb/avg_bandwidth 以加权 (数据量) 或求平均;
 3. timeline_analysis: 执行跨度取 min(first) 与 max(last) 重新计算;
 4. 热点 kernels: 合并所有 hot_kernels.json, 同名 kernel 的 total_time_ms/count 累加, avg_time_ms 重新计算, 保留 max_time_ms 最大值;
 5. ncu_detailed_analysis: 按键合并, 数值字段求平均, 瓶颈列表去重;
 6. focus_analysis: 同上;

CLI:
  python multi_node_aggregator.py --output merged_report_dir dir1 dir2 dir3 --ingest --kb-path knowledge_store 

环境变量:
  DEFAULT_KB_PATH   覆盖 --kb-path 默认值

"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

try:
    from backend.knowledge_bases.kb_ingest import ingest_json_to_faiss
except Exception:
    ingest_json_to_faiss = None  # type: ignore

def _safe_load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def _merge_kernel_analysis(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    total_kernels = sum(i.get('total_kernels', 0) for i in items)
    total_kernel_time = sum(i.get('total_kernel_time', 0.0) for i in items)
    # unique_kernels 合并: 需要每个条目可能没有列表, 保留最大/或求和 (粗略)
    unique_sets = []
    for i in items:
        # 无法直接拿列表, comprehensive_analysis 里没有实际名称集合; 用 unique_kernels 字段近似估计
        count = i.get('unique_kernels')
        if isinstance(count, int):
            unique_sets.append(count)
    # 选择最大作为整体 unique 近似 (避免重复加总过大)
    merged_unique = max(unique_sets) if unique_sets else 0
    avg_kernel_time = (total_kernel_time / total_kernels) if total_kernels else 0.0
    # top_kernels/kernel_distribution 字段在不同节点意义混合, 此处不拼接原长字符串, 仅保留第一个
    first = items[0]
    return {
        'total_kernels': total_kernels,
        'unique_kernels': merged_unique,
        'total_kernel_time': total_kernel_time,
        'avg_kernel_time': avg_kernel_time,
    }

def _merge_memory(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    total_transfers = sum(i.get('total_transfers', 0) for i in items)
    total_data_mb = sum(i.get('total_data_mb', 0.0) for i in items)
    # 平均带宽: 简单加权平均 (按 data_mb)
    bw_parts = []
    for i in items:
        bw = i.get('avg_bandwidth'); data = i.get('total_data_mb', 0.0)
        if isinstance(bw, (int, float)) and data > 0:
            bw_parts.append((bw, data))
    weighted_bw = sum(bw*d for bw, d in bw_parts)/sum(d for _, d in bw_parts) if bw_parts else 0.0
    return {
        'total_transfers': total_transfers,
        'total_data_mb': total_data_mb,
        'avg_bandwidth': weighted_bw,
    }

def _merge_timeline(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    first_event = min((i.get('first_event_time') for i in items if isinstance(i.get('first_event_time'), (int, float))), default=None)
    last_event = max((i.get('last_event_time') for i in items if isinstance(i.get('last_event_time'), (int, float))), default=None)
    total_events = sum(i.get('total_events', 0) for i in items)
    span = (last_event - first_event) if (first_event is not None and last_event is not None) else None
    return {
        'total_events': total_events,
        'execution_span': span,
        'first_event_time': first_event,
        'last_event_time': last_event,
    }

def _merge_hot_kernels(lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for lst in lists:
        for k in lst:
            name = str(k.get('name'))
            if name not in merged:
                merged[name] = {
                    'name': name,
                    'total_time_ms': k.get('total_time_ms', 0.0),
                    'count': k.get('count', 0),
                    'max_time_ms': k.get('max_time_ms', k.get('avg_time_ms', 0.0)),
                }
            else:
                merged[name]['total_time_ms'] += k.get('total_time_ms', 0.0)
                merged[name]['count'] += k.get('count', 0)
                merged[name]['max_time_ms'] = max(merged[name]['max_time_ms'], k.get('max_time_ms', k.get('avg_time_ms', 0.0)))
    # 计算 avg_time_ms
    for v in merged.values():
        v['avg_time_ms'] = v['total_time_ms'] / v['count'] if v['count'] else 0.0
    # 排序按 total_time_ms
    return sorted(merged.values(), key=lambda x: x['total_time_ms'], reverse=True)

def _merge_ncu_detailed(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # items 是多个 comprehensive_results['ncu_detailed_analysis'] dict
    merged: Dict[str, Dict[str, Any]] = {}
    for detail in items:
        for kname, data in detail.items():
            tgt = merged.setdefault(kname, {
                'kernels_analyzed': 0,
                'bottlenecks_found': 0,
                'gpu_utilization': {},
                'memory_analysis': {},
                'bottleneck_summary': []
            })
            tgt['kernels_analyzed'] += data.get('kernels_analyzed', 0)
            tgt['bottlenecks_found'] += data.get('bottlenecks_found', 0)
            # 合并 gpu_utilization 数值平均
            gu = data.get('gpu_utilization', {})
            for key, val in gu.items():
                if isinstance(val, (int, float)):
                    lst = tgt['gpu_utilization'].setdefault(key, [])
                    lst.append(val)
            mem = data.get('memory_analysis', {})
            for key, val in mem.items():
                if isinstance(val, dict):
                    # 只支持一层 metrics dict flatten
                    for mkey, mval in val.items():
                        if isinstance(mval, (int, float)):
                            lst = tgt['memory_analysis'].setdefault(mkey, [])
                            lst.append(mval)
            # 瓶颈去重按 description
            for b in data.get('bottleneck_summary', []):
                if b not in tgt['bottleneck_summary']:
                    tgt['bottleneck_summary'].append(b)
    # 平均化数值列表
    for kname, data in merged.items():
        for key, lst in list(data['gpu_utilization'].items()):
            if isinstance(lst, list) and lst:
                data['gpu_utilization'][key] = sum(lst)/len(lst)
        for key, lst in list(data['memory_analysis'].items()):
            if isinstance(lst, list) and lst:
                data['memory_analysis'][key] = sum(lst)/len(lst)
    return merged

def _merge_focus(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # focus metrics structure similar to detailed but simpler
    merged: Dict[str, Dict[str, Any]] = {}
    for focus in items:
        for kname, data in focus.items():
            tgt = merged.setdefault(kname, {
                'kernels_analyzed': 0,
                'gpu_utilization': {},
                'memory_analysis': {},
                'bottleneck_summary': []
            })
            tgt['kernels_analyzed'] += data.get('kernels_analyzed', 0)
            gu = data.get('gpu_utilization', {})
            for key, val in gu.items():
                if isinstance(val, (int, float)):
                    lst = tgt['gpu_utilization'].setdefault(key, [])
                    lst.append(val)
            mem_bw = data.get('memory_analysis', {}).get('bandwidth_stats', {})
            for key, val in mem_bw.items():
                if isinstance(val, (int, float)):
                    lst = tgt['memory_analysis'].setdefault(key, [])
                    lst.append(val)
            for b in data.get('bottleneck_summary', []):
                if b not in tgt['bottleneck_summary']:
                    tgt['bottleneck_summary'].append(b)
    # average
    for kname, data in merged.items():
        for key, lst in list(data['gpu_utilization'].items()):
            data['gpu_utilization'][key] = sum(lst)/len(lst) if lst else None
        for key, lst in list(data['memory_analysis'].items()):
            data['memory_analysis'][key] = sum(lst)/len(lst) if lst else None
    return merged

def merge_analysis_dirs(dirs: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    comps = []
    hot_lists = []
    for d in dirs:
        p = Path(d)
        comp = _safe_load_json(p / 'comprehensive_analysis.json')
        if comp:
            comps.append(comp)
        hot = _safe_load_json(p / 'hot_kernels.json')
        if isinstance(hot, list):
            hot_lists.append(hot)
    if not comps:
        raise RuntimeError('未找到任何 comprehensive_analysis.json 文件')
    # Merge overview sections
    kernel_items = [c.get('nsys_overview', {}).get('kernel_analysis', {}) for c in comps]
    mem_items = [c.get('nsys_overview', {}).get('memory_analysis', {}) for c in comps]
    time_items = [c.get('nsys_overview', {}).get('timeline_analysis', {}) for c in comps]
    merged_hot = _merge_hot_kernels(hot_lists)
    merged_detail = _merge_ncu_detailed([c.get('ncu_detailed_analysis', {}) for c in comps])
    merged_focus = _merge_focus([c.get('ncu_focus_analysis', {}) for c in comps])

    merged = {
        'timestamp': datetime.utcnow().isoformat(),
        'nodes_count': len(comps),
        'source_dirs': dirs,
        'nsys_overview': {
            'kernel_analysis': _merge_kernel_analysis(kernel_items),
            'memory_analysis': _merge_memory(mem_items),
            'timeline_analysis': _merge_timeline(time_items),
        },
        'hot_kernels_count': len(merged_hot),
        'hot_kernels_merged': merged_hot[:50],  # limit for readability
        'ncu_detailed_analysis': merged_detail,
        'ncu_focus_analysis': merged_focus
    }
    return merged, merged_hot

def write_reports(merged: Dict[str, Any], all_hot: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'multi_node_comprehensive_analysis.json'
    json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path = out_dir / 'multi_node_integrated_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 多节点集成性能分析报告\n\n')
        f.write(f'- 生成时间: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'- 节点数量: {merged.get("nodes_count")}\n')
        f.write('## 汇总 Kernel 概览\n')
        ka = merged['nsys_overview']['kernel_analysis']
        f.write(f'- 总 kernels 数: {ka.get("total_kernels",0)}\n')
        f.write(f'- 总执行时间(ms): {ka.get("total_kernel_time",0):.2f}\n')
        f.write(f'- 平均单 kernel 时间(ms): {ka.get("avg_kernel_time",0):.4f}\n')
        f.write('## 合并热点 Kernels (前20)\n')
        for hk in all_hot[:20]:
            f.write(f'- {hk["name"]}: total={hk["total_time_ms"]:.2f}ms avg={hk["avg_time_ms"]:.3f}ms count={hk["count"]}\n')
        f.write('\n## 焦点 NCU 聚合 (若存在)\n')
        for kname, data in (merged.get('ncu_focus_analysis', {}) or {}).items():
            gu = data.get('gpu_utilization', {})
            mem = data.get('memory_analysis', {})
            f.write(f'### {kname}\n')
            if gu:
                f.write(f'- 平均SM效率: {gu.get("average_sm_efficiency","N/A")}\n')
                f.write(f'- Occupancy: {gu.get("achieved_occupancy","N/A")}\n')
            if mem:
                f.write(f'- 平均带宽: {mem.get("average_bandwidth","N/A")}\n')
            bsum = data.get('bottleneck_summary', [])
            if bsum:
                f.write('- 瓶颈: ' + ', '.join(b.get('description','') for b in bsum) + '\n')
        f.write('\n## 优化建议概要\n')
        f.write('- 优先关注累计时间最高的前10个合并热点 kernel\n')
        f.write('- 对 SM 效率偏低的焦点内核进行算子融合或访存优化\n')
        f.write('- 针对平均带宽低且占用时间长的内核分析访存模式 (例如 coalescing / cache 利用)\n')
    return md_path

def main(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description='多节点综合分析结果合并工具')
    ap.add_argument('dirs', nargs='+', help='各节点分析输出目录 (包含 comprehensive_analysis.json)')
    ap.add_argument('--output', default='merged_multi_node', help='合并结果输出目录')
    ap.add_argument('--ingest', action='store_true', help='将合并 JSON 写入知识库向量索引')
    ap.add_argument('--kb-path', default=os.getenv('DEFAULT_KB_PATH','knowledge_store'), help='知识库目录')
    args = ap.parse_args(argv)
    merged, all_hot = merge_analysis_dirs(args.dirs)
    out_dir = Path(args.output)
    md_path = write_reports(merged, all_hot, out_dir)
    print(f'✅ 合并报告生成: {md_path}')
    if args.ingest:
        if ingest_json_to_faiss is None:
            print('⚠️ 未加载知识库摄取模块，跳过 ingest')
        else:
            try:
                ingest_json_to_faiss(json.dumps(merged, ensure_ascii=False), index_dir=Path(args.kb_path))
                print(f'📥 已摄取合并报告 JSON 到知识库: {args.kb_path}')
            except Exception as e:
                print(f'⚠️ 摄取失败: {e}')

if __name__ == '__main__':
    main(sys.argv[1:])
