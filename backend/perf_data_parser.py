#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance Data Parser

解析 nsys 与 ncu 整合后的 comprehensive_analysis.json 以及可能的 NCU CSV/rep 文件，抽取：
- 全局 kernel 时间与 idle_fraction (估算)
- 每个热点 kernel 的时间占比与分类标签
- 内存带宽 / 传输统计
- 可扩展的 SM 效率 (如果 ncu 分析提供)

Idle 估算: 使用 timeline execution_span 与总 kernel 时间差值 / execution_span。
若 execution_span 缺失或异常返回 None.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import csv

def load_comprehensive(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def compute_idle_fraction(data: Dict[str, Any]) -> Optional[float]:
    nsys = data.get('nsys_overview', {})
    kernel = nsys.get('kernel_analysis', {})
    timeline = nsys.get('timeline_analysis', {})
    total_kernel_time = kernel.get('total_kernel_time')  # ms
    execution_span = timeline.get('execution_span')      # seconds?
    if total_kernel_time is None or execution_span is None:
        return None
    # normalize units: execution_span may be seconds (from sample). Convert to ms.
    span_ms = execution_span * 1000.0
    if span_ms <= 0:
        return None
    idle = max(span_ms - total_kernel_time, 0.0)
    return idle / span_ms

def parse_hot_kernels(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    hot = []
    hk = data.get('hot_kernels', [])
    total_kernel_time = data.get('nsys_overview', {}).get('kernel_analysis', {}).get('total_kernel_time', 0.0)
    for k in hk:
        t = k.get('total_time_ms', 0.0)
        pct = (t / total_kernel_time * 100.0) if total_kernel_time else 0.0
        hot.append({
            'name': k.get('name',''),
            'total_time_ms': t,
            'avg_time_ms': k.get('avg_time_ms', 0.0),
            'count': k.get('count', 0),
            'time_pct': pct,
            'classification': classify_kernel(k.get('name',''))
        })
    return hot

def classify_kernel(name: str) -> str:
    lower = name.lower()
    if any(x in lower for x in ['gemm','matmul','mm','cublas']):
        return 'compute'
    if any(x in lower for x in ['mem','ld','st','copy']):
        return 'memory'
    if any(x in lower for x in ['cudart','runtime','launch']):
        return 'launch'
    return 'other'

def parse_bandwidth_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    mem = data.get('nsys_overview', {}).get('memory_analysis', {})
    return {
        'total_transfers': mem.get('total_transfers'),
        'total_data_mb': mem.get('total_data_mb'),
        'avg_bandwidth_gb_s': mem.get('avg_bandwidth'),
    }

def aggregate_metrics(comprehensive: Dict[str, Any]) -> Dict[str, Any]:
    idle_fraction = compute_idle_fraction(comprehensive)
    hot = parse_hot_kernels(comprehensive)
    bandwidth = parse_bandwidth_stats(comprehensive)
    return {
        'idle_fraction': idle_fraction,
        'hot_kernels': hot,
        'bandwidth': bandwidth
    }

# Optional future: parse NCU CSV for SM efficiency (currently empty in sample)

def parse_ncu_csv(csv_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return None
    return rows

__all__ = [
    'load_comprehensive', 'aggregate_metrics', 'parse_ncu_csv'
]
