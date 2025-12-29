#!/usr/bin/env python3
"""
merge_profiles.py

扫描指定目录 (默认 /workspace/Agent/AI_Agent_Complete) 下的性能分析产物:
- nsys: *.nsys-rep, *_kernels_*.csv, *.json (由 nsys_parser 导出的 json)
- ncu: *.ncu-rep, *.csv (NCU 导出), *_ncu_summary.json

生成统一索引:
1. index.json : 汇总每类文件的元数据 (尺寸/修改时间/类型/推断kernel数量等)
2. summary.md : 人类可读概览 (分类、数量、示例前若干文件)
3. kernels_catalog.txt : 聚合所有可解析出的 kernel 名称(去重)用于前端快速显示

使用:
    python merge_profiles.py --scan-dir /workspace/Agent/AI_Agent_Complete --max-kernels 500

可扩展点:
- 增加对报告内部指标的统计 (平均 SM 效率等) 需要解析 CSV/JSON 内容，这里做轻量实现。
"""
from __future__ import annotations
import argparse
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

PROFILE_EXT_GROUPS = {
    'nsys_rep': ['.nsys-rep'],
    'ncu_rep': ['.ncu-rep'],
    'csv': ['.csv'],
    'json': ['.json'],
}

KERNEL_NAME_PAT = re.compile(r'(Kernel|cuda|cublas|cutlass|flash|triton|gemm|matmul|aten)', re.IGNORECASE)

def guess_kind(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == '.nsys-rep':
        return 'nsys_rep'
    if suf == '.ncu-rep':
        return 'ncu_rep'
    if suf == '.csv':
        # 粗略区分 ncu csv vs nsys csv
        name = path.name.lower()
        if 'kern' in name and 'cuda' in name:
            return 'nsys_csv'
        if 'ncu' in name or 'full_capture' in name:
            return 'ncu_csv'
        return 'csv'
    if suf == '.json':
        name = path.name.lower()
        if name.endswith('_ncu_summary.json'):
            return 'ncu_summary_json'
        return 'json'
    return 'other'

def collect_kernel_names_from_csv(path: Path, limit: int = 300) -> List[str]:
    names: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row:
                    continue
                if header is None:
                    header = [c.strip() for c in row]
                    continue
                # 查找 Kernel 名列
                def find_idx(possible):
                    if not header:
                        return None
                    lower_map = {h.lower(): i for i, h in enumerate(header)}
                    for c in possible:
                        if c.lower() in lower_map:
                            return lower_map[c.lower()]
                    return None
                name_idx = find_idx(['Kernel Name', 'Kernel', 'Name'])
                if name_idx is not None and name_idx < len(row):
                    val = row[name_idx].strip()
                    if val and not val.isdigit() and KERNEL_NAME_PAT.search(val):
                        names.append(val)
                if len(names) >= limit:
                    break
    except Exception:
        return []
    # 去重保持顺序
    seen = set(); uniq = []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq

def collect_kernel_names_from_json(path: Path, limit: int = 300) -> List[str]:
    try:
        text = path.read_text(encoding='utf-8')
        data = json.loads(text)
    except Exception:
        return []
    found: List[str] = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = k.lower()
                if isinstance(v, (str, int)) and any(t in kl for t in ['name','mangled','demangle']):
                    sv = str(v).strip()
                    if len(sv) > 3 and not sv.isdigit() and KERNEL_NAME_PAT.search(sv):
                        found.append(sv)
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)
    walk(data)
    # 去重
    seen = set(); uniq = []
    for n in found:
        if n not in seen:
            seen.add(n); uniq.append(n)
        if len(uniq) >= limit:
            break
    return uniq

def build_index(scan_dir: Path, max_kernels: int) -> Dict:
    files = [p for p in scan_dir.rglob('*') if p.is_file()]
    index: Dict[str, List[Dict]] = {}
    all_kernel_names: List[str] = []
    for f in files:
        kind = guess_kind(f)
        if kind == 'other':
            continue
        meta = {
            'path': str(f),
            'size': f.stat().st_size,
            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            'kind': kind
        }
        index.setdefault(kind, []).append(meta)
        # 尝试提取 kernel 名
        extracted: List[str] = []
        if kind in ('nsys_csv','ncu_csv','csv'):
            extracted = collect_kernel_names_from_csv(f, limit=50)
        elif kind in ('json','ncu_summary_json'):
            extracted = collect_kernel_names_from_json(f, limit=50)
        if extracted:
            all_kernel_names.extend(extracted)
    # 去重 kernel 名称
    seen: Set[str] = set(); uniq_kernel = []
    for k in all_kernel_names:
        if k not in seen:
            seen.add(k); uniq_kernel.append(k)
        if len(uniq_kernel) >= max_kernels:
            break
    return {
        'generated_at': datetime.now().isoformat(),
        'scan_dir': str(scan_dir),
        'groups': index,
        'unique_kernel_names_count': len(uniq_kernel),
        'unique_kernel_names': uniq_kernel[:max_kernels]
    }

def write_outputs(scan_dir: Path, index: Dict) -> None:
    index_path = scan_dir / 'index.json'
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8')

    # kernels_catalog.txt
    catalog_path = scan_dir / 'kernels_catalog.txt'
    with open(catalog_path, 'w', encoding='utf-8') as f:
        for k in index.get('unique_kernel_names', []):
            f.write(k + '\n')

    # summary.md
    summary_path = scan_dir / 'summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# 性能分析文件汇总\n\n")
        f.write(f"生成时间: {index['generated_at']}\n\n")
        for group, items in index['groups'].items():
            f.write(f"## {group} (共 {len(items)} 个)\n\n")
            for it in items[:10]:
                f.write(f"- {it['path']} (size={it['size']})\n")
            f.write('\n')
        f.write(f"## 去重后 Kernel 名称 ({index['unique_kernel_names_count']} 个, 显示最多 {len(index['unique_kernel_names'])})\n\n")
        for k in index['unique_kernel_names'][:50]:
            f.write(f"- {k}\n")
    print(f"✅ 已生成: {index_path}, {catalog_path}, {summary_path}")

def main():
    ap = argparse.ArgumentParser(description='合并 nsys/ncu 分析产物生成统一索引')
    ap.add_argument('--scan-dir', default='/workspace/Agent/AI_Agent_Complete', help='扫描目录')
    ap.add_argument('--max-kernels', type=int, default=500, help='最大聚合 kernel 名称数量')
    args = ap.parse_args()
    scan_dir = Path(args.scan_dir)
    if not scan_dir.exists():
        raise SystemExit(f'目录不存在: {scan_dir}')
    index = build_index(scan_dir, args.max_kernels)
    write_outputs(scan_dir, index)

if __name__ == '__main__':
    main()
