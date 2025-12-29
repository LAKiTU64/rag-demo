#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 nsys 报告中提取热点 CUDA kernel 名称，供 ncu --kernel-name 使用。

使用方式:
    python extract_nsys_kernels.py --rep run_profile.nsys-rep --top-k 8 --min-avg-ms 0.05 --out kernels.txt

输出:
    kernels.txt (每行一个 kernel 名称，已按总耗时降序过滤)

后续直接使用:
    ncu --kernel-name "<第一行>" --kernel-name "<第二行>" ... <原始命令>

注意:
    - 如果名称仍为数字或 __unnamed_ 前缀，运行脚本时可以加 --include-placeholder 让其保留
    - H100 / Hopper 等架构下大型 GEMM kernel 名较长，可优先尝试 demangled 精确匹配；匹配失败再用 regex: 前缀
"""

import argparse
from pathlib import Path
from typing import List
from nsys_parser import NsysParser, NsysAnalyzer


def extract_kernels(rep: Path, top_k: int, min_avg_ms: float, include_placeholder: bool) -> List[str]:
    parser = NsysParser(str(rep))
    parser.parse()
    analyzer = NsysAnalyzer(parser)
    analyzer.analyze()
    names = analyzer.get_top_kernel_names(top_k=top_k, min_duration_ms=min_avg_ms)

    if not include_placeholder:
        def is_placeholder(n: str) -> bool:
            low = n.lower()
            return low.isdigit() or low.startswith('__unnamed_')
        names = [n for n in names if not is_placeholder(n)]
    return names


def main():
    ap = argparse.ArgumentParser(description="Extract top CUDA kernel names from nsys report")
    ap.add_argument('--rep', required=True, help='nsys .nsys-rep 文件路径')
    ap.add_argument('--top-k', type=int, default=10, help='返回的最大 kernel 数量')
    ap.add_argument('--min-avg-ms', type=float, default=0.0, help='过滤平均耗时低于该阈值的 kernel (ms)')
    ap.add_argument('--out', type=str, default='kernels.txt', help='输出文件路径 (每行一个 kernel 名)')
    ap.add_argument('--include-placeholder', action='store_true', help='是否保留数字或 __unnamed_ 内核名')
    args = ap.parse_args()

    rep_path = Path(args.rep)
    if not rep_path.exists():
        raise SystemExit(f"❌ nsys-rep 不存在: {rep_path}")

    names = extract_kernels(rep_path, args.top_k, args.min_avg_ms, args.include_placeholder)
    if not names:
        print("⚠️ 未提取到任何 kernel 名称")
    else:
        out_path = Path(args.out)
        out_path.write_text('\n'.join(names), encoding='utf-8')
        print(f"✅ 已写出 {len(names)} 个 kernel 名到: {out_path}")
        print("示例 ncu 调用 (可手动编辑补充):")
        if names:
            demo = ' '.join([f"--kernel-name '{n}'" for n in names[:min(len(names),4)]])
            print(f"ncu {demo} --set full -o ncu_hotkernels -- <原始命令>")

if __name__ == '__main__':
    main()
