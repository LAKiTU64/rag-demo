#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance Analysis LangChain Tools

封装一体化分析 (nsys + ncu + 报告 + 回写知识库) 作为可调用工具。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess

try:
    from langchain.tools import BaseTool
except Exception:
    BaseTool = object  # 允许无 langchain 环境时导入

class RunIntegratedPerfAnalysisTool(BaseTool):
    name = "run_integrated_perf_analysis"
    description = (
        "运行一体化性能分析：给定模型路径及参数，执行 nsys -> 提取热点 -> ncu -> 生成增强报告并回写知识库。"
        "输入JSON: {model_path:str,batch_size:int,input_len:int,output_len:int,max_kernels?:int}"
    )

    def _run(self, query: str) -> str:  # 返回 JSON 字符串
        try:
            params = json.loads(query)
        except Exception:
            return json.dumps({"status": "error", "message": "输入不是有效JSON"}, ensure_ascii=False)

        model_path = params.get('model_path')
    batch_size = int(params.get('batch_size', 8))
    # 默认输入输出长度升级为 2048 / 1024
    input_len = int(params.get('input_len', 2048))
    output_len = int(params.get('output_len', 1024))
        max_kernels = int(params.get('max_kernels', 3))

        if not model_path or not Path(model_path).exists():
            return json.dumps({"status": "error", "message": f"模型路径不存在: {model_path}"}, ensure_ascii=False)

        # 调用现有 Python API (直接导入 NSysToNCUAnalyzer)
        try:
            from backend.utils.nsys_to_ncu_analyzer import NSysToNCUAnalyzer
            from backend.report_generator import generate_enriched_report
            from backend.knowledge_bases.kb_ingest import ingest_plain_text_to_faiss
        except Exception as e:
            return json.dumps({"status": "error", "message": f"导入分析模块失败: {e}"}, ensure_ascii=False)

        analyzer = NSysToNCUAnalyzer(
            f"auto_analysis_b{batch_size}_i{input_len}_o{output_len}"
        )
        cmd = [
            'python', '-m', 'sglang.bench_one_batch',
            '--model-path', model_path,
            '--batch-size', str(batch_size),
            '--input-len', str(input_len),
            '--output-len', str(output_len),
            '--load-format', 'dummy'
        ]
        try:
            # 先执行全量 NCU 采集 (不做过滤) 使用 compute 集合
            full_capture_file = analyzer.full_ncu_capture(cmd, profile_name="ncu_full_capture", set_name="compute", launch_limit=None)
            nsys_file = analyzer.step1_nsys_analysis(cmd, "auto_overview")
            hot = analyzer.step2_extract_hot_kernels(nsys_file, top_k=8)
            ncu_files = []
            if hot:
                ncu_files = analyzer.step3_ncu_targeted_analysis(cmd, hot, max_kernels=max_kernels)
            comprehensive = analyzer.step4_comprehensive_analysis(ncu_files)
            base_report = analyzer.generate_final_report(comprehensive)
            enriched_path = generate_enriched_report(analyzer.output_dir, comprehensive)
            # 回写报告向量
            try:
                report_text = Path(enriched_path).read_text(encoding='utf-8')
                ingest_plain_text_to_faiss(report_text)
            except Exception:
                pass
            return json.dumps({
                "status": "ok",
                "output_dir": str(analyzer.output_dir),
                "hot_kernels": hot[:10],
                "full_ncu_rep": full_capture_file,
                "ncu_files": ncu_files,
                "report_basic": base_report,
                "report_enriched": enriched_path
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"分析失败: {e}"}, ensure_ascii=False)

    async def _arun(self, query: str) -> str:  # 可异步复用同步逻辑
        return self._run(query)

__all__ = ['RunIntegratedPerfAnalysisTool']