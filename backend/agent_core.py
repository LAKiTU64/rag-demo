#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent核心模块 - 集成NSys和NCU性能分析 + Agentic-RAG
"""

import re
import os
import asyncio
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple


# 导入分析工具
sys.path.insert(0, str(Path(__file__).parent))

from utils.nsys_to_ncu_analyzer import create_sglang_analysis_workflow
from offline_llm import get_offline_qwen_client
from knowledge_bases.vector_kb_manager import VectorKBManager

OFFLINE_QWEN_PATH = Path(os.getenv("QWEN_LOCAL_MODEL_PATH", "./.models/Qwen/Qwen3-4B"))


class AIAgent:
    """AI Agent核心类 - 自动化性能分析（支持 Agentic-RAG）"""

    def __init__(self, config: Dict):
        self.config = config

        # sglang 和模型路径
        self.sglang_path = Path(config.get("sglang_path"))
        self.models_path = Path(config.get("models_path"))
        self.model_mappings = config.get("model_mappings")

        # 输出目录
        self.results_dir = Path(config.get("output", {}).get("results_dir"))
        self.results_dir.mkdir(exist_ok=True)

        # 本地 LLM 客户端（用于 Agentic 决策）
        self.offline_qwen_path = Path(config.get("offline_qwen_path"))
        self.llm_client = get_offline_qwen_client(self.offline_qwen_path)

        # 分析工具配置
        self.profiling_config = config.get("profiling_tools")
        self.analysis_defaults = config.get("analysis_defaults")

        # 缓存
        self.last_analysis_dir: Optional[str] = None
        self.last_analysis_dirs: List[str] = []
        self.last_analysis_reports: List[str] = []
        self.last_analysis_table: Optional[str] = None

        # 向量知识库相关
        self.kb = VectorKBManager()
        kb_config = config.get("vector_store")
        self.persist_directory = kb_config.get("persist_directory")
        self.embedding_model = kb_config.get("embedding_model")
        self.chunk_size = kb_config.get("chunk_size")
        self.chunk_overlap = kb_config.get("chunk_overlap")
        self.default_search_k = kb_config.get("default_search_k")
        self.similarity_threshold = kb_config.get("similarity_threshold")

    async def process_message(self, message: str) -> str:
        """Agentic-RAG 主流程：检索 → 推理 → 执行或回答"""

        # Step 1: RAG 检索
        retrieved_contexts = self.kb.search(query=message, k=3)
        rag_context = ""
        if retrieved_contexts:
            rag_snippets = [
                f"【{res['doc_id']}】{res['content'][:300]}"
                for res in retrieved_contexts
            ]
            rag_context = "\n".join(rag_snippets)

        # Step 2: 构造决策 Prompt
        rag_prompt = f"""你是一个高性能计算（HPC）与大模型性能分析专家。
请根据以下信息判断用户意图，并决定是否需要启动 NSys/NCU 性能分析流程。

### 用户原始请求
{message}

### 相关知识库片段（如有）
{rag_context if rag_context else "无相关历史文档"}

### 你的任务
1. 如果知识库已包含足够答案（例如常见问题、已知瓶颈、优化建议），请直接回答。
2. 如果请求涉及对具体模型的性能分析（如“分析 qwen-7b”、“测试 batch_size=4”），则必须返回严格 JSON 格式：
   {{"action": "run_analysis", "model": "模型名", "analysis_type": "类型", "params": {{"batch_size": [...], "input_len": [...], "output_len": [...]}}}}
3. 如果信息不足或模型未知，请返回：
   {{"action": "clarify", "message": "请说明..."}}

只输出 JSON 或直接回答，不要解释。"""

        # Step 3: 调用 LLM 决策
        try:
            decision_output = self.llm_client.generate(rag_prompt, max_tokens=512)
            decision_text = decision_output.strip()

            # 尝试解析 JSON
            if decision_text.startswith("{") and decision_text.endswith("}"):
                decision = json.loads(decision_text)

                if decision.get("action") == "run_analysis":
                    model_name = decision["model"]
                    analysis_type = decision["analysis_type"]
                    params = decision.get("params", {})
                    return await self._execute_analysis_flow(
                        model_name, analysis_type, params
                    )

                elif decision.get("action") == "clarify":
                    return f"💡 {decision['message']}"

            else:
                # LLM 直接回答（知识库命中）
                return decision_text

        except Exception as e:
            print(f"⚠️ LLM 决策失败，回退到规则引擎: {e}")
            return await self._fallback_rule_based_process(message)

    async def _fallback_rule_based_process(self, message: str) -> str:
        """原规则引擎逻辑（兼容 fallback）"""
        model_name = self._extract_model_name(message)
        analysis_type = self._extract_analysis_type(message)
        params = self._extract_parameters(message)

        if not params.get("batch_size"):
            params["batch_size"] = self.analysis_defaults.get("batch_size", [1])
        if not params.get("input_len"):
            params["input_len"] = self.analysis_defaults.get("input_len", [128])
        if not params.get("output_len"):
            params["output_len"] = self.analysis_defaults.get("output_len", [1])

        if model_name:
            model_path = self._resolve_model_path(model_name)
            if not model_path:
                return f"❌ 未找到模型 '{model_name}'，可用模型: {', '.join(self.model_mappings.keys())}"
            return await self._run_analysis(
                model_path=model_path, analysis_type=analysis_type, params=params
            )
        else:
            return f"💡 请指定模型，如“分析 qwen-7b”。可用模型: {', '.join(self.model_mappings.keys())}"

    async def _execute_analysis_flow(
        self, model_name: str, analysis_type: str, params: Dict
    ) -> str:
        model_path = self._resolve_model_path(model_name)
        if not model_path:
            return f"❌ 模型路径解析失败: {model_name}"
        return await self._run_analysis(
            model_path=model_path, analysis_type=analysis_type, params=params
        )

    async def _run_analysis(
        self, model_path: str, analysis_type: str, params: Dict
    ) -> str:
        results = []
        self.last_analysis_table = None
        self.last_analysis_reports = []
        self.last_analysis_dirs = []
        self.last_analysis_dir = None

        batch_sizes = params.get("batch_size", [1])
        input_lens = params.get("input_len", [128])
        output_lens = params.get("output_len", [1])

        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens

        # ========== 新增：测试模式 - 跳过真实分析 ==========
        if os.getenv("AGENT_TEST_MODE", "0") == "1":
            # 模拟一个性能报告文本
            mock_report = """
一、总体统计
- 总kernels数量: 42
- 总kernel执行时间: 125.6 ms

二、热点Kernels（按时间降序）
1. flash_attn_fwd_kernel
   - 执行时间: 45.2 ms
   - 时间占比: 36.0%
2. rms_norm_kernel
   - 执行时间: 28.7 ms
   - 时间占比: 22.8%
3. fused_mlp_kernel
   - 执行时间: 18.3 ms
   - 时间占比: 14.6%
"""
            report_path = self.results_dir / "mock_integrated_performance_report.md"
            report_path.write_text(mock_report.strip(), encoding="utf-8")

            # 构造模拟的 run_records
            run_records = [("0", self.results_dir)]

            # 继续走后续的表格生成和返回逻辑
        else:
            # ========== 原有真实分析逻辑（仅在非测试模式下执行）==========
            try:
                analysis_workflow = create_sglang_analysis_workflow()
                workflow_output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    analysis_workflow,
                    str(model_path),
                    batch_size,
                    input_len,
                    output_len,
                )

                run_records: List[Tuple[str, Path]] = []
                if isinstance(workflow_output, list):
                    for idx, item in enumerate(workflow_output):
                        gpu_label: str
                        output_path: Optional[str] = None
                        if isinstance(item, dict):
                            gpu_label = str(item.get("gpu", idx))
                            output_path = item.get("dir") or item.get("path")
                        else:
                            gpu_label = str(idx)
                            output_path = str(item)
                        if output_path:
                            run_records.append((gpu_label, Path(output_path)))
                elif workflow_output:
                    run_records.append(("0", Path(str(workflow_output))))

                if not run_records:
                    results.append("⚠️ **分析已完成，但未找到输出目录**")
                    return "\n".join(results)

            except Exception as e:
                import traceback

                error_detail = traceback.format_exc()
                results.append(f"""
❌ **分析执行失败**

错误信息: {str(e)}

详细错误:
{error_detail}

💡 **常见问题解决**:
1. 确保已安装 nsys 和 ncu 工具
2. 确保 SGlang 已正确安装
3. 确保模型文件路径正确
4. 确保有足够的 GPU 内存
""")
                return "\n".join(results)

        # ========== 以下为公共后处理逻辑（测试/真实模式共用）==========
        self.last_analysis_dirs = [str(path) for _, path in run_records]

        report_infos = []
        missing_reports = []
        for idx, (gpu_label, output_dir) in enumerate(run_records):
            report_path = output_dir / "integrated_performance_report.md"
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8")
                report_infos.append(
                    {
                        "gpu": gpu_label,
                        "dir": output_dir,
                        "report": report_path,
                        "text": report_text,
                        "index": idx,
                    }
                )
            else:
                missing_reports.append(output_dir)

        if not report_infos:
            dir_lines = "\n".join(f"  • {path}" for _, path in run_records)
            results.append(f"""
⚠️ **分析已完成，但未生成报告文件**

📁 结果目录:
{dir_lines}
💡 请检查目录中的其他输出文件
""")
            return "\n".join(results)

        primary_info = report_infos[0]
        self.last_analysis_dir = str(primary_info["dir"])
        self.last_analysis_reports = [str(info["report"]) for info in report_infos]
        summary = self._extract_report_summary(primary_info["text"])

        try:
            loop = asyncio.get_event_loop()
            if len(report_infos) > 1:
                table_markdown = self._generate_multi_gpu_table(
                    [info["text"] for info in report_infos],
                    [info["gpu"] for info in report_infos],
                )
            else:
                table_markdown = await loop.run_in_executor(
                    None, self._generate_report_table, primary_info["text"]
                )
        except Exception as table_exc:
            table_markdown = f"⚠️ 表格生成失败: {table_exc}"

        self.last_analysis_table = table_markdown

        dir_lines = "\n".join(
            f"  • {self._format_gpu_label(info['gpu'], info['index'])}: {info['dir']}"
            for info in report_infos
        )

        missing_lines = ""
        if missing_reports:
            missing_lines = "\n".join(f"  • {path}" for path in missing_reports)
            missing_lines = f"\n⚠️ 未找到以下目录的报告文件:\n{missing_lines}\n"

        results.append(f"""
✅ **分析完成!**

📁 **结果目录**:
{dir_lines}
📄 **报告文件**: {primary_info["report"]}
{missing_lines}
{summary}

📌 **热点Kernel表格预览**:
{table_markdown}

🔍 **详细报告**: 请查看 {primary_info["report"]}
📊 **可视化图表**: 请查看对应结果目录中的图片文件
""")

        return "\n".join(results)

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
        return client.report_to_table(report_text)

    def _generate_multi_gpu_table(
        self, report_texts: List[str], gpu_labels: List[str]
    ) -> str:
        if not report_texts:
            return "⚠️ 未找到可用的报告内容"

        parsed_entries = [
            self._parse_kernel_entries_from_report(text) for text in report_texts
        ]
        if not parsed_entries or not parsed_entries[0]:
            return "⚠️ 未能解析多GPU表格数据"

        label_cells = [
            self._format_gpu_label(lbl, idx) for idx, lbl in enumerate(gpu_labels)
        ]
        header_cells = ["Kernel"]
        for lbl in label_cells:
            header_cells.extend([f"{lbl} Duration(ms)", f"{lbl} Ratio(%)"])

        header = "| " + " | ".join(header_cells) + " |"
        divider = "| " + " | ".join(["---"] * len(header_cells)) + " |"

        max_len = max(len(entries) for entries in parsed_entries)
        rows = []
        for idx in range(max_len):
            name_candidates = []
            for entries in parsed_entries:
                if idx < len(entries) and entries[idx]["name"]:
                    name_candidates.append(entries[idx]["name"])
            base_name = name_candidates[0] if name_candidates else f"Kernel {idx + 1}"
            alt_names = {nm for nm in name_candidates if nm != base_name}
            if alt_names:
                merged_name = base_name + " / " + " / ".join(sorted(alt_names))
            else:
                merged_name = base_name

            row_cells = [merged_name]
            for entries in parsed_entries:
                if idx < len(entries):
                    row_cells.append(entries[idx]["duration"])
                    row_cells.append(entries[idx]["ratio"])
                else:
                    row_cells.extend(["", ""])
            rows.append("| " + " | ".join(row_cells) + " |")

        return "\n".join([header, divider, *rows])

    def _parse_kernel_entries_from_report(
        self, report_text: str
    ) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        lines = report_text.splitlines()
        idx = 0
        total_lines = len(lines)
        while idx < total_lines:
            raw_line = lines[idx]
            if raw_line.strip().startswith("二、"):
                break
            match = re.match(r"^\s*\d+\.\s+(.*)$", raw_line)
            if match:
                name = match.group(1).strip()
                duration = ""
                ratio = ""
                idx += 1
                while idx < total_lines:
                    line = lines[idx].strip()
                    if line.startswith("- 执行时间"):
                        dur_match = re.search(r"([0-9.]+)\s*ms", line)
                        if dur_match:
                            duration = dur_match.group(1)
                    elif line.startswith("- 时间占比"):
                        ratio_match = re.search(r"([0-9.]+)\s*%", line)
                        if ratio_match:
                            ratio = ratio_match.group(1)
                    elif re.match(r"^\s*\d+\.", lines[idx]) or line.startswith("二、"):
                        break
                    idx += 1
                entries.append({"name": name, "duration": duration, "ratio": ratio})
            else:
                idx += 1
        return entries

    @staticmethod
    def _format_gpu_label(label: str, index: int) -> str:
        if not label:
            return f"GPU{index}"
        normalized = label.strip()
        if not normalized:
            return f"GPU{index}"
        if normalized.lower().startswith("gpu"):
            return normalized.upper()
        return f"GPU{normalized}"

    def _extract_report_summary(self, report_content: str) -> str:
        lines = report_content.split("\n")
        summary_lines = []

        for i, line in enumerate(lines):
            if "总kernels数量" in line or "总kernel执行时间" in line:
                summary_lines.append(line)
            elif "🔥 识别的热点Kernels" in line:
                summary_lines.append("\n**🔥 热点Kernels (Top 3):**")
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() and lines[j].startswith(("1.", "2.", "3.")):
                        summary_lines.append(lines[j][:100])
                break

        if summary_lines:
            return "\n".join(summary_lines)
        else:
            return "**📊 分析报告已生成，请查看详细文件**"

    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        if model_name in self.model_mappings:
            mapped_path = self.model_mappings[model_name]
            if Path(mapped_path).is_absolute():
                return mapped_path
            full_path = self.models_path / mapped_path
            return str(full_path)

        if Path(model_name).exists():
            return model_name

        potential_path = self.models_path / model_name
        if potential_path.exists():
            return str(potential_path)

        return None

    def _extract_model_name(self, prompt: str) -> Optional[str]:
        for model_name in self.model_mappings.keys():
            if model_name.lower() in prompt.lower():
                return model_name

        patterns = [
            r"llama[^/\s]*-?\d*[^/\s]*-?\d+[bB]?",
            r"qwen[^/\s]*-?\d*[^/\s]*-?\d+[bB]?",
            r"chatglm[^/\s]*-?\d+[bB]?",
            r"baichuan[^/\s]*-?\d+[bB]?",
            r"vicuna[^/\s]*-?\d+[bB]?",
            r"mistral[^/\s]*-?\d+[bB]?",
            r"mixtral[^/\s]*-?\d+[bB]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _extract_analysis_type(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if (
            "ncu" in prompt_lower
            or "kernel" in prompt_lower
            or "深度" in prompt_lower
            or "nsight compute" in prompt_lower
        ):
            return "ncu (深度kernel分析)"
        elif (
            "nsys" in prompt_lower
            or "全局" in prompt_lower
            or "nsight systems" in prompt_lower
        ):
            return "nsys (全局性能分析)"
        elif "集成" in prompt_lower or "综合" in prompt_lower or "完整" in prompt_lower:
            return "auto (集成分析: nsys + ncu)"
        else:
            return "auto (集成分析: nsys + ncu)"

    def _extract_parameters(self, prompt: str) -> Dict:
        params = {}

        batch_match = re.search(
            r"batch[-_\s]*size?[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)", prompt, re.IGNORECASE
        )
        if batch_match:
            batch_sizes = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", batch_match.group(1))
                if x.strip()
            ]
            params["batch_size"] = batch_sizes

        input_match = re.search(
            r"input[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)",
            prompt,
            re.IGNORECASE,
        )
        if input_match:
            input_lens = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", input_match.group(1))
                if x.strip()
            ]
            params["input_len"] = input_lens

        output_match = re.search(
            r"output[-_\s]*len[gth]*[：:\s=]*(\d+(?:\s*[,，]\s*\d+)*)",
            prompt,
            re.IGNORECASE,
        )
        if output_match:
            output_lens = [
                int(x.strip())
                for x in re.split(r"[,，\s]+", output_match.group(1))
                if x.strip()
            ]
            params["output_len"] = output_lens

        return params

    def get_available_models(self) -> List[str]:
        return list(self.model_mappings.keys())

    def get_analysis_status(self) -> Dict:
        return {
            "available_models": self.get_available_models(),
            "results_directory": str(self.results_dir),
            "nsys_enabled": self.profiling_config.get("nsys", {}).get("enabled", True),
            "ncu_enabled": self.profiling_config.get("ncu", {}).get("enabled", True),
        }


# ==================== 简单测试用例 ====================
if __name__ == "__main__":
    from pathlib import Path

    # 创建测试文档目录
    test_doc_dir = Path("./documents")
    test_doc_dir.mkdir(exist_ok=True)
    test_file = test_doc_dir / "optim_tips.md"
    if not test_file.exists():
        test_file.write_text(
            "# Qwen 优化建议\n"
            "当 batch_size > 8 时，L2 缓存命中率显著下降。\n"
            "建议 input_len 控制在 512 以内以避免显存溢出。\n"
            "热点 kernel: flash_attn_fwd, rms_norm_kernel\n"
            "对于 qwen-1.8b，推荐 batch_size=1~4。"
        )

    # 初始化 KB 并加载
    kb = VectorKBManager()
    kb.add_document(str(test_file))

    # 模拟配置
    mock_config = {
        "sglang_path": "./SGlang",
        "models_path": "./models",
        "model_mappings": {
            "qwen-1.8b": "Qwen1.5-1.8B",
            "llama-3-8b": "Meta-Llama-3-8B",
        },
        "output": {"results_dir": "./test_results"},
        "analysis_defaults": {
            "batch_size": [1],
            "input_len": [128],
            "output_len": [32],
        },
    }

    agent = AIAgent(mock_config)

    async def run_tests():
        print("🔍 测试 1: 知识库问答（应直接回答）")
        resp1 = await agent.process_message("Qwen 大 batch 有什么问题？")
        print(resp1)
        print("\n" + "=" * 60 + "\n")

        print("🚀 测试 2: 启动性能分析（应触发分析流程）")
        resp2 = await agent.process_message(
            "分析 qwen-1.8b，batch_size=4, input_len=256"
        )
        print(resp2)

    asyncio.run(run_tests())
