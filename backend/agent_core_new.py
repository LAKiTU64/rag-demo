import re
import os
import asyncio
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import yaml

# 设置多卡环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 导入分析工具
sys.path.insert(0, str(Path(__file__).parent))

from utils.nsys_to_ncu_analyzer import create_sglang_analysis_workflow
from offline_llm import get_offline_qwen_client
from knowledge_bases.vector_kb_manager import VectorKBManager


class AIAgent:
    """AI Agent核心类 - 自动化性能分析（支持 Agentic-RAG）"""

    def __init__(self, config: Dict):
        # === 保持原样，不做任何修改 ===
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
        """
        Agentic-RAG 主流程：
        1. 检索知识库（提供上下文）
        2. 由 LLM 完全解析用户意图、模型、参数、分析类型
        3. 若解析成功 → 执行分析；否则 → 抛出异常
        """

        # Step 1: 检索知识库（用于上下文，不影响决策）
        retrieved_contexts = self.kb.search(query=message, k=3)
        rag_context = ""
        if retrieved_contexts:
            # 使用完整内容，不截断
            rag_snippets = [
                f"【{res['doc_id']}】{res['content']}" for res in retrieved_contexts
            ]
            rag_context = "\n\n".join(rag_snippets)

        # Step 2: 让 LLM 完全解析结构化请求
        try:
            parsed_request = await self._parse_user_intent_with_llm(
                message, rag_context
            )
        except Exception as e:
            raise ValueError(f"LLM 无法解析用户请求: {e}")

        # Step 3: 执行分析（唯一出口）
        return await self._execute_analysis_flow(
            model_name=parsed_request["model"],
            analysis_type=parsed_request["analysis_type"],
            params=parsed_request["params"],
        )

    async def _parse_user_intent_with_llm(
        self, user_query: str, rag_context: str
    ) -> Dict:
        """
        由 LLM 完全解析用户意图，返回严格结构化字典。
        """

        prompt = f"""
你是一个高性能计算（HPC）与大模型性能分析专家。请严格按以下规则解析用户请求。

### 用户原始请求
{user_query}

### 相关知识库上下文（可选参考，但不要被误导）
{rag_context if rag_context else "无"}

### 输出要求
请输出一个 **严格符合 JSON 格式** 的对象，包含以下字段：
- "model": 字符串，模型名称（如 "qwen3-4b"）。必须从用户请求中提取，不要猜测。
- "analysis_type": 字符串，必须是以下之一：
    - "nsys" 表示全局性能分析（nsight systems）
    - "ncu" 表示深度 kernel 分析（nsight compute）
    - "auto" 表示集成分析（nsys + ncu）
- "params": 对象，包含以下可选数值数组：
    - "batch_size": 整数列表，如 [1]
    - "input_len": 整数列表，如 [128]
    - "output_len": 整数列表，如 [1]

### 注意
- 如果用户未指定 batch_size/input_len/output_len，请使用合理默认值（如 batch_size=[1]）。
- 不要输出任何解释、Markdown、或额外文本。
- 只输出 JSON。

### 示例输出
{{"model": "qwen3-4b", "analysis_type": "auto", "params": {{"batch_size": [1], "input_len": [128], "output_len": [1]}}}}
"""

        raw_output = self.llm_client.generate(
            prompt,
            max_tokens=512,
            mode="structured",  # 👈 关键：指定为结构化任务
        ).strip()

        # 强制 JSON 解析
        if not (raw_output.startswith("{") and raw_output.endswith("}")):
            raise ValueError(f"LLM 输出非 JSON 格式: {raw_output[:200]}...")

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e} | 原始输出: {raw_output[:200]}...")

        # 验证必要字段
        required_keys = {"model", "analysis_type", "params"}
        if not required_keys.issubset(result.keys()):
            raise ValueError(
                f"缺少必要字段。需要: {required_keys}, 实际: {set(result.keys())}"
            )

        # 确保 params 是 dict
        if not isinstance(result["params"], dict):
            result["params"] = {}

        # 补全默认参数（仅当缺失时）
        defaults = self.analysis_defaults
        if "batch_size" not in result["params"]:
            result["params"]["batch_size"] = defaults.get("batch_size", [1])
        if "input_len" not in result["params"]:
            result["params"]["input_len"] = defaults.get("input_len", [128])
        if "output_len" not in result["params"]:
            result["params"]["output_len"] = defaults.get("output_len", [1])

        # 标准化分析类型
        at = result["analysis_type"].lower()
        if "ncu" in at or "kernel" in at or "compute" in at:
            result["analysis_type"] = "ncu"
        elif "nsys" in at or "systems" in at or "global" in at:
            result["analysis_type"] = "nsys"
        else:
            result["analysis_type"] = "auto"

        return result

    async def _execute_analysis_flow(
        self, model_name: str, analysis_type: str, params: Dict
    ) -> str:
        model_path = self._resolve_model_path(model_name)
        if not model_path:
            raise ValueError(
                f"模型路径解析失败: '{model_name}'。可用模型: {list(self.model_mappings.keys())}"
            )
        return await self._run_analysis(
            model_path=model_path, analysis_type=analysis_type, params=params
        )

    # ========== 以下方法保持不变（从 _run_analysis 开始到文件结束）==========
    # （为节省篇幅，此处省略，实际使用时保留原代码）

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
            run_records = [("0", self.results_dir)]
        else:
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
                    raise RuntimeError("分析完成但未返回输出目录")

            except Exception as e:
                import traceback

                error_detail = traceback.format_exc()
                return f"""
❌ **分析执行失败**

错误信息: {str(e)}

详细错误:
{error_detail}

💡 **常见问题解决**:
1. 确保已安装 nsys 和 ncu 工具
2. 确保 SGlang 已正确安装
3. 确保模型文件路径正确
4. 确保有足够的 GPU 内存
"""

        # ========== 公共后处理逻辑 ==========
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
            return f"""
⚠️ **分析已完成，但未生成报告文件**

📁 结果目录:
{dir_lines}
💡 请检查目录中的其他输出文件
"""

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

        return f"""
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
"""

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        # 注意：这里修正了原代码的 bug（多了一个 self 参数）
        from offline_llm import get_offline_qwen_client

        # 实际应从配置获取路径，但为简化，假设 client 已存在
        # 更好的做法是传入 client，但为兼容性，临时重建
        # TODO: 后续可注入 client
        client = get_offline_qwen_client(Path(__file__).parent / "dummy")  # 仅示意
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
        # 此方法现在仅用于 _resolve_model_path 的辅助，主逻辑由 LLM 负责
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
        # 此方法现在仅用于 fallback（但已移除），保留仅为兼容
        prompt_lower = prompt.lower()
        if (
            "ncu" in prompt_lower
            or "kernel" in prompt_lower
            or "深度" in prompt_lower
            or "nsight compute" in prompt_lower
        ):
            return "ncu"
        elif (
            "nsys" in prompt_lower
            or "全局" in prompt_lower
            or "nsight systems" in prompt_lower
        ):
            return "nsys"
        else:
            return "auto"

    def _extract_parameters(self, prompt: str) -> Dict:
        # 此方法现在仅用于 fallback（但已移除），保留仅为兼容
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
    import yaml
    from pathlib import Path
    import asyncio

    # 导入config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # agent初始化
    agent = AIAgent(config_yaml)

    # 构建知识库
    document_dir = Path("documents")
    if document_dir.exists():
        for file_path in document_dir.iterdir():
            if file_path.is_file():
                agent.kb.add_document(str(file_path))
                print(f"已添加文档: {file_path}")
    else:
        print(f"文档目录不存在: {document_dir}")

    # ==============================
    # 🔧 新增：测试结构化意图解析（调试用）
    # ==============================
    async def test_structured_parsing():
        print("\n🧪 测试结构化意图解析...")
        user_query = "分析一下qwen3-4b模型，batch_size=1"
        rag_context = ""  # 可留空或模拟
        try:
            intent = await agent._parse_user_intent_with_llm(user_query, rag_context)
            print(f"✅ 解析成功: {intent}")
        except Exception as e:
            print(f"❌ 解析失败: {e}")

    # ==============================
    # 🔍 原有：端到端问答测试
    # ==============================
    async def run_end_to_end_test():
        print("\n🔍 端到端问答测试...")
        try:
            response = await agent.process_message("分析一下qwen3-4b模型，batch_size=1")
            print(f"✅ 最终响应:\n{response}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # ==============================
    # 🚀 运行测试
    # ==============================
    print("🚀 启动测试套件")
    asyncio.run(test_structured_parsing())  # 先测解析
    asyncio.run(run_end_to_end_test())  # 再测完整流程
