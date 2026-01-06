import re
import os
import asyncio
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import yaml

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

# 导入分析工具 (确保路径正确)
sys.path.insert(0, str(Path(__file__).parent))

from utils.nsys_to_ncu_analyzer import create_sglang_analysis_workflow
from offline_llm import get_offline_qwen_client
from knowledge_bases.vector_kb_manager import VectorKBManager


class AIAgent:
    """AI Agent核心类 - V3版 (Intent-First: Analysis/Chat/QA)"""

    def __init__(self, config: Dict):
        self.config = config

        # sglang 和模型路径
        self.sglang_path = Path(config.get("sglang_path"))
        self.models_path = Path(config.get("models_path"))
        self.model_mappings = config.get("model_mappings", {})

        # 输出目录
        self.results_dir = Path(config.get("output", {}).get("results_dir", "results"))
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # 本地 LLM 客户端
        self.offline_qwen_path = Path(config.get("offline_qwen_path"))
        self.llm_client = get_offline_qwen_client(self.offline_qwen_path)

        # 分析工具配置
        self.profiling_config = config.get("profiling_tools", {})
        self.analysis_defaults = config.get("analysis_defaults", {})

        # 缓存
        self.last_analysis_dir: Optional[str] = None
        self.last_analysis_dirs: List[str] = []
        self.last_analysis_reports: List[str] = []
        self.last_analysis_table: Optional[str] = None

        # 向量知识库相关
        self.kb = VectorKBManager()
        kb_config = config.get("vector_store", {})
        self.persist_directory = kb_config.get("persist_directory")
        self.embedding_model = kb_config.get("embedding_model")

        # 对话历史缓冲区
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_turns = 6  # 保留最近 6 轮对话

    async def process_message(self, message: str) -> str:
        """
        Agentic-RAG 主流程 (V3 - 三轨并行):
        1. [Router] 意图识别 (Analysis / Chat / QA)
        2. [Branch]
           - Analysis: 执行工具 (Action)
           - Chat: 自由闲聊 (Free Style)
           - QA: 检索知识库 (Strict RAG)
        """
        # Step 1: 意图路由
        try:
            decision = await self._parse_intent_three_way(message, self.chat_history)
        except Exception as e:
            return f"❌ **意图识别失败**: {str(e)}"

        intent = decision.get("intent", "qa")
        response_text = ""

        # Step 2: 分支处理
        if intent == "analysis":
            # === 分支 A: 性能分析 (Action) ===
            print(
                f"[DEBUG] 识别为分析意图: 模型={decision.get('model')}, 参数={decision.get('params')}"
            )
            try:
                analysis_result = await self._execute_analysis_flow(
                    model_name=decision.get("model"),
                    analysis_type=decision.get("analysis_type", "auto"),
                    params=decision.get("params", {}),
                )
                response_text = analysis_result
            except Exception as e:
                response_text = f"❌ **分析启动失败**: {str(e)}"

        elif intent == "chat":
            # === 分支 B: 纯闲聊 (Free Style) ===
            # 不查库，给予模型自由度
            chat_prompt = f"""
你是一个专业但友好的 AI 性能分析专家。请简短、自然地回复用户的闲聊。
不要胡编乱造技术数据，但可以进行自我介绍或日常对话。

用户: {message}
助手:
"""
            try:
                raw_res = self.llm_client.generate(chat_prompt, max_tokens=256).strip()
                response_text = f"🤖 **闲聊模式**\n{raw_res}"
            except Exception as e:
                response_text = f"❌ **回复生成失败**: {str(e)}"

        else:
            # === 分支 C: 专业问答 (Strict RAG) ===
            # 关键修改：k=6，大幅增加长尾 Kernel 名称的召回率
            retrieved_contexts = self.kb.search(query=message, k=6)
            rag_context = ""
            if retrieved_contexts:
                rag_snippets = [
                    f"【文档片段 {i + 1}】\n{res['content']}"
                    for i, res in enumerate(retrieved_contexts)
                ]
                rag_context = "\n\n".join(rag_snippets)

            try:
                answer = await self._generate_strict_qa_response(message, rag_context)
                ref_count = len(retrieved_contexts)
                response_text = f"🤖 **专业问答**\n{answer}\n\n---\n💡 *基于 {ref_count} 条知识库片段回答*"
            except Exception as e:
                response_text = f"❌ **回答生成失败**: {str(e)}"

        # Step 3: 更新对话历史
        self.chat_history.append({"role": "user", "content": message})

        # 简化历史存储
        history_response = response_text
        if intent == "analysis":
            model_used = decision.get("model")
            history_response = f"已完成对 {model_used} 的性能分析。"

        self.chat_history.append({"role": "assistant", "content": history_response})
        if len(self.chat_history) > self.max_history_turns * 2:
            self.chat_history = self.chat_history[-self.max_history_turns * 2 :]

        return response_text

    async def _parse_intent_three_way(
        self, user_query: str, history: List[Dict[str, str]]
    ) -> Dict:
        """
        阶段一：三分类意图识别 (修复版 - 强化语义理解，拒绝无脑关键词)
        """
        available_models = list(self.model_mappings.keys())
        models_str = ", ".join([f'"{m}"' for m in available_models])

        history_str = "无"
        if history:
            history_lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"].replace("\n", " ")[:100]
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)

        prompt = f"""
你是一个中枢路由 Agent。请根据用户输入的**语义**（而不仅仅是关键词）判断意图。

### 可用模型参考
[{models_str}]

### 核心判别逻辑 (Logic) - 请仔细区分 "询问" 与 "执行"

1. **Analysis (执行分析)**:
   - **核心特征**: 用户想**立即运行**某个任务，或者**设置**参数来跑测试。
   - **强触发词**: "分析", "运行", "测一下", "跑", "profile", "ncu", "nsys".
   - **参数设置**: 只有当包含**赋值意图**时（如 "bs=1", "bs设为4", "batch_size 为 1"），才算 Analysis。
   - **示例**: "跑一下 qwen", "分析 qwen batch_size=1", "测试性能".

2. **QA (专业问答/咨询)**:
   - **核心特征**: 用户想**获取知识**、询问建议、查询文档或数据。
   - **强触发词**: "推荐", "是多少", "范围", "什么", "瓶颈", "文档".
   - **关键区分**: 如果用户问 "推荐 batch_size 是多少"，这是 **QA**，不是 Analysis！
   - **示例**: "qwen 推荐的 batch_size 是多少", "kernel 0 的瓶颈是什么", "显存占用高吗".

3. **Chat (闲聊)**:
   - 纯粹的社交、打招呼、自我介绍。
   - 示例: "你好", "你是谁", "谢谢".

### 用户输入
{user_query}

### 对话历史
{history_str}

### 输出格式 (JSON)
{{
    "intent": "analysis" | "qa" | "chat",
    "model": "模型名 (Analysis模式必填，QA模式可留空)",
    "params": {{ "batch_size": [1], ... }}
}}
"""
        raw_output = self.llm_client.generate(
            prompt, max_tokens=256, mode="structured"
        ).strip()

        # JSON 清洗
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        try:
            result = json.loads(raw_output)
        except Exception:
            # 兜底策略修正：不要看到 batch_size 就认为是 analysis
            # 只有包含明确动作动词时，才兜底为 analysis
            action_keywords = ["分析", "跑", "测", "profile", "运行"]
            if any(k in user_query for k in action_keywords):
                return {"intent": "analysis", "model": "", "params": {}}
            return {"intent": "qa"}

        # Analysis 参数补全 (保持不变)
        if result.get("intent") == "analysis":
            if "params" not in result or not isinstance(result["params"], dict):
                result["params"] = {}
            defaults = self.analysis_defaults
            for key in ["batch_size", "input_len", "output_len"]:
                if key not in result["params"]:
                    result["params"][key] = defaults.get(key, [1])

        return result

    async def _generate_strict_qa_response(
        self, user_query: str, rag_context: str
    ) -> str:
        """
        阶段二（仅 QA）：极度严格的 RAG 生成
        """
        prompt = f"""
你是一个严谨的数据分析员。你必须完全依据【参考资料】回答用户关于 GPU 性能数据的提问。

### 参考资料
{rag_context if rag_context else "（警告：未检索到相关文档，可能需要告知用户资料缺失）"}

### 用户问题
{user_query}

### 严格约束 (Strict Rules)
1. **数据精确性**：如果用户询问某个 Kernel 的具体指标（如瓶颈数、带宽），**必须**在参考资料中找到**完全匹配**的 Kernel 名称后才能回答。
2. **拒绝猜测**：如果资料里有 "Kernel A" 和 "Kernel B"，但用户问 "Kernel C"，你必须回答："资料中未找到 Kernel C 的数据"。**严禁**把 A 的数据安在 C 头上。
3. **原文引用**：回答时尽量使用资料中的原话或数据。
4. **空值处理**：如果资料为空或不相关，直接回答：“抱歉，知识库中没有相关信息。”

### 回答：
"""
        return self.llm_client.generate(prompt, max_tokens=1024).strip()

    async def _execute_analysis_flow(
        self, model_name: str, analysis_type: str, params: Dict
    ) -> str:
        model_path = self._resolve_model_path(model_name)
        if not model_path:
            # 明确抛出错误，让用户知道是模型配置问题
            raise ValueError(
                f"模型路径解析失败: '{model_name}'。\n"
                f"请检查 config.yaml 中的 'model_mappings' 是否包含该模型，"
                f"或者模型文件是否存在于: {self.models_path}"
            )
        return await self._run_analysis(
            model_path=model_path, analysis_type=analysis_type, params=params
        )

    async def _run_analysis(
        self, model_path: str, analysis_type: str, params: Dict
    ) -> str:
        # 参数提取
        batch_sizes = params.get("batch_size", [1])
        input_lens = params.get("input_len", [128])
        output_lens = params.get("output_len", [1])
        batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
        input_len = input_lens[0] if isinstance(input_lens, list) else input_lens
        output_len = output_lens[0] if isinstance(output_lens, list) else output_lens

        # Mock 模式 (开发调试用)
        if os.getenv("AGENT_TEST_MODE", "0") == "1":
            print("[DEBUG] 运行在测试模式 (Mock Analysis)")
            mock_dir = self.results_dir / f"mock_analysis_b{batch_size}"
            mock_dir.mkdir(exist_ok=True)
            mock_report = f"""
一、总体统计
- 模型: {Path(model_path).name}
- Batch: {batch_size}, Input: {input_len}
- 总kernels数量: 42
- 总kernel执行时间: 125.6 ms

二、热点Kernels（按时间降序）
1. flash_attn_fwd_kernel
   - 执行时间: 45.2 ms
   - 时间占比: 36.0%
2. rms_norm_kernel
   - 执行时间: 28.7 ms
   - 时间占比: 22.8%
"""
            report_path = mock_dir / "integrated_performance_report.md"
            report_path.write_text(mock_report.strip(), encoding="utf-8")
            run_records = [("0", mock_dir)]
        else:
            # 真实运行
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
                        output_path = (
                            item.get("dir") or item.get("path")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        gpu_label = (
                            str(item.get("gpu", idx))
                            if isinstance(item, dict)
                            else str(idx)
                        )
                        if output_path:
                            run_records.append((gpu_label, Path(output_path)))
                elif workflow_output:
                    run_records.append(("0", Path(str(workflow_output))))

                if not run_records:
                    raise RuntimeError("分析完成但未返回输出目录")

            except Exception as e:
                import traceback

                return f"""
❌ **分析执行失败**
错误信息: {str(e)}
详细错误:
{traceback.format_exc()}
"""

        # 结果后处理
        report_infos = []
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
                    }
                )

        if not report_infos:
            return f"⚠️ 分析完成，但未生成报告文件。\n目录: {[str(p[1]) for p in run_records]}"

        primary_info = report_infos[0]
        summary = self._extract_report_summary(primary_info["text"])

        # 生成表格
        try:
            if len(report_infos) > 1:
                table_markdown = self._generate_multi_gpu_table(
                    [info["text"] for info in report_infos],
                    [info["gpu"] for info in report_infos],
                )
            else:
                table_markdown = self._generate_report_table(primary_info["text"])
        except Exception:
            table_markdown = "⚠️ (表格生成失败)"

        dir_lines = "\n".join(
            f"  • {info['gpu']}: {info['dir']}" for info in report_infos
        )

        return f"""
✅ **分析完成!**

📁 **结果目录**:
{dir_lines}
📄 **报告文件**: {primary_info["report"]}
{summary}

📌 **热点Kernel表格预览**:
{table_markdown}
"""

    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        if not model_name:
            return None
        # 1. 映射表
        if model_name in self.model_mappings:
            mapped_path = self.model_mappings[model_name]
            if Path(mapped_path).is_absolute():
                return mapped_path
            return str(self.models_path / mapped_path)
        # 2. 物理路径检查
        if Path(model_name).exists():
            return model_name
        potential_path = self.models_path / model_name
        if potential_path.exists():
            return str(potential_path)
        return None

    @staticmethod
    def _generate_report_table(report_text: str) -> str:
        # 简易表格生成
        from offline_llm import get_offline_qwen_client

        client = get_offline_qwen_client(Path(__file__).parent / "dummy")
        return client.report_to_table(report_text)

    def _generate_multi_gpu_table(
        self, report_texts: List[str], gpu_labels: List[str]
    ) -> str:
        # 复用多卡逻辑 (简化版)
        if not report_texts:
            return ""
        entries = self._parse_kernel_entries_from_report(report_texts[0])
        header = (
            "| Kernel | " + " | ".join([f"{lbl} Duration" for lbl in gpu_labels]) + " |"
        )
        sep = "|---" * (len(gpu_labels) + 1) + "|"
        rows = []
        for entry in entries[:5]:  # Top 5
            rows.append(
                f"| {entry['name']} | {entry['duration']} |"
                + " ... |" * (len(gpu_labels) - 1)
            )
        return f"{header}\n{sep}\n" + "\n".join(rows)

    def _parse_kernel_entries_from_report(
        self, report_text: str
    ) -> List[Dict[str, str]]:
        entries = []
        lines = report_text.splitlines()
        current_entry = {}
        for line in lines:
            name_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
            if name_match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = {
                    "name": name_match.group(1).strip(),
                    "duration": "-",
                    "ratio": "-",
                }
            dur_match = re.search(r"执行时间[:\s]+([0-9.]+\s*ms)", line)
            if dur_match and current_entry:
                current_entry["duration"] = dur_match.group(1)
            if "二、" in line:
                break
        if current_entry:
            entries.append(current_entry)
        return entries

    def _extract_report_summary(self, report_content: str) -> str:
        lines = report_content.split("\n")
        summary_lines = []
        for i, line in enumerate(lines):
            if "总kernels数量" in line or "总kernel执行时间" in line:
                summary_lines.append(line)
            elif "热点Kernels" in line:
                summary_lines.append("\n**🔥 热点Kernels (Top 3):**")
                count = 0
                for j in range(i + 1, len(lines)):
                    if re.match(r"^\s*\d+\.", lines[j]):
                        summary_lines.append(lines[j][:100])
                        count += 1
                        if count >= 3:
                            break
                break
        return "\n".join(summary_lines) if summary_lines else ""


# ==================== Main CLI ====================
if __name__ == "__main__":
    # 1. Load Config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到 {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # 2. Init Agent
    print("🔄 正在初始化 AI Agent...")
    try:
        agent = AIAgent(config_yaml)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 3. Load Knowledge Base
    document_dir = Path("documents")
    if document_dir.exists():
        print("📚 正在加载知识库文档...")
        count = 0
        for file_path in document_dir.iterdir():
            if file_path.is_file() and file_path.suffix in [".md", ".txt"]:
                agent.kb.add_document(str(file_path))
                count += 1
        print(f"✅ 已加载 {count} 个文档。")
    else:
        print("⚠️ 文档目录不存在，跳过加载。")

    # 4. Interactive Loop
    async def interactive_chat_loop():
        style = Style.from_dict({"user-prompt": "#00aa00 bold", "text": "#ffffff"})
        session = PromptSession(history=InMemoryHistory())

        print("\n" + "=" * 60)
        print("🤖 AI 性能分析助手 (V3 - 混合意图模式)")
        print("💡 支持指令: '分析 qwen' | 提问: '瓶颈是什么' | 闲聊: '你是谁'")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = await session.prompt_async(
                    HTML("<user-prompt>User ></user-prompt> "), style=style
                )
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 再见！")
                    break

                print("\n⏳ Agent 正在思考...")
                response = await agent.process_message(user_input)
                print("-" * 20 + " Agent 回复 " + "-" * 20)
                print(response)
                print("-" * 52 + "\n")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

    asyncio.run(interactive_chat_loop())
