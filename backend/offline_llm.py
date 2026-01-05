from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, List, Dict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_PROMPT_TEMPLATE = (
    "你是GPU性能分析专家。阅读以下性能报告，为其中列出的每个 kernel 生成 Markdown 表格，"
    "列出 Kernel 名称、执行时长(ms)、时间占比(%)。\n"
    "表头固定为: Kernel | Duration(ms) | Ratio(%)。\n"
    "报告内容如下:\n\n{report}\n\n 请仅输出 Markdown 表格。"
)


def _truncate_kernel_column(markdown: str, max_len: int = 30) -> str:
    lines = markdown.splitlines()
    truncated = []
    for line in lines:
        stripped = line.strip()
        if "|" not in stripped:
            truncated.append(line)
            continue

        core = stripped.strip("|")
        cells = [cell.strip() for cell in core.split("|")]
        if not cells:
            truncated.append(line)
            continue

        header_or_separator = cells[0].lower() == "kernel" or all(
            set(cell) <= {"-", ":", " "} for cell in cells
        )
        if header_or_separator:
            truncated.append(line)
            continue

        rebuilt = "| " + " | ".join(cells) + " |"
        truncated.append(rebuilt)
    return "\n".join(truncated)


def _extract_table_only(markdown: str) -> str:
    lines = markdown.splitlines()
    collected = []
    table_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if table_started:
                break
            continue
        if "|" not in stripped:
            if table_started:
                break
            continue
        if not table_started:
            header_cells = [
                cell.strip().lower() for cell in stripped.strip("|").split("|")
            ]
            if not header_cells or "kernel" not in header_cells[0]:
                continue
            table_started = True
        collected.append(stripped if stripped.startswith("|") else f"| {stripped} |")
    return "\n".join(collected).strip()


def _parse_kernel_entries(report_text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in report_text.splitlines():
        number_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if number_match:
            if current.get("name") and current.get("duration") and current.get("ratio"):
                entries.append(current)
            current = {"name": number_match.group(1).strip()}
            continue
        if not current:
            continue
        if "duration" not in current:
            time_match = re.search(r"执行时间[:：]\s*([0-9.]+)\s*ms", line)
            if time_match:
                current["duration"] = time_match.group(1)
                continue
        if "ratio" not in current:
            ratio_match = re.search(r"时间占比[:：]\s*([0-9.]+)%", line)
            if ratio_match:
                current["ratio"] = ratio_match.group(1)
                continue
    if current.get("name") and current.get("duration") and current.get("ratio"):
        entries.append(current)
    return entries


def _build_table_from_entries(entries: List[Dict[str, str]]) -> str:
    if not entries:
        return ""
    rows = ["| Kernel | Duration(ms) | Ratio(%) |", "| --- | --- | --- |"]
    for item in entries:
        rows.append(f"| {item['name']} | {item['duration']} | {item['ratio']} |")
    return "\n".join(rows)


class OfflineQwenClient:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        # 不再使用 TextGenerationPipeline（它不支持 chat template）
        # 改为直接调用 model.generate()

    def report_to_table(self, report_text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(report=report_text)
        # 构造聊天消息
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=10000,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_ids.shape[1] :]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        cleaned = _extract_table_only(generated.strip())
        if not cleaned or cleaned.count("|") < 6:
            parsed_entries = _parse_kernel_entries(report_text)
            cleaned = _build_table_from_entries(parsed_entries)
        if not cleaned:
            cleaned = generated.strip()
        return _truncate_kernel_column(cleaned)

    # ✅ 修复后的 generate 方法：支持 chat template + 确定性输出
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        mode: str = "conversation",  # 新增参数，默认为对话模式
    ) -> str:
        """
        使用本地 Qwen 模型生成文本。

        Args:
            prompt: 用户输入
            max_tokens: 最大生成长度
            mode:
                - "structured": 用于工具调用、意图解析等，强制纯 JSON，禁用 <think>
                - "conversation": 用于回答用户问题，允许自然语言、推理、<think>
        """
        if mode == "structured":
            messages = [
                {
                    "role": "system",
                    "content": "你是一个高性能计算专家。请直接输出结果，不要任何思考过程，不要使用 <think> 标签，不要解释，不要 Markdown，只输出纯 JSON。",
                },
                {"role": "user", "content": prompt},
            ]
            temperature = 0.0
            do_sample = False
            top_p = None
        elif mode == "conversation":
            messages = [{"role": "user", "content": prompt}]
            temperature = 0.3
            do_sample = True
            top_p = 0.9
        else:
            raise ValueError("mode 必须是 'structured' 或 'conversation'")

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        outputs = self.model.generate(input_ids, **gen_kwargs)
        generated_ids = outputs[0][input_ids.shape[1] :]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 仅在 structured 模式下清理 <think>（保险）
        if mode == "structured":
            generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL)
            generated = generated.strip()

        return generated


_client_lock = threading.Lock()
_cached_client: Optional[OfflineQwenClient] = None


def get_offline_qwen_client(model_dir: Path) -> OfflineQwenClient:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = OfflineQwenClient(model_dir)
    return _cached_client
