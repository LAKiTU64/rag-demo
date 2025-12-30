from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional, List, Dict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

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
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def report_to_table(self, report_text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(report=report_text)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=10000,
            temperature=0.2,
            do_sample=False,
        )
        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt) :]
        cleaned = _extract_table_only(generated.strip())
        if not cleaned or cleaned.count("|") < 6:
            parsed_entries = _parse_kernel_entries(report_text)
            cleaned = _build_table_from_entries(parsed_entries)
        if not cleaned:
            cleaned = generated.strip()
        return _truncate_kernel_column(cleaned)

    # ✅ 新增方法：用于 Agentic-RAG 的决策推理
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        使用本地 Qwen 模型生成文本（用于 RAG 决策）
        """
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,  # 降低随机性，提高稳定性
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        # 去掉输入 prompt（如果模型返回了完整上下文）
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].lstrip()
        return generated.strip()


_client_lock = threading.Lock()
_cached_client: Optional[OfflineQwenClient] = None


def get_offline_qwen_client(model_dir: Path) -> OfflineQwenClient:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                _cached_client = OfflineQwenClient(model_dir)
    return _cached_client
