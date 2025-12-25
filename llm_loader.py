# llm_loader.py
import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


def load_qwen3_14b_local(model_path="~/models/Qwen/Qwen3-14B"):
    model_path = os.path.expanduser(model_path)

    # 不使用量化，直接加载 BF16 模型，利用 4xA10 多卡
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,  # Qwen 官方推荐 use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",          # 自动将层分配到多卡
        trust_remote_code=True,     # Qwen 必须开启
        torch_dtype=torch.bfloat16, # A10 支持 bfloat16，高效且精度高
        # attn_implementation="flash_attention_2",  # 可选：若安装了 flash-attn 可加速
    )

    # 构建生成 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
