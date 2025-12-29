#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单测试: 知识库 JSON 摄取流程"""
from pathlib import Path
import json

from knowledge_bases.kb_ingest import ingest_json_to_faiss, DEFAULT_INDEX_DIR

def main():
    sample = {
        "theoretical_limits": {
            "memory_bandwidth": {
                "GPU_A": "1200 GB/s",
                "GPU_B": "900 GB/s"
            },
            "latency_models": {
                "attention": "O(n^2) 受限于序列长度",
                "flash_attention": "近似 O(n * sqrt(n)) 在优化实现中降低常数项"
            },
            "scaling": {
                "tokens_vs_params": "Chinchilla optimal scaling law",
                "compute_opt": "根据 FLOPs 与训练步数平衡"
            }
        }
    }
    json_str = json.dumps(sample, ensure_ascii=False)
    result = ingest_json_to_faiss(json_str)
    print("摄取结果:", result)
    assert result["status"] == "success"
    assert result["fragments_added"] > 0
    assert Path(result["index_dir"]).exists()

if __name__ == "__main__":
    main()
