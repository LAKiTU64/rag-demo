import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from knowledge_bases.vector_kb_manager import VectorKBManager
from offline_llm import get_offline_qwen_client


class AIAgent:
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
        
