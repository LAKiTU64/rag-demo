import os
from typing import List, TypedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, END

from VectorKBManager import VectorKBManager


# --- Config ---
MODEL_PATH = "./.models/Qwen/Qwen3-4B"


# ===============================
# 1. LLM 构建（本地 Qwen3-4B）
# ===============================
def load_local_llm():
    model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型不存在: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.05,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)


# ===============================
# 2. Graph State
# ===============================
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


# ===============================
# 3. Agentic-RAG
# ===============================
class MyRagAgent:
    def __init__(self, kb: VectorKBManager):
        self.kb = kb
        self.llm = load_local_llm()
        self.retriever = kb.as_retriever(k=3, t=0.5)

        self.graph = self._build_graph()

    # ---------- Node: 检索 ----------
    def retrieve_node(self, state: RAGState):
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs}

    # ---------- Node: 生成 ----------
    def generate_node(self, state: RAGState):
        context = "\n\n".join(d.page_content for d in state["documents"])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个信息抽取引擎，只能从上下文中提取已有信息，"
                    "不得总结、不得推理、不得补充不存在的内容。"
                    "如果无法确定，请回答“未在文档中明确给出”。",
                ),
                (
                    "human",
                    "上下文（可能包含多个Kernel，请逐一判断）：\n{context}\n\n"
                    "问题：{question}\n\n"
                    "要求：\n"
                    "1. 只列出满足条件的 Kernel 名称\n"
                    "2. 给出对应的具体数值\n"
                    "3. 不要解释推理过程",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke(
            {
                "context": context,
                "question": state["question"],
            }
        )

        return {"answer": response}

    # ---------- Graph ----------
    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate", self.generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ---------- API ----------
    def ask(self, question: str) -> str:
        result = self.graph.invoke(
            {
                "question": question,
                "documents": [],
                "answer": "",
            }
        )
        return result["answer"]


# ===============================
# 4. main 测试
# ===============================
if __name__ == "__main__":
    # ---- 初始化向量库 ----
    kb = VectorKBManager()

    # ---- 加载 documents ----
    DOCS_DIR = "./documents"

    print(f"\n🚀 加载文档目录: {DOCS_DIR}")
    for fname in os.listdir(DOCS_DIR):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath):
            kb.add_document(fpath)

    kb.get_overview()

    # ---- Agent ----
    agent = MyRagAgent(kb)

    # ---- Query 1 ----
    q1 = "哪些Kernel函数的瓶颈数>=3？"
    print("\n🧪 Query 1:", q1)
    print("🤖 Answer:\n", agent.ask(q1))

    # ---- Query 2 ----
    q2 = "哪个Kernel的执行时间占比最高？"
    print("\n🧪 Query 2:", q2)
    print("🤖 Answer:\n", agent.ask(q2))
