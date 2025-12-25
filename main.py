# main.py
from llm_loader import load_qwen3_14b_local
from rag_agent import create_rag_chain
from VectorKBManager import VectorKBManager

if __name__ == "__main__":
    # 1. 加载向量库
    kb = VectorKBManager()

    # 2. 加载本地 LLM
    print("正在加载 Qwen3-14B 模型...")
    llm = load_qwen3_14b_local("~/models/Qwen/Qwen3-14B")

    # 3. 创建 RAG Chain
    rag = create_rag_chain(kb, llm)

    # 4. 交互式问答
    print("\n✅ RAG Agent 已启动！输入 'quit' 退出。\n")
    while True:
        question = input("👤 你: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        try:
            answer = rag.invoke(question)
            print(f"🤖 助手: {answer}\n")
        except Exception as e:
            print(f"❌ 错误: {e}\n")
