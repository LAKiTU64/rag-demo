# rag_agent.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def create_rag_chain(vector_kb_manager, llm):
    retriever = vector_kb_manager.as_retriever(search_kwargs={"k": 3, "t": 0.5})

    # 构建 Prompt（适配 Qwen 的对话格式）
    prompt_template = """
    你是一个智能助手，请根据以下上下文回答用户的问题。如果上下文不足以回答，请说“根据现有资料无法回答”。

    上下文：
    {context}

    问题：{question}

    回答：
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
