import torch

# --- LangGraph 核心组件 ---
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- 模型相关 ---
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 类型提示 ---
from typing import Annotated, Sequence, TypedDict, Literal

# --- 知识库 ---
from VectorKBManager import VectorKBManager

# --- Config ---
MODEL_PATH = "/workspaces/rag-demo/.models/Qwen/Qwen3-4B"


# ==============================================================================
# 1. 定义 Agent 状态
# ==============================================================================
class AgentState(TypedDict):
    """Agent 的状态定义"""

    # add_messages 是一个 reducer，会自动合并消息列表
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ==============================================================================
# 2. 模型加载
# ==============================================================================
def load_local_llm(model_path):
    """加载本地 HuggingFace 模型并包装为 LangChain LLM"""
    print(f"⏳ 正在加载本地模型: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )

        # 创建 HuggingFace Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.1})

        print("✅ 模型加载成功!")
        return llm

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


# ==============================================================================
# 3. 定义工具
# ==============================================================================
kb = VectorKBManager()


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索内部知识库获取相关文档。

    当需要回答关于特定事实、内部文档、公司数据，或者你不确定答案时，
    必须使用此工具进行搜索。

    Args:
        query: 搜索查询字符串，应该是具体的关键词或问题

    Returns:
        检索到的相关文档内容
    """
    print(f"\n🔍 [Tool] 正在检索知识库: '{query}'")

    try:
        results = kb.search(query, k=3)

        if not results:
            return "❌ 知识库中没有找到相关信息。"

        # 构建结构化上下文
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[文档 {i}] (相似度: {r.get('score', 'N/A')})\n{r['content']}\n"
            )

        context = "\n".join(context_parts)
        print(f"✅ 找到 {len(results)} 条相关文档")
        return context

    except Exception as e:
        return f"⚠️ 检索过程出错: {str(e)}"


# ==============================================================================
# 4. 构建 LangGraph Agent (从零构建，完全控制)
# ==============================================================================
def build_react_agent(llm):
    """
    使用 LangGraph 构建 ReAct Agent（兼容 HuggingFace 模型）

    由于 HuggingFacePipeline 不支持原生工具调用，我们使用 ReAct 提示词模式
    """
    print("\n🔧 正在构建 ReAct Agent Graph...")

    # 准备工具
    tools = [search_knowledge_base]
    tools_dict = {tool.name: tool for tool in tools}

    # 构建工具描述（供 LLM 参考）
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )

    # ========== 定义节点函数 ==========

    def call_model(state: AgentState):
        """Agent 节点：使用 ReAct 提示词调用 LLM"""
        messages = state["messages"]

        # 获取对话历史
        conversation_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation_history.append(f"Assistant: {msg.content}")

        # 构建 ReAct 风格的提示词
        react_prompt = f"""你是一个专业的 AI 助手，可以使用以下工具来回答问题：

{tool_descriptions}

请按照以下格式思考和回答：
Thought: [你对问题的思考]
Action: [工具名称]
Action Input: [工具的输入参数]
Observation: [工具返回的结果会在这里]
... (重复 Thought/Action/Action Input/Observation 直到你知道最终答案)
Final Answer: [给用户的最终回答]

对话历史：
{chr(10).join(conversation_history)}

现在开始回答最后一个问题。记住：如果需要查找信息，必须使用工具！
"""

        # 调用模型
        response = llm.invoke(react_prompt)

        # 解析响应，检查是否需要调用工具
        ai_message = AIMessage(content=response)

        return {"messages": [ai_message]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """
        条件边：通过解析 AI 响应判断是否需要调用工具
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage):
            return "end"

        content = last_message.content.strip()

        # 检查是否包含 "Action:" 关键字（ReAct 模式）
        if "Action:" in content and "Final Answer:" not in content:
            return "tools"

        return "end"

    def execute_tools(state: AgentState):
        """工具节点：解析并执行工具调用"""
        messages = state["messages"]
        last_message = messages[-1]

        # 解析工具调用
        content = last_message.content

        # 简单的解析逻辑
        tool_name = None
        tool_input = None

        for line in content.split("\n"):
            if line.startswith("Action:"):
                tool_name = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                tool_input = line.replace("Action Input:", "").strip()

        # 执行工具
        if tool_name and tool_name in tools_dict and tool_input:
            print(f"🛠️  执行工具: {tool_name}('{tool_input}')")
            try:
                result = tools_dict[tool_name].invoke(tool_input)
                observation = f"\nObservation: {result}\n"
            except Exception as e:
                observation = f"\nObservation: 工具执行出错: {str(e)}\n"
        else:
            observation = "\nObservation: 未能正确解析工具调用\n"

        # 将工具结果作为新消息返回
        # 注意：这里我们将结果追加到 AI 消息中，而不是创建新消息
        updated_content = last_message.content + observation
        updated_message = AIMessage(content=updated_content)

        # 替换最后一条消息
        new_messages = list(messages[:-1]) + [updated_message]

        return {"messages": new_messages}

    # ========== 构建图 ==========

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", call_model)  # LLM 推理节点
    workflow.add_node("tools", execute_tools)  # 工具执行节点（手动实现）

    # 设置入口点
    workflow.add_edge(START, "agent")

    # 添加条件边：agent 之后根据情况决定走向
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # 需要调用工具
            "end": END,  # 结束
        },
    )

    # 工具执行后回到 agent
    workflow.add_edge("tools", "agent")

    # 编译图（添加检查点以支持记忆）
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    print("✅ ReAct Agent 构建完成!")
    print("📝 使用 ReAct 提示词模式（兼容 HuggingFace 模型）")
    return app


# ==============================================================================
# 5. 运行 Agent
# ==============================================================================
def run_agent_stream(agent, query: str, thread_id: str = "default"):
    """流式运行 Agent，实时查看中间步骤"""
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"🗣️  用户: {query}")
    print(f"{'=' * 60}\n")

    # 构建输入
    input_data = {"messages": [HumanMessage(content=query)]}

    # 流式输出每个步骤
    for event in agent.stream(input_data, config=config, stream_mode="values"):
        # event 包含完整的状态
        messages = event.get("messages", [])
        if messages:
            last_msg = messages[-1]

            # 根据消息类型打印不同内容
            if isinstance(last_msg, HumanMessage):
                print(f"👤 用户: {last_msg.content}\n")
            elif isinstance(last_msg, AIMessage):
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🤖 Agent 决定调用工具: {last_msg.tool_calls[0]['name']}\n")
                elif last_msg.content:
                    print(f"🤖 Agent 回复: {last_msg.content}\n")

    print(f"{'=' * 60}\n")


def run_agent_sync(agent, query: str, thread_id: str = "default"):
    """同步运行 Agent，直接获取最终结果"""
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=query)]}

    # 使用 invoke 获取最终状态
    final_state = agent.invoke(input_data, config=config)

    # 提取最后的 AI 消息
    messages = final_state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "未获取到有效回复"


# ==============================================================================
# 🚀 主运行流程
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LangGraph ReAct Agent 启动")
    print("=" * 60)

    # 0. 准备测试数据
    print("\n📚 准备知识库...")
    if kb.vectorstore._collection.count() <= 1:
        demo_content = """
LangGraph 是 LangChain 生态系统的核心编排引擎。
相比 AgentExecutor，LangGraph 提供了循环图结构和更强的状态控制。
LangGraph 支持复杂的多步骤工作流、条件分支和状态持久化。
它允许开发者构建更灵活的 Agent 系统，适用于生产环境。
LangGraph 的核心优势在于其显式的图结构定义和细粒度的控制能力。
"""
        with open("demo_graph.txt", "w", encoding="utf-8") as f:
            f.write(demo_content)
        kb.add_document("demo_graph.txt")
        print("✅ 测试文档已添加")

    # 1. 加载模型
    llm = load_local_llm(MODEL_PATH)

    if not llm:
        print("❌ 模型加载失败，程序退出")
        exit(1)

    # 2. 构建 Agent
    agent_app = build_react_agent(llm)

    # 3. 测试查询
    test_queries = [
        "LangGraph 和 AgentExecutor 相比有什么优势？",
        "LangGraph 的核心优势是什么？",
    ]

    # 方式1: 流式输出 (推荐用于调试)
    print("\n" + "🔹" * 30)
    print("方式1: 流式输出")
    print("🔹" * 30)
    for query in test_queries[:1]:
        run_agent_stream(agent_app, query, thread_id="session_1")

    # 方式2: 同步获取结果 (推荐用于生产)
    print("\n" + "🔹" * 30)
    print("方式2: 同步调用")
    print("🔹" * 30)
    if len(test_queries) > 1:
        result = run_agent_sync(agent_app, test_queries[1], thread_id="session_2")
        print(f"🗣️  用户: {test_queries[1]}")
        print(f"🤖 Agent 回复:\n{result}\n")

    print("\n✨ 测试完成!")
