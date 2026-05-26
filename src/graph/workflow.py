from __future__ import annotations

import asyncio
import os
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore

from ._paths import ensure_project_paths
from .all_route import (
    route_after_first_chatbot,
    route_after_intent,
    route_evaluate_node,
    route_human_approval_node,
    route_human_node,
    route_llm_or_retriever,
)
from .evaluate_node import evaluate_answer
from .idea_review_node import research_assistant_node
from .intent_node import classify_intent_node
from .my_state import InvalidInputError, RecsysRAGState
from .print_messages import pretty_print_messages
from .save_context import get_milvus_writer
from .search_node import SearchContextToolNode, retriever_node
from .tools import my_search, search_context

ensure_project_paths()

from project.recommendate_project.myllm import multiModal_llm
from vector_ingest_pipeline.utils import image_to_data_url


tools = [search_context]
web_tools = [my_search]


def process_input(state: RecsysRAGState, config):
    user_name = config["configurable"].get("user_name", "ZS")
    knowledge_base_id = config["configurable"].get("knowledge_base_id") or ""
    collection_name = config["configurable"].get("collection_name") or ""
    task_type_hint = config["configurable"].get("task_type_hint") or ""
    messages = state.get("messages", [])
    if not messages:
        raise InvalidInputError("用户输入的消息为空。")

    last_message = messages[-1]
    input_type = "has_text"
    text_content = None
    image_url = None

    if not isinstance(last_message, HumanMessage):
        raise InvalidInputError(f"用户输入的消息错误！原始输入：{last_message}")

    content = last_message.content
    if isinstance(content, str):
        text_content = content
    else:
        for item in content:
            if item.get("type") == "text":
                text_content = item.get("text")
            elif item.get("type") == "image_url":
                image_data = item.get("image_url", {})
                if isinstance(image_data, dict):
                    image_url = image_data.get("url")
                elif isinstance(image_data, str):
                    image_url = image_data

    if text_content and image_url:
        input_type = "image_with_text"
    elif not text_content and image_url:
        input_type = "only_image"

    return {
        "input_type": input_type,
        "user": user_name,
        "knowledge_base_id": knowledge_base_id,
        "collection_name": collection_name,
        "task_type_hint": task_type_hint,
        "input_text": text_content,
        "input_image": image_url,
    }


def first_chatbot(state: RecsysRAGState):
    task_type = state.get("task_type", "qa")
    if task_type == "direct_response":
        system_message = SystemMessage(
            content=(
                "你是本项目的推荐系统论文知识库助手，不要自称通用 ChatGLM，也不要说自己不能访问数据库或互联网。"
                "当用户问候、询问你是谁或询问这个项目中你能做什么时，直接用中文简短回答。"
                "你应说明：你可以在本项目中帮助检索和解读推荐系统论文知识库，做论文问答、文献总结、论文对比、"
                "idea 创新性/可行性评估、研究计划和实验方案设计，也可以检索相关图表/图片；"
                "知识库不足或用户拒绝当前答案时，工作流可以使用互联网搜索工具做兜底。"
            )
        )
        return {"messages": [multiModal_llm.invoke([system_message, *state["messages"]])]}

    llm_with_tools = multiModal_llm.bind_tools(tools)
    system_message = SystemMessage(
        content=(
            "你是本项目的推荐系统论文知识库助手，不要自称通用 ChatGLM，也不要说自己不能访问数据库或互联网。"
            "当前用户输入是实质性科研/论文问题。你可以先调用 search_context 检索用户历史上下文；"
            "如果历史上下文不足，后续工作流会继续检索 Milvus 推荐系统论文知识库。"
            "本项目还配置了互联网搜索兜底：当知识库答案未通过评估或被用户拒绝时，会调用 my_search 补充公开资料。"
            "不要凭空编造论文结论。"
            f"当前识别到的任务类型是 {task_type}。"
        )
    )
    return {"messages": [llm_with_tools.invoke([system_message, *state["messages"]])]}


def second_chatbot(state: RecsysRAGState):
    return {"messages": [multiModal_llm.invoke(state["messages"])]}


def third_chatbot(state: RecsysRAGState):
    context_retrieved = state.get("context_retrieved", [])
    images = state.get("images_retrieved", [])
    input_text = state.get("input_text")
    input_image = state.get("input_image")

    context = _format_context(context_retrieved)
    task_type = state.get("task_type", "qa")
    collection_name = state.get("collection_name") or "默认推荐系统论文知识库"
    task_prompt = _task_prompt_addendum(task_type)
    system_prompt = f"""
你是推荐系统论文知识库助手。请只基于下方检索到的论文片段回答用户问题。

规则：
1. 如果上下文中没有足够依据，请明确说明“知识库中没有找到足够依据”，不要编造。
2. 回答推荐系统概念、论文方法、实验结论时，优先引用片段中的论文标题、主题、页码或文件名。
3. 如果检索到相关图片或表格，请优先回答“哪些图片最相似”，并说明标题、页码、相似度/得分和可见差异；不要只给论文泛泛摘要。
4. 使用清晰的 Markdown，中文回答，必要时用编号列表。
5. 当前任务类型：{task_type}。请遵循对应任务要求。
6. 当前选择的知识库/collection：{collection_name}。

任务要求：
{task_prompt}

检索上下文：
{context}

相关图片/表格检索结果：
{_format_images(images) if images else "无"}
"""
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})
    if not user_content:
        user_content.append({"type": "text", "text": "请根据检索结果回答。"})

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    return {"messages": [multiModal_llm.invoke(messages)]}


def human_approval(state: RecsysRAGState):
    return {}


def fourth_chatbot(state: RecsysRAGState):
    llm_with_tools = multiModal_llm.bind_tools(web_tools)
    input_text = state.get("input_text") or "请基于公开资料补充回答推荐系统相关问题。"
    task_type = state.get("task_type", "qa")
    system_message = SystemMessage(
        content=(
            "你是推荐系统研究助手。当前知识库回答未通过评估或被用户拒绝。"
            "请优先调用互联网搜索工具 my_search，再结合搜索结果生成谨慎回答，并说明信息来自互联网兜底搜索。"
            f"当前任务类型是 {task_type}，回答结构需要贴合该任务。"
        )
    )
    message = HumanMessage(content=[{"type": "text", "text": input_text}])
    return {"messages": [llm_with_tools.invoke([system_message, message])]}


def _format_context(context_retrieved: list[dict]) -> str:
    if not context_retrieved:
        return "没有检索到相关的论文上下文。"

    pieces = []
    for index, hit in enumerate(context_retrieved, start=1):
        source = ", ".join(
            str(value)
            for value in [
                hit.get("title"),
                hit.get("topic"),
                hit.get("filename"),
                _format_pages(hit),
            ]
            if value
        )
        pieces.append(
            f"片段 {index}：\n"
            f"{hit.get('text') or ''}\n"
            f"来源：{source or '未知'}\n"
            f"类别：{hit.get('category') or '未知'}；得分：{hit.get('score')}"
        )
    return "\n\n".join(pieces)


def _format_images(images_retrieved: list[dict]) -> str:
    if not images_retrieved:
        return "无"

    lines = []
    for index, image in enumerate(images_retrieved, start=1):
        source = ", ".join(
            str(value)
            for value in [
                image.get("title"),
                image.get("filename"),
                _format_pages(image),
            ]
            if value
        )
        lines.append(
            f"[{index}] {source or '未知来源'}；类别：{image.get('category') or '未知'}；"
            f"得分：{image.get('score')}；路径：{image.get('image_path') or image.get('path')}"
        )
    return "\n".join(lines)


def _format_pages(hit: dict) -> str:
    page_start = hit.get("page_start")
    page_end = hit.get("page_end")
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def _task_prompt_addendum(task_type: str) -> str:
    prompts = {
        "qa": (
            "围绕用户问题直接回答；给出关键论文、方法、实验或图表证据；"
            "如果只能找到部分依据，请明确边界。"
        ),
        "idea_review": (
            "按“相关已有工作、与用户 idea 的相似点、潜在差异、可扩展创新点、"
            "可行性风险、下一步检索建议”组织回答。不要给出无证据的学术价值判断。"
        ),
        "literature_summary": (
            "按“研究问题、核心方法、实验设置、主要结果、结论与局限、可复用启发”"
            "组织结构化文献摘要。"
        ),
        "paper_compare": (
            "按“比较维度、各论文/方法共同点、关键差异、适用场景、证据来源”组织回答；"
            "没有足够片段时说明缺少哪一侧证据。"
        ),
        "research_plan": (
            "按“研究目标、已有依据、可行路线、实验设计、风险与备选方案、阶段性产出”"
            "组织研究计划；每个建议都应能追溯到检索证据或明确标注为推断。"
        ),
    }
    return prompts.get(task_type, prompts["qa"])


checkpointer = InMemorySaver()
store = InMemoryStore()
builder = StateGraph(RecsysRAGState)

builder.add_node("process_input", process_input)
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("research_assistant_node", research_assistant_node)
builder.add_node("first_chatbot", first_chatbot)
builder.add_node("search_context", SearchContextToolNode(tools=tools))
builder.add_node("retriever_node", retriever_node)
builder.add_node("second_chatbot", second_chatbot)
builder.add_node("third_chatbot", third_chatbot)
builder.add_node("evaluate_node", evaluate_answer)
builder.add_node("human_approval", human_approval)
builder.add_node("fourth_chatbot", fourth_chatbot)
builder.add_node("web_search_node", ToolNode(tools=web_tools))

builder.add_edge(START, "process_input")
builder.add_edge("process_input", "classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_after_intent,
    {
        "retriever_node": "retriever_node",
        "research_assistant_node": "research_assistant_node",
        "first_chatbot": "first_chatbot",
    },
)
builder.add_edge("research_assistant_node", "evaluate_node")
builder.add_conditional_edges(
    "first_chatbot",
    route_after_first_chatbot,
    {"search_context": "search_context", "retriever_node": "retriever_node", END: END},
)
builder.add_conditional_edges(
    "search_context",
    route_llm_or_retriever,
    {"retriever_node": "retriever_node", "second_chatbot": "second_chatbot"},
)
builder.add_edge("retriever_node", "third_chatbot")
builder.add_conditional_edges("third_chatbot", route_evaluate_node, {"evaluate_node": "evaluate_node", END: END})
builder.add_conditional_edges("evaluate_node", route_human_node, {"human_approval": "human_approval", END: END})
builder.add_conditional_edges("human_approval", route_human_approval_node, {"fourth_chatbot": "fourth_chatbot", END: END})
builder.add_conditional_edges("fourth_chatbot", tools_condition, {"tools": "web_search_node", END: END})
builder.add_edge("web_search_node", "fourth_chatbot")

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=["human_approval"],
)

session_id = str(uuid.uuid4())
config = {
    "configurable": {
        "user_name": os.getenv("RECSYS_USER_NAME", "ZS"),
        "thread_id": session_id,
    }
}


def update_state(user_answer: str, graph_config: dict) -> None:
    graph.update_state(
        config=graph_config,
        values={"human_answer": "approve" if user_answer == "approve" else "rejected"},
    )


async def execute_graph(user_input: str, graph_config: dict | None = None) -> str:
    graph_config = graph_config or config
    result = ""
    current_state = graph.get_state(graph_config)
    if current_state.next:
        update_state(user_input, graph_config)
        async for chunk in graph.astream(None, graph_config, stream_mode="values"):
            pretty_print_messages(chunk, last_message=True)
        current_state = graph.get_state(graph_config)
        messages = current_state.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            result = str(messages[-1].content)
            if current_state.values.get("task_type") != "direct_response":
                await _save_final_answer(messages, current_state.values)
        return result

    message = _build_human_message(user_input)
    # 获取当前消息历史，避免重复添加
    existing_messages = current_state.values.get("messages", [])
    async for chunk in graph.astream({"messages": [message]}, graph_config, stream_mode="values"):
        pretty_print_messages(chunk, last_message=True)

    current_state = graph.get_state(graph_config)
    if current_state.next:
        pending_answer = current_state.values.get("final_response")
        if not pending_answer:
            messages = current_state.values.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                pending_answer = str(messages[-1].content)
        approval_prompt = (
            "\n\n---\n\n"
            "系统已完成自我评估，当前回答需要人工审批。\n"
            "如果认可当前输出，请选择“接受”；否则请选择“拒绝”，系统将启用互联网搜索兜底。"
        )
        return f"{pending_answer or ''}{approval_prompt}"

    messages = current_state.values.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        result = str(messages[-1].content)
        if current_state.values.get("task_type") != "direct_response":
            await _save_final_answer(messages, current_state.values)
    return result


def _build_human_message(user_input: str) -> HumanMessage:
    text = None
    image_url = None
    if "&" in user_input:
        text, image = user_input.split("&", 1)
        if image and os.path.isfile(image):
            image_url = image_to_data_url(image)
        elif image.startswith("data:"):
            image_url = image
    elif os.path.isfile(user_input):
        image_url = image_to_data_url(user_input)
    else:
        text = user_input

    content = []
    if text:
        content.append({"type": "text", "text": text})
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return HumanMessage(content=content)


async def _save_final_answer(messages: list, state_values: dict) -> None:
    user_question = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                user_question = message.content
            else:
                for item in message.content:
                    if item.get("type") == "text":
                        user_question = item.get("text")
                        break
            break

    answer = messages[-1].content
    await get_milvus_writer().async_insert(
        context_text=str(answer),
        user=state_values.get("user", "ZS"),
        message_type="AIMessage",
        question=user_question,
        evaluation_score=state_values.get("evaluate_score"),
        enable_dedup=True,
    )


async def main() -> None:
    while True:
        user_input = input("用户输入(文本和图片用&隔开，exit退出)：")
        if user_input.lower() in {"exit", "quit", "退出"}:
            break
        response = await execute_graph(user_input)
        if response:
            print("AI:", response)


if __name__ == "__main__":
    asyncio.run(main())
