from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from graph.my_state import InvalidInputError

from .schemas import ChatRequest, ResearchRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("RECSYS_TEXT_DEVICE", "cpu")


TASKS = [
    {
        "task_type": "qa",
        "label": "论文问答",
        "description": "围绕推荐系统论文、方法、实验、图表做证据化回答。",
    },
    {
        "task_type": "literature_summary",
        "label": "文献总结",
        "description": "按研究问题、方法、数据、指标、结果和局限组织总结。",
    },
    {
        "task_type": "paper_compare",
        "label": "论文/方法对比",
        "description": "按任务、模型、数据、指标、结论和适用场景对比。",
    },
    {
        "task_type": "idea_review",
        "label": "Idea 评审",
        "description": "检索相关工作，分析创新空间、相似点、风险和实验设计。",
    },
    {
        "task_type": "research_plan",
        "label": "研究计划",
        "description": "生成技术路线、baseline、指标、消融实验和阶段产出。",
    },
]


async def run_research_api(request: ResearchRequest) -> dict[str, Any]:
    from research_agents.orchestrator import run_research_workflow

    result = run_research_workflow(
        request.query,
        task_type=request.task_type,
        top_k=request.top_k,
        on_demand_top_k=request.on_demand_top_k,
        use_llm_reader=request.use_llm_reader,
        use_milvus=request.use_milvus,
    )
    return normalize_research_result(result)


async def run_chat_api(request: ChatRequest) -> dict[str, Any]:
    from graph.workflow import execute_graph, graph
    from .knowledge_bases import find_collection_for_kb

    session_id = request.session_id or str(uuid.uuid4())
    graph_config = {
        "configurable": {
            "user_name": request.user_name,
            "thread_id": session_id,
            "knowledge_base_id": request.knowledge_base_id,
            "collection_name": find_collection_for_kb(request.knowledge_base_id),
            "task_type_hint": request.task_type_hint,
        }
    }
    user_input = build_graph_input(request)
    answer = await execute_graph(user_input, graph_config)
    state_snapshot = graph.get_state(graph_config)
    state_values = state_snapshot.values or {}
    return {
        "session_id": session_id,
        "answer": answer,
        "requires_approval": bool(state_snapshot.next),
        "task_type": state_values.get("task_type"),
        "state": extract_state_for_api(state_values),
    }


async def stream_chat_api(request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    from graph.workflow import _build_human_message, _save_final_answer, graph, update_state
    from .knowledge_bases import find_collection_for_kb

    session_id = request.session_id or str(uuid.uuid4())
    graph_config = {
        "configurable": {
            "user_name": request.user_name,
            "thread_id": session_id,
            "knowledge_base_id": request.knowledge_base_id,
            "collection_name": find_collection_for_kb(request.knowledge_base_id),
            "task_type_hint": request.task_type_hint,
        }
    }
    yield stream_event("meta", session_id=session_id)

    current_state = graph.get_state(graph_config)
    if current_state.next:
        user_input = build_graph_input(request)
        yield stream_event("progress", title="收到人工审批", detail=approval_detail(user_input))
        update_state(user_input, graph_config)
        async for update in graph.astream(None, graph_config, stream_mode="updates"):
            for node_name, payload in update.items():
                yield progress_event(node_name, payload)
        final_state = graph.get_state(graph_config)
        answer = final_answer_from_state(final_state.values or {})
        if answer and final_state.values.get("task_type") != "direct_response":
            await _save_final_answer(final_state.values.get("messages") or [], final_state.values)
        async for event in stream_answer(answer, final_state.values or {}, bool(final_state.next), session_id):
            yield event
        return

    user_input = build_graph_input(request)
    yield stream_event("progress", title="解析用户输入", detail=input_detail(request))
    message = _build_human_message(user_input)
    async for update in graph.astream({"messages": [message]}, graph_config, stream_mode="updates"):
        for node_name, payload in update.items():
            yield progress_event(node_name, payload)

    final_state = graph.get_state(graph_config)
    state_values = final_state.values or {}
    answer = final_answer_from_state(state_values)
    if final_state.next:
        answer = with_approval_prompt(answer)
        yield stream_event(
            "approval_required",
            title="等待人工审批",
            detail="当前回答已生成，系统评估后需要用户确认。",
        )
    elif answer and state_values.get("task_type") != "direct_response":
        await _save_final_answer(state_values.get("messages") or [], state_values)

    async for event in stream_answer(answer, state_values, bool(final_state.next), session_id):
        yield event


def normalize_research_result(result: dict[str, Any]) -> dict[str, Any]:
    agent_outputs = result.get("agent_outputs") or {}
    planner = agent_outputs.get("research_planner") or {}
    raw = dict(result)
    raw.pop("agent_outputs", None)
    return {
        "task_type": result.get("task_type", ""),
        "status": result.get("status", ""),
        "retrieval_source": result.get("retrieval_source", ""),
        "final_report": result.get("final_report", ""),
        "related_work": result.get("related_work") or [],
        "task_paper_profiles": result.get("task_paper_profiles") or [],
        "scores": result.get("scores") or {},
        "agent_trace": result.get("agent_trace") or [],
        "agent_outputs": agent_outputs,
        "gaps": result.get("gaps") or [],
        "innovation_suggestions": result.get("innovation_suggestions") or [],
        "experiment_suggestions": result.get("experiment_suggestions") or [],
        "risk_assessment": result.get("risk_assessment") or [],
        "next_steps": planner.get("next_steps") or [],
        "raw": raw,
    }


def build_graph_input(request: ChatRequest) -> str:
    text = (request.message or "").strip()
    image = request.image_data_url or request.image_path
    if not text and not image:
        raise InvalidInputError("message、image_path 或 image_data_url 至少需要提供一个。")
    if text and image:
        return f"{text}&{image}"
    return text or str(image)


def stream_event(event: str, **payload: Any) -> dict[str, Any]:
    return {"event": event, **payload}


def progress_event(node_name: str, payload: Any) -> dict[str, Any]:
    title = NODE_PROGRESS_TITLES.get(node_name, node_name)
    detail = summarize_node_update(node_name, payload)
    return stream_event("progress", node=node_name, title=title, detail=detail)


NODE_PROGRESS_TITLES = {
    "process_input": "解析输入",
    "classify_intent": "识别任务类型",
    "research_assistant_node": "运行科研助手工作流",
    "first_chatbot": "准备回答或历史检索",
    "search_context": "检索用户历史上下文",
    "retriever_node": "检索论文知识库",
    "second_chatbot": "基于历史上下文生成回答",
    "third_chatbot": "基于知识库证据生成回答",
    "evaluate_node": "评估回答质量",
    "human_approval": "等待人工审批",
    "fourth_chatbot": "生成互联网兜底回答",
    "web_search_node": "调用互联网搜索工具",
}


def summarize_node_update(node_name: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return "节点已完成。"
    if node_name == "classify_intent":
        task_type = payload.get("task_type")
        confidence = payload.get("task_confidence")
        if task_type:
            return f"识别为 {task_type}，置信度 {confidence if confidence is not None else '未知'}。"
    if node_name == "research_assistant_node":
        related = payload.get("related_work") or []
        trace = payload.get("agent_trace") or []
        return f"已召回 {len(related)} 篇相关工作，记录 {len(trace)} 个 Agent 步骤。"
    if node_name == "retriever_node":
        contexts = payload.get("context_retrieved") or []
        images = payload.get("images_retrieved") or []
        return f"知识库返回 {len(contexts)} 个文本片段，{len(images)} 个图片/表格资源。"
    if node_name == "evaluate_node":
        score = payload.get("evaluate_score")
        faithfulness = payload.get("faithfulness")
        return f"RAGAS 深度评估完成：主分数 {score if score is not None else '未知'}，忠实度 {faithfulness if faithfulness is not None else '未知'}。"
    if node_name == "web_search_node":
        return "互联网搜索工具已返回补充信息。"
    if node_name == "fourth_chatbot":
        return "已根据兜底搜索结果生成补充回答。"
    if node_name == "human_approval":
        return "工作流已暂停，等待用户接受或拒绝当前答案。"
    return "节点已完成。"


def final_answer_from_state(state_values: dict[str, Any]) -> str:
    messages = state_values.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        return str(messages[-1].content)
    return str(state_values.get("final_response") or "")


def with_approval_prompt(answer: str) -> str:
    approval_prompt = (
        "\n\n---\n\n"
        "系统已完成自我评估，当前回答需要人工审批。\n"
        "如果认可当前输出，请选择“接受”；否则请选择“拒绝”，系统将启用互联网搜索兜底。"
    )
    return f"{answer or ''}{approval_prompt}"


async def stream_answer(
    answer: str,
    state_values: dict[str, Any],
    requires_approval: bool,
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    for delta in split_stream_text(answer):
        yield stream_event("answer_delta", delta=delta)
        await asyncio.sleep(0)
    yield stream_event(
        "final",
        session_id=session_id,
        answer=answer,
        requires_approval=requires_approval,
        task_type=state_values.get("task_type"),
        state=extract_state_for_api(state_values),
    )


def split_stream_text(text: str, size: int = 360) -> list[str]:
    if not text:
        return []
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        newline = text.rfind("\n", start, end)
        if newline > start + 80:
            end = newline + 1
        pieces.append(text[start:end])
        start = end
    return pieces


def input_detail(request: ChatRequest) -> str:
    parts = []
    if request.message:
        parts.append("文本")
    if request.image_path or request.image_data_url:
        parts.append("图片")
    return f"已收到{' + '.join(parts) or '输入'}，准备进入工作流。"


def approval_detail(user_input: str) -> str:
    normalized = user_input.strip().lower()
    if normalized == "approve":
        return "用户接受当前知识库回答，准备返回已生成报告。"
    return "用户拒绝当前知识库回答，准备启用互联网搜索兜底。"


def extract_state_for_api(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_type": state.get("input_type"),
        "input_text": state.get("input_text"),
        "user": state.get("user"),
        "task_confidence": state.get("task_confidence"),
        "task_reason": state.get("task_reason"),
        "task_signals": state.get("task_signals") or [],
        "related_work": state.get("related_work") or [],
        "task_paper_profiles": state.get("task_paper_profiles") or [],
        "context_retrieved": state.get("context_retrieved") or [],
        "images_retrieved": state.get("images_retrieved") or [],
        "evaluate_score": state.get("evaluate_score"),
        "response_relevancy": state.get("response_relevancy"),
        "context_relevance": state.get("context_relevance"),
        "context_precision": state.get("context_precision"),
        "faithfulness": state.get("faithfulness"),
        "idea_scores": state.get("idea_scores") or {},
        "agent_trace": state.get("agent_trace") or [],
        "agent_outputs": state.get("agent_outputs") or {},
        "final_response": state.get("final_response"),
        "messages": extract_messages(state.get("messages") or []),
    }


def extract_messages(messages: list[Any]) -> list[dict[str, Any]]:
    extracted = []
    for message in messages:
        role = "message"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        extracted.append(
            {
                "role": role,
                "content": getattr(message, "content", message),
            }
        )
    return extracted
