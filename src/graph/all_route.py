from __future__ import annotations

from langgraph.constants import END

from ._paths import ensure_project_paths
from .my_state import RecsysRAGState

ensure_project_paths()


def route_only_image(state: RecsysRAGState):
    if state.get("input_type") == "only_image":
        return "retriever_node"
    return "first_chatbot"


def route_after_intent(state: RecsysRAGState):
    if state.get("input_type") in {"only_image", "image_with_text"}:
        return "retriever_node"
    if state.get("task_type") == "direct_response":
        return "first_chatbot"
    if state.get("task_type") in {"qa", "idea_review", "literature_summary", "paper_compare", "research_plan"}:
        return "research_assistant_node"
    return "first_chatbot"


def route_after_first_chatbot(state: RecsysRAGState):
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No message found in input")

    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "search_context"
    if state.get("task_type") == "direct_response":
        return END
    return "retriever_node"


def route_llm_or_retriever(state: RecsysRAGState):
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No message found in input")

    tool_message = messages[-1]
    if not getattr(tool_message, "content", None) or tool_message.content == "没有找到相关的历史上下文信息。":
        return "retriever_node"
    return "second_chatbot"


def route_evaluate_node(state: RecsysRAGState):
    if state.get("input_type") in {"only_image", "image_with_text"}:
        return END
    if state.get("task_type") == "direct_response":
        return END
    return "evaluate_node"


def route_human_node(state: RecsysRAGState):
    if state.get("task_type") in {"qa", "idea_review", "literature_summary", "paper_compare", "research_plan"}:
        return "human_approval"
    evaluate_score = state.get("evaluate_score")
    if evaluate_score is None:
        return "human_approval"
    if evaluate_score >= 0.7:
        return END
    return "human_approval"


def route_human_approval_node(state: RecsysRAGState):
    if state.get("human_answer") == "approve":
        return END
    return "fourth_chatbot"
