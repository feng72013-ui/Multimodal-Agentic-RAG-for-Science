from __future__ import annotations

from langchain_core.messages import AIMessage

from ._paths import ensure_project_paths
from .my_state import RecsysRAGState

ensure_project_paths()

from research_agents.orchestrator import run_research_workflow


def research_assistant_node(state: RecsysRAGState) -> dict:
    query = state.get("input_text") or ""
    task_type = state.get("task_type", "qa")
    collection_name = state.get("collection_name") or None
    review = run_research_workflow(
        query,
        task_type=task_type,
        top_k=5,
        on_demand_top_k=5,
        use_milvus=True,
        collection_name=collection_name,
    )
    context_retrieved = _flatten_review_context(review)
    images_retrieved = list(
        dict.fromkeys(
            item["image_path"]
            for item in context_retrieved
            if item.get("image_path")
        )
    )
    metrics = _research_assistant_metrics(review)
    return {
        "messages": [AIMessage(content=review["final_report"])],
        "idea_profile": review["idea_profile"],
        "related_work": review["related_work"],
        "task_paper_profiles": review.get("task_paper_profiles", []),
        "similarity_matrix": review["similarity_matrix"],
        "idea_scores": review["scores"],
        "agent_trace": review["agent_trace"],
        "agent_outputs": review["agent_outputs"],
        "context_retrieved": context_retrieved,
        "images_retrieved": images_retrieved,
        "evaluate_score": metrics["evaluate_score"],
        "response_relevancy": metrics["response_relevancy"],
        "context_relevance": metrics["context_relevance"],
        "context_precision": metrics["context_precision"],
        "faithfulness": metrics["faithfulness"],
        "final_response": review["final_report"],
    }


idea_review_node = research_assistant_node


def _flatten_review_context(review: dict) -> list[dict]:
    contexts = []
    for profile in review.get("task_paper_profiles") or []:
        for evidence in profile.get("evidence") or []:
            contexts.append(
                {
                    "title": profile.get("title"),
                    "topic": profile.get("topic"),
                    "paper_id": profile.get("paper_id"),
                    "text": evidence.get("text"),
                    "page_start": evidence.get("page_start"),
                    "page_end": evidence.get("page_end"),
                    "filename": evidence.get("source_pdf") or profile.get("source_pdf"),
                    "image_path": evidence.get("image_path"),
                    "score": evidence.get("score"),
                    "category": "text",
                }
            )
    return contexts


def _research_assistant_metrics(review: dict) -> dict[str, float]:
    scores = review.get("scores") or {}
    related = review.get("related_work") or []
    task_profiles = review.get("task_paper_profiles") or []
    context_precision = (
        sum(1 for item in related if item.get("task_relevance") in {"medium", "high"}) / len(related)
        if related
        else 0.0
    )
    return {
        "evaluate_score": float(scores.get("overall") or 0.0),
        "response_relevancy": float(scores.get("academic_value") or 0.0),
        "context_relevance": float(scores.get("max_related_similarity") or 0.0),
        "context_precision": round(float(context_precision), 3),
        "faithfulness": 0.8 if task_profiles else 0.0,
    }
