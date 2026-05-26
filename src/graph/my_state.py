from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class RecsysRAGState(TypedDict, total=False):
    """State shared by the recommendation-paper RAG workflow."""

    messages: Annotated[list[AnyMessage], add_messages]

    input_type: Literal["has_text", "only_image", "image_with_text"]
    input_image: str
    input_text: str
    user: str
    knowledge_base_id: str
    collection_name: str
    task_type_hint: Literal["qa", "idea_review", "literature_summary", "paper_compare", "research_plan"]

    task_type: Literal["direct_response", "qa", "idea_review", "literature_summary", "paper_compare", "research_plan"]
    task_confidence: float
    task_reason: str
    task_signals: List[str]

    idea_profile: Dict[str, Any]
    related_work: List[Dict[str, Any]]
    task_paper_profiles: List[Dict[str, Any]]
    similarity_matrix: List[Dict[str, Any]]
    idea_scores: Dict[str, Any]
    agent_trace: List[Dict[str, Any]]
    agent_outputs: Dict[str, Any]

    context_retrieved: List[Dict[str, Any]]
    images_retrieved: List[Dict[str, Any]]
    needs_retrieval: bool

    evaluation_score: float
    evaluate_score: float
    response_relevancy: float
    context_relevance: float
    context_precision: float
    faithfulness: float

    final_response: str
    human_answer: str


class InvalidInputError(Exception):
    """Raised when the workflow receives an unsupported input message."""

    def __init__(self, message: str, error_code: int = 400):
        self.message = message
        self.error_code = error_code
        super().__init__(message)
