from __future__ import annotations

from langchain_core.messages import AIMessage

from .my_state import RecsysRAGState


async def evaluate_answer(state: RecsysRAGState):
    """Evaluate answer quality with Ragas when available."""
    context_retrieved = state.get("context_retrieved", [])
    input_text = state.get("input_text")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if not isinstance(last_message, AIMessage):
        return _fallback_scores(state, 0.0)

    retrieved_contexts = [ctx.get("text", "") for ctx in context_retrieved if ctx.get("text")]
    if not input_text or not retrieved_contexts:
        return _fallback_scores(state, 0.0)

    try:
        from ragas import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextRelevance,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )

        from myllm import embedding, glm4_flash as llm
    except Exception:
        return _fallback_scores(state, 0.0)

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding)
    sample = SingleTurnSample(
        user_input=input_text,
        retrieved_contexts=retrieved_contexts,
        response=str(last_message.content),
    )

    results: dict[str, float | None] = {}
    try:
        scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        results["response_relevancy"] = float(await scorer.single_turn_ascore(sample))
    except Exception:
        results["response_relevancy"] = 0.5

    try:
        scorer = ContextRelevance(llm=evaluator_llm)
        results["context_relevance"] = float(await scorer.single_turn_ascore(sample))
    except Exception:
        results["context_relevance"] = None

    try:
        scorer = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
        results["context_precision"] = float(await scorer.single_turn_ascore(sample))
    except Exception:
        results["context_precision"] = None

    try:
        scorer = Faithfulness(llm=evaluator_llm)
        results["faithfulness"] = float(await scorer.single_turn_ascore(sample))
    except Exception:
        results["faithfulness"] = None

    evaluate_score = float(results["response_relevancy"] or 0.0)
    return {
        "evaluate_score": evaluate_score,
        "response_relevancy": evaluate_score,
        "context_relevance": results.get("context_relevance"),
        "context_precision": results.get("context_precision"),
        "faithfulness": results.get("faithfulness"),
    }


def _fallback_scores(state: RecsysRAGState, default: float) -> dict:
    existing = state.get("evaluate_score")
    if existing is None:
        existing = default
    return {
        "evaluate_score": float(existing),
        "response_relevancy": float(state.get("response_relevancy") or existing or default),
        "context_relevance": float(state.get("context_relevance") or 0.0),
        "context_precision": float(state.get("context_precision") or 0.0),
        "faithfulness": float(state.get("faithfulness") or 0.0),
    }
