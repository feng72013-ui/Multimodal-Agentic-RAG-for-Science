from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .my_state import RecsysRAGState


TaskType = Literal["direct_response", "qa", "idea_review", "literature_summary", "paper_compare", "research_plan"]


@dataclass(frozen=True)
class IntentResult:
    task_type: TaskType
    confidence: float
    reason: str
    signals: list[str]


_TASK_KEYWORDS: dict[TaskType, tuple[str, ...]] = {
    "direct_response": (
        "你是谁",
        "你能做什么",
        "能为我做什么",
        "你的功能",
        "你有什么功能",
        "你会什么",
    ),
    "idea_review": (
        "idea",
        "想法",
        "创意",
        "创新点",
        "创新性",
        "可行性",
        "学术价值",
        "扩展方向",
        "研究空白",
        "有没有人做过",
        "是否已有",
        "novelty",
        "research gap",
        "feasible",
        "feasibility",
    ),
    "literature_summary": (
        "总结",
        "概括",
        "精读",
        "阅读这篇",
        "读这篇",
        "阅读文献",
        "阅读论文",
        "文献阅读",
        "文献思路",
        "代码是否开源",
        "是否开源",
        "论文摘要",
        "结构化摘要",
        "研究问题",
        "summarize",
        "summary",
        "paper reading",
    ),
    "paper_compare": (
        "对比",
        "比较",
        "区别",
        "异同",
        "差异",
        "优缺点",
        "哪篇",
        "哪个方法",
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference",
    ),
    "research_plan": (
        "研究计划",
        "research plan",
        "实验设计",
        "实验方案",
        "技术路线",
        "路线图",
        "开题",
        "方案",
        "规划",
        "研究项目",
        "proposal",
        "plan",
        "roadmap",
        "experiment design",
        "怎么开展",
        "下一步怎么做",
        "如何推进",
    ),
    "qa": (
        "什么是",
        "解释",
        "介绍",
        "如何理解",
        "有哪些论文",
        "找出",
        "请问",
        "what is",
        "explain",
        "find",
        "which papers",
    ),
}

_TASK_PRIORITY: tuple[TaskType, ...] = (
    "direct_response",
    "idea_review",
    "research_plan",
    "paper_compare",
    "literature_summary",
    "qa",
)


def classify_task(text: str | None, input_type: str | None = None) -> IntentResult:
    """Classify the research assistant task with deterministic, testable rules."""
    if input_type == "only_image":
        return IntentResult(
            task_type="qa",
            confidence=0.65,
            reason="仅图片输入默认进入图表/论文问答检索路径。",
            signals=["only_image"],
        )

    normalized = _normalize(text or "")
    if not normalized:
        return IntentResult(
            task_type="direct_response",
            confidence=0.5,
            reason="没有可分类的文本，回退到直接回应。",
            signals=[],
        )

    direct_result = _classify_direct_response(normalized)
    if direct_result is not None:
        return direct_result

    matches = {
        task_type: _matched_keywords(normalized, keywords)
        for task_type, keywords in _TASK_KEYWORDS.items()
    }
    scores = {
        task_type: len(signals) + _pattern_bonus(normalized, task_type)
        for task_type, signals in matches.items()
    }

    best_task = max(_TASK_PRIORITY, key=lambda task_type: (scores[task_type], -_TASK_PRIORITY.index(task_type)))
    if scores[best_task] <= 0:
        best_task = "qa"

    signals = matches[best_task]
    confidence = _confidence(scores[best_task], best_task)
    reason = _reason(best_task, signals, scores[best_task])
    return IntentResult(
        task_type=best_task,
        confidence=confidence,
        reason=reason,
        signals=signals,
    )


def classify_intent_node(state: RecsysRAGState) -> dict:
    result = classify_task(state.get("input_text"), state.get("input_type"))
    if result.task_type == "direct_response":
        return {
            "task_type": result.task_type,
            "task_confidence": result.confidence,
            "task_reason": result.reason,
            "task_signals": result.signals,
        }

    task_type_hint = state.get("task_type_hint")
    if task_type_hint and state.get("input_type") != "only_image":
        return {
            "task_type": task_type_hint,
            "task_confidence": 0.9,
            "task_reason": f"使用前端工作区指定的任务类型：{task_type_hint}。",
            "task_signals": ["workspace_task_hint"],
        }

    return {
        "task_type": result.task_type,
        "task_confidence": result.confidence,
        "task_reason": result.reason,
        "task_signals": result.signals,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matched_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword.lower() in text]


def _pattern_bonus(text: str, task_type: TaskType) -> int:
    if task_type == "direct_response" and re.fullmatch(
        r"(你好|您好|hello|hi|hey|你是谁|你能做什么|你有什么功能|你会什么|help|帮助)[。？！!?.\s]*",
        text,
    ):
        return 3
    if task_type == "literature_summary" and re.search(r"(这篇|该篇|某篇|paper|论文).*(总结|摘要|精读|summarize)", text):
        return 2
    if task_type == "paper_compare" and re.search(r"(.+)(和|与|vs|versus)(.+)(对比|比较|区别|差异|compare)", text):
        return 2
    if task_type == "idea_review" and re.search(r"(idea|想法|创意).*(评估|可行|创新|已有|novelty)", text):
        return 2
    if task_type == "research_plan" and re.search(r"(计划|路线|方案|项目|proposal|roadmap|实验).*(设计|推进|开展|规划|plan)", text):
        return 2
    return 0


def _confidence(score: int, task_type: TaskType) -> float:
    if task_type == "direct_response":
        return min(0.95, 0.65 + score * 0.1)
    if task_type == "qa" and score <= 0:
        return 0.55
    return min(0.95, 0.6 + score * 0.1)


def _reason(task_type: TaskType, signals: list[str], score: int) -> str:
    if task_type == "direct_response":
        signal_text = "、".join(signals[:5]) if signals else "闲聊/项目能力说明"
        return f"命中直接回应信号：{signal_text}；无需检索。"
    if not signals and task_type == "qa":
        return "未命中特定科研任务信号，回退到普通知识库问答。"
    signal_text = "、".join(signals[:5]) if signals else "规则模式"
    return f"命中 {task_type} 任务信号：{signal_text}；规则分数={score}。"


def _classify_direct_response(text: str) -> IntentResult | None:
    if re.fullmatch(r"(你好|您好|hello|hi|hey)[。？！!?.\s]*", text):
        return IntentResult(
            task_type="direct_response",
            confidence=0.95,
            reason="用户只是问候，直接回应即可。",
            signals=["greeting"],
        )
    if re.search(r"(你是谁|你能做什么|能为我做什么|你有什么功能|你的功能|你会什么|这个项目.*做什么)", text):
        return IntentResult(
            task_type="direct_response",
            confidence=0.9,
            reason="用户询问助手身份或项目能力，直接说明系统能力，不检索论文库。",
            signals=["assistant_identity_or_capability"],
        )
    return None
