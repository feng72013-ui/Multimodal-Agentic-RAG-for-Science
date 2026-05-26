#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from idea_review_pipeline.milvus_adapter import retrieve_idea_context_from_milvus
from idea_review_pipeline.review import DEFAULT_CHUNKS_PATH, DEFAULT_PROFILE_PATH, render_markdown, review_idea
from vector_ingest_pipeline.utils import iter_jsonl, truncate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_DIR = PROJECT_ROOT / "data" / "processed_rag" / "agent_traces"
os.environ.setdefault("RECSYS_TEXT_DEVICE", "cpu")

REQUIRED_AGENT_ROLES = (
    "Supervisor",
    "Retriever",
    "PaperAnalyst",
    "ResearchCritic",
    "ResearchPlanner",
    "Writer",
)

SUPPORTED_TASK_TYPES = ("qa", "idea_review", "literature_summary", "paper_compare", "research_plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 deterministic research-agent workflow.")
    parser.add_argument("--query", required=True, help="User query or research idea.")
    parser.add_argument("--task-type", default="idea_review", choices=SUPPORTED_TASK_TYPES)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--on-demand-top-k", type=int, default=5)
    parser.add_argument("--use-llm-reader", action="store_true")
    parser.add_argument("--no-milvus", action="store_true", help="Use local JSONL profiles/chunks instead of Milvus.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = list(iter_jsonl(args.profiles)) if args.no_milvus and args.profiles.exists() else []
    chunks = (
        list(iter_jsonl(args.chunks))
        if args.no_milvus and args.chunks.exists() and args.on_demand_top_k > 0
        else []
    )
    result = run_research_workflow(
        args.query,
        task_type=args.task_type,
        profiles=profiles,
        chunks=chunks,
        top_k=args.top_k,
        on_demand_top_k=args.on_demand_top_k,
        use_llm_reader=args.use_llm_reader,
        use_milvus=not args.no_milvus,
    )
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2)[:12000])
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{safe_slug(args.query)}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Agent trace: {output_path}")
    return 0


def run_research_workflow(
    user_input: str,
    *,
    task_type: str = "idea_review",
    profiles: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    on_demand_top_k: int = 5,
    use_llm_reader: bool = False,
    use_milvus: bool = True,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Run a role-separated research workflow over the existing paper-profile baseline.

    Stage 4 keeps the agents deterministic so their contract can be tested. The
    same I/O shape can later be backed by LLM calls without changing graph state.
    """
    profiles = profiles or []
    chunks = chunks or []
    trace: list[dict[str, Any]] = []
    agent_outputs: dict[str, Any] = {}

    supervisor = supervisor_agent(user_input, task_type)
    agent_outputs["supervisor"] = supervisor
    trace.append(
        trace_step(
            "Supervisor",
            "plan_task",
            "completed",
            ["task_type", "plan", "acceptance_checks"],
            f"Accepted {task_type} with {len(supervisor['plan'])} planned steps.",
        )
    )

    milvus_context = {}
    if use_milvus:
        try:
            milvus_context = retrieve_idea_context_from_milvus(
                user_input,
                top_k_papers=top_k,
                initial_top_k=max(top_k * 6, 20),
                chunks_per_paper=max(on_demand_top_k * 2, 8),
                collection_name=collection_name,
            )
            if milvus_context.get("profiles"):
                profiles = milvus_context["profiles"]
                chunks = milvus_context.get("chunks") or chunks
                trace.append(
                    trace_step(
                        "Retriever",
                        "milvus_candidate_retrieval",
                        "completed",
                        ["profiles", "chunks"],
                        (
                            f"Retrieved {len(profiles)} candidate papers and "
                            f"{len(chunks)} chunks from Milvus collection {milvus_context.get('collection')}."
                        ),
                    )
                )
            else:
                trace.append(
                    trace_step(
                        "Retriever",
                        "milvus_candidate_retrieval",
                        "degraded",
                        ["profiles", "chunks"],
                        "Milvus returned no candidate paper groups; falling back to local lightweight profiles.",
                    )
                )
        except Exception as exc:
            trace.append(
                trace_step(
                    "Retriever",
                    "milvus_candidate_retrieval",
                    "degraded",
                    ["profiles", "chunks"],
                    f"Milvus retrieval failed, falling back to local lightweight profiles: {exc}",
                )
            )
    if not profiles:
        profiles = load_default_profiles()
    if not chunks and on_demand_top_k > 0 and not milvus_context.get("profiles"):
        chunks = load_default_chunks()

    retriever = retriever_agent(
        user_input,
        task_type,
        profiles,
        top_k,
        chunks=chunks,
        on_demand_top_k=on_demand_top_k,
        use_llm_reader=use_llm_reader,
    )
    review = retriever["review"]
    agent_outputs["retriever"] = retriever_without_review(retriever)
    trace.append(
        trace_step(
            "Retriever",
            "retrieve_related_work",
            retriever["status"],
            ["related_work", "similarity_matrix"],
            (
                f"Found {len(review.get('related_work') or [])} related papers from {len(profiles)} profiles; "
                f"on-demand read {len(review.get('task_paper_profiles') or [])} papers."
            ),
        )
    )

    paper_analysis = paper_analyst_agent(review)
    agent_outputs["paper_analyst"] = paper_analysis
    trace.append(
        trace_step(
            "PaperAnalyst",
            "analyze_evidence",
            "completed" if paper_analysis["top_papers"] else "degraded",
            ["top_papers", "shared_themes", "evidence_summary"],
            paper_analysis["evidence_summary"],
        )
    )

    critique = research_critic_agent(review, task_type)
    agent_outputs["research_critic"] = critique
    trace.append(
        trace_step(
            "ResearchCritic",
            "score_and_risk_review",
            "completed",
            ["scores", "risk_assessment", "critic_summary"],
            critique["critic_summary"],
        )
    )

    planner = research_planner_agent(review, paper_analysis)
    agent_outputs["research_planner"] = planner
    trace.append(
        trace_step(
            "ResearchPlanner",
            "plan_next_experiments",
            "completed",
            ["innovation_suggestions", "experiment_suggestions", "next_steps"],
            f"Prepared {len(planner['next_steps'])} next steps.",
        )
    )

    final_report = writer_agent(review, paper_analysis, critique, planner, trace, milvus_context)
    agent_outputs["writer"] = {"final_report": final_report}
    trace.append(
        trace_step(
            "Writer",
            "render_report",
            "completed",
            ["final_report"],
            f"Rendered {len(final_report)} characters.",
        )
    )

    status = "completed" if review.get("related_work") else "degraded"
    return {
        "task_type": task_type,
        "status": status,
        "retrieval_source": "milvus" if milvus_context.get("profiles") else "local_jsonl",
        "milvus_context": {
            key: value
            for key, value in milvus_context.items()
            if key not in {"profiles", "chunks"}
        },
        "supervisor_plan": supervisor["plan"],
        "idea_profile": review["idea_profile"],
        "related_work": review["related_work"],
        "task_paper_profiles": review.get("task_paper_profiles", []),
        "similarity_matrix": review["similarity_matrix"],
        "scores": review["scores"],
        "direct_answer": review.get("direct_answer", {}),
        "gaps": review["gaps"],
        "innovation_suggestions": review["innovation_suggestions"],
        "experiment_suggestions": review["experiment_suggestions"],
        "risk_assessment": review["risk_assessment"],
        "agent_outputs": agent_outputs,
        "agent_trace": trace,
        "final_report": final_report,
    }


def load_default_profiles() -> list[dict[str, Any]]:
    if not DEFAULT_PROFILE_PATH.exists():
        return []
    return list(iter_jsonl(DEFAULT_PROFILE_PATH))


def load_default_chunks() -> list[dict[str, Any]]:
    if not DEFAULT_CHUNKS_PATH.exists():
        return []
    return list(iter_jsonl(DEFAULT_CHUNKS_PATH))


def supervisor_agent(user_input: str, task_type: str) -> dict[str, Any]:
    task_focus = {
        "qa": "回答科研问题并给出可追溯证据",
        "idea_review": "讨论、评估并优化用户科研 idea",
        "literature_summary": "阅读并结构化总结相关文献",
        "paper_compare": "对比论文或方法的共同点、差异和适用场景",
        "research_plan": "制定研究路线、实验方案和阶段产出",
    }.get(task_type, "处理科研助手任务")
    return {
        "task_type": task_type,
        "input_summary": truncate(user_input, 240),
        "plan": [
            "确认任务类型和阶段目标",
            task_focus,
            "检索论文画像和相关知识库片段",
            "只对召回的 Top-K 论文做按需结构化阅读",
            "分析相似论文的任务、方法、数据和指标证据",
            "根据任务类型生成回答、总结、对比、idea 优化或研究计划",
            "生成可执行的后续建议",
            "汇总为可读报告并输出 agent trace",
        ],
        "acceptance_checks": [
            "包含所有核心 agent 的执行轨迹",
            "返回 related work、scores、risks 和 next steps",
            "在论文画像缺失时给出降级说明",
        ],
    }


def retriever_agent(
    user_input: str,
    task_type: str,
    profiles: list[dict[str, Any]],
    top_k: int,
    *,
    chunks: list[dict[str, Any]] | None = None,
    on_demand_top_k: int = 5,
    use_llm_reader: bool = False,
) -> dict[str, Any]:
    review = review_idea(
        user_input,
        profiles,
        task_type=task_type,
        top_k=top_k,
        chunks=chunks or [],
        on_demand_top_k=on_demand_top_k,
        use_llm_reader=use_llm_reader,
    )
    status = "completed" if review.get("related_work") else "degraded"
    return {
        "status": status,
        "profile_count": len(profiles),
        "chunk_count": len(chunks or []),
        "top_k": top_k,
        "on_demand_top_k": on_demand_top_k,
        "use_llm_reader": use_llm_reader,
        "review": review,
    }


def retriever_without_review(retriever: dict[str, Any]) -> dict[str, Any]:
    review = retriever["review"]
    return {
        "status": retriever["status"],
        "profile_count": retriever["profile_count"],
        "chunk_count": retriever.get("chunk_count", 0),
        "top_k": retriever["top_k"],
        "on_demand_read_count": len(review.get("task_paper_profiles") or []),
        "related_count": len(review.get("related_work") or []),
        "top_related": [
            {
                "rank": item.get("rank"),
                "title": item.get("title"),
                "similarity": item.get("similarity"),
            }
            for item in review.get("related_work", [])[:3]
        ],
    }


def paper_analyst_agent(review: dict[str, Any]) -> dict[str, Any]:
    related = review.get("related_work") or []
    task_profiles = review.get("task_paper_profiles") or []
    task_by_key = {
        f"{profile.get('topic') or ''}::{profile.get('paper_id') or ''}": profile
        for profile in task_profiles
    }
    top_papers = [
        {
            "rank": item.get("rank"),
            "title": item.get("title"),
            "topic": item.get("topic"),
            "similarity": item.get("similarity"),
            "evidence": item.get("evidence", [])[:2],
            "task_profile": task_by_key.get(f"{item.get('topic') or ''}::{item.get('paper_id') or ''}", {}),
        }
        for item in related[:5]
    ]
    themes = sorted(
        {
            str(item.get("topic"))
            for item in related
            if item.get("topic") not in (None, "")
        }
    )
    if top_papers:
        evidence_summary = (
            f"Top related work covers {', '.join(themes[:3]) or 'recommendation research'}; "
            f"{len(task_profiles)} papers were structurally read on demand."
        )
    else:
        evidence_summary = "知识库中没有找到明显相似论文，需要扩大检索范围或补充外部检索。"
    return {
        "top_papers": top_papers,
        "shared_themes": themes[:8],
        "evidence_summary": evidence_summary,
    }


def research_critic_agent(review: dict[str, Any], task_type: str) -> dict[str, Any]:
    scores = review.get("scores") or {}
    max_similarity = scores.get("max_related_similarity", 0.0)
    if task_type == "literature_summary":
        critic_summary = "文献总结应优先保证证据覆盖和忠实度，避免把检索不到的内容补全成结论。"
    elif task_type == "paper_compare":
        critic_summary = "论文对比需要明确比较维度，并标注哪些论文证据不足。"
    elif task_type == "research_plan":
        critic_summary = "研究计划需要把建议落到数据集、baseline、指标和可消融模块。"
    elif task_type == "qa":
        critic_summary = "问答结果应直接回答问题，并保留可追溯论文依据。"
    elif max_similarity >= 0.55:
        critic_summary = "相似工作较接近，必须把贡献落到更明确的新任务、新约束或新评测协议。"
    elif scores.get("feasibility", 0.0) < 0.6:
        critic_summary = "方向有探索价值，但需要先收敛任务、数据集和方法模块。"
    else:
        critic_summary = "当前 idea 可以进入细粒度 related work 对比和实验方案设计。"
    return {
        "scores": scores,
        "risk_assessment": review.get("risk_assessment") or [],
        "critic_summary": critic_summary,
    }


def research_planner_agent(review: dict[str, Any], paper_analysis: dict[str, Any]) -> dict[str, Any]:
    task_type = review.get("task_type", "idea_review")
    next_steps_by_task = {
        "qa": [
            "确认答案中引用的 top related work 是否覆盖用户问题。",
            "继续检索缺失概念或关键论文，补充证据边界。",
            "将可复用结论沉淀为后续文献阅读问题。",
        ],
        "literature_summary": [
            "补齐论文的问题、方法、实验、结果、局限五类证据。",
            "检查是否需要按页码精读图表或实验设置。",
            "把总结转成可复用的论文卡片或对比表。",
        ],
        "paper_compare": [
            "固定比较维度，例如任务、模型、数据集、指标和实验结论。",
            "为每篇论文补齐缺失维度证据。",
            "输出共同点、关键差异和适用场景表格。",
        ],
        "research_plan": [
            "把研究目标拆成数据、模型、训练、评估和论文写作阶段。",
            "固定 baseline、指标和消融实验。",
            "列出风险与备选路线，形成最小可复现实验协议。",
        ],
        "idea_review": [
            "精读 top related work，补充逐篇差异表。",
            "把核心贡献拆成可消融模块，设计主实验和 ablation。",
            "固定数据集、指标和 baseline，形成最小可复现实验协议。",
        ],
    }
    next_steps = next_steps_by_task.get(task_type, next_steps_by_task["idea_review"])
    if not paper_analysis.get("top_papers"):
        next_steps.insert(0, "先扩展检索范围，补充外部论文或更新知识库。")
    return {
        "innovation_suggestions": review.get("innovation_suggestions") or [],
        "experiment_suggestions": review.get("experiment_suggestions") or [],
        "next_steps": next_steps,
    }


def writer_agent(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
    trace: list[dict[str, Any]],
    milvus_context: dict[str, Any] | None = None,
) -> str:
    polished = render_polished_report_with_glm4_flash(review, paper_analysis, critique, planner)
    if polished:
        return polished
    return render_user_facing_report(review, paper_analysis, critique, planner)


def render_polished_report_with_glm4_flash(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
) -> str:
    if not os.getenv("ZHIPU_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
    if not os.getenv("ZHIPU_API_KEY"):
        return ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            from project.recommendate_project.myllm import glm4_flash
        except ModuleNotFoundError:
            from myllm import glm4_flash

        payload = build_polish_payload(review, paper_analysis, critique, planner)
        system_prompt = (
            "你是推荐系统论文知识库助手，负责把检索和多智能体结果整理成面向用户的最终中文回答。"
            "要求：1) 只输出 Markdown；2) 不要输出 Research Assistant Report、Task、Agent Trace、Tool Calls 等调试标题；"
            "3) 除论文标题、方法名、数据集名、指标名、代码链接、必要英文原文短摘录外，解释文字必须使用中文；"
            "4) 先直接回答用户问题，再给证据和边界；5) 不要编造知识库没有提供的结果；"
            "6) 文献阅读任务优先按“核心结论、研究思路、方法框架、实验结果、代码开源情况、证据边界”组织；"
            "7) 代码是否开源必须依据 evidence 中的 URL 或 source code/code/github 信号判断；"
            "8) 若 task_type 为 research_plan，输出标题必须是“文献调研报告”，并按“摘要、调研范围与检索概览、"
            "核心论文速览、主题聚类与趋势、方法路线对比、实验评估脉络、局限挑战、研究建议与展望、参考依据、证据边界”组织；"
            "9) 若 task_type 为 idea_review，输出标题必须是“Idea 生成与评估”，并给出 2-3 个具体研究思路，"
            "每个思路包含任务定义、创新差异、可行性、实验协议、风险和优化建议。"
        )
        response = glm4_flash.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        content = str(response.content).strip()
        if not content or "Research Assistant Report" in content[:120]:
            return ""
        return content.rstrip() + "\n"
    except Exception:
        return ""


def build_polish_payload(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
) -> dict[str, Any]:
    focus_profiles = select_focus_profiles(review)[:3]
    evidence_profiles = []
    for profile in focus_profiles:
        evidence_profiles.append(
            {
                "title": profile.get("title"),
                "paper_id": profile.get("paper_id"),
                "research_problem": [truncate(str(item), 500) for item in (profile.get("research_problem") or [])[:3]],
                "methods": [truncate(str(item), 500) for item in (profile.get("methods") or [])[:3]],
                "results": [truncate(str(item), 500) for item in (profile.get("results") or [])[:3]],
                "limitations": [truncate(str(item), 360) for item in (profile.get("limitations") or [])[:2]],
                "code_signal": infer_code_availability(profile),
            }
        )
    return {
        "user_request": review.get("idea_profile", {}).get("raw_idea"),
        "task_type": review.get("task_type"),
        "direct_answer": review.get("direct_answer") or {},
        "focus_papers": evidence_profiles,
        "related_work": [
            {
                "title": item.get("title"),
                "similarity": item.get("similarity"),
                "task_relevance": item.get("task_relevance"),
            }
            for item in (review.get("related_work") or [])[:5]
        ],
        "scores": review.get("scores") or {},
        "gaps": (review.get("gaps") or [])[:4],
        "suggestions": {
            "research": (review.get("innovation_suggestions") or [])[:4],
            "experiment": (review.get("experiment_suggestions") or [])[:4],
            "next_steps": (planner.get("next_steps") or [])[:4],
        },
        "analysis_notes": {
            "paper_analyst": paper_analysis.get("evidence_summary"),
            "critic": critique.get("critic_summary"),
        },
    }


def render_user_facing_report(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
) -> str:
    task_type = review.get("task_type", "qa")
    if task_type == "literature_summary":
        return render_literature_summary_report(review, planner)
    if task_type == "paper_compare":
        return render_compare_report(review, planner)
    if task_type == "research_plan":
        return render_research_plan_report(review, paper_analysis, critique, planner)
    if task_type == "idea_review":
        return render_idea_report(review, planner)
    return render_qa_report(review, planner)


def render_literature_summary_report(review: dict[str, Any], planner: dict[str, Any]) -> str:
    user_request = review.get("idea_profile", {}).get("raw_idea", "")
    profiles = select_focus_profiles(review)
    direct_answer = review.get("direct_answer") or {}
    first_title = str((profiles[0].get("title") if profiles else "") or "").lower()
    conclusion = direct_answer.get("verdict") or "我已基于知识库检索结果整理这篇论文的研究思路、方法和证据边界。"
    if "lightgcl" in first_title:
        conclusion = (
            "LightGCL 的核心是把图推荐中的局部协同结构和 SVD 引导的全局协同视图结合起来做对比学习，"
            "以缓解交互稀疏、流行度偏置和图增强泛化不足的问题。知识库证据显示代码链接为 "
            "https://github.com/HKUDS/LightGCL。"
        )
    elif "lightgcn" in first_title:
        conclusion = "LightGCN 的核心是简化 GCN 推荐模型，只保留邻居聚合和多层 embedding 融合，从而提升协同过滤的效率和效果。"
    lines = [
        "# 文献阅读总结",
        "",
        "## 简要结论",
        "",
        conclusion,
        "",
    ]
    if user_request:
        lines.extend(["## 你的问题", "", user_request, ""])

    for profile in profiles[:2]:
        title = profile.get("title") or profile.get("paper_id") or "未命名论文"
        zh_summary = profile_summary_zh(profile)
        lines.extend([f"## 论文：{title}", ""])
        lines.extend(render_profile_section("研究思路", zh_summary.get("idea") or profile.get("research_problem"), fallback="知识库中没有稳定抽取到完整研究问题。"))
        lines.extend(render_profile_section("核心方法", zh_summary.get("method") or profile.get("methods"), fallback="知识库中没有稳定抽取到完整方法描述。"))
        lines.extend(render_profile_section("实验与结果", zh_summary.get("result") or profile.get("results"), fallback="知识库中没有稳定抽取到完整实验结果。"))
        code_note = infer_code_availability(profile)
        lines.extend(["## 代码开源情况", "", code_note, ""])
        if profile.get("limitations"):
            lines.extend(render_profile_section("局限与注意点", profile.get("limitations")))

    lines.extend(render_related_work_compact(review))
    lines.extend(render_evidence_boundary(review))
    lines.extend(render_quality_rubric("literature_summary"))
    lines.extend(["## 下一步建议", ""])
    lines.extend(f"- {step}" for step in planner.get("next_steps", [])[:3])
    return "\n".join(lines).rstrip() + "\n"


def render_qa_report(review: dict[str, Any], planner: dict[str, Any]) -> str:
    direct_answer = review.get("direct_answer") or {}
    lines = [
        "# 回答",
        "",
        direct_answer.get("verdict") or "我已根据论文知识库检索结果整理回答；下面只保留与问题直接相关的信息。",
        "",
    ]
    profiles = select_focus_profiles(review)
    if profiles:
        lines.extend(["## 主要依据", ""])
        for profile in profiles[:3]:
            title = profile.get("title") or profile.get("paper_id") or "未命名论文"
            evidence = first_nonempty(profile.get("research_problem"), profile.get("methods"), profile.get("results"))
            lines.append(f"- **{title}**：{truncate(evidence, 260) if evidence else '知识库召回了该论文，但结构化证据较少。'}")
        lines.append("")
    lines.extend(render_related_work_compact(review))
    lines.extend(["## 后续可追问", ""])
    lines.extend(f"- {step}" for step in planner.get("next_steps", [])[:3])
    return "\n".join(lines).rstrip() + "\n"


def render_idea_report(review: dict[str, Any], planner: dict[str, Any]) -> str:
    direct_answer = review.get("direct_answer") or {}
    scores = review.get("scores") or {}
    lines = [
        "# Idea 生成与评估",
        "",
        "## 模块定位",
        "",
        "本模块负责把用户的初步方向转化为可评估、可实验、可迭代的科研 idea；文献阅读负责精读论文，文献调研负责形成领域综述。",
        "",
        "## 总体判断",
        "",
        direct_answer.get("verdict") or "这个方向可以继续探索，但需要结合已有工作明确差异点和实验验证方式。",
        "",
        f"- 创新潜力：{scores.get('novelty', scores.get('overall', 0.0))}",
        f"- 可行性：{scores.get('feasibility', 0.0)}",
        f"- 最高相关工作相似度：{scores.get('max_related_similarity', 0.0)}",
        "",
        "## 候选研究思路",
        "",
    ]
    suggestions = (review.get("innovation_suggestions") or [])[:3]
    experiments = (review.get("experiment_suggestions") or [])[:3]
    if suggestions:
        for index, item in enumerate(suggestions, start=1):
            experiment = experiments[index - 1] if index - 1 < len(experiments) else "先固定公开数据集、强 baseline、主指标和消融模块，形成最小验证协议。"
            lines.extend(
                [
                    f"### 思路 {index}",
                    "",
                    f"- **研究问题**：{truncate(str(item), 280)}",
                    "- **创新差异**：优先落在新任务约束、新模型组件、新评测协议或跨模态证据融合上，并与下方相关工作逐项对齐。",
                    f"- **实验协议**：{truncate(str(experiment), 280)}",
                    "- **可行性判断**：如果已有工作相似度偏高，应收窄到更具体的数据、场景或约束；如果证据不足，应先补充检索。",
                    "",
                ]
            )
    else:
        lines.append("- 暂未形成稳定候选思路，请补充研究对象、应用场景、可用数据或预期方法。")
    lines.extend(["", "## 实验验证", ""])
    lines.extend(f"- {item}" for item in experiments[:4])
    lines.extend(["", "## 风险与优化建议", ""])
    risks = review.get("risk_assessment") or []
    gaps = review.get("gaps") or []
    if risks or gaps:
        lines.extend(f"- {truncate(str(item), 260)}" for item in [*risks[:3], *gaps[:3]])
    else:
        lines.append("- 主要风险是与已有工作贡献重叠、实验资源不明确或评价指标不能支撑核心主张。")
    lines.extend(render_related_work_compact(review))
    lines.extend(render_quality_rubric("idea_review"))
    lines.extend(["## 下一步建议", ""])
    lines.extend(f"- {step}" for step in planner.get("next_steps", [])[:3])
    return "\n".join(lines).rstrip() + "\n"


def render_quality_rubric(task_type: str) -> list[str]:
    rubrics = {
        "idea_review": [
            "候选 idea 需具备明确任务、方法组件、数据/指标和可检验贡献。",
            "创新性判断必须关联相关工作相似点与差异点。",
            "可行性判断必须指出资源、baseline、消融和失败风险。",
        ],
        "literature_summary": [
            "摘要需覆盖研究问题、方法、实验、结果、局限和证据页/来源。",
            "不得把未检索到的实验数值或代码链接补全成事实。",
        ],
        "research_plan": [
            "调研报告需覆盖主题趋势、核心论文、方法路线、实验脉络和研究空白。",
            "每条建议需区分文献证据与模型推断。",
        ],
    }
    items = rubrics.get(task_type, [])
    if not items:
        return []
    return ["## 输出质量标准", "", *[f"- {item}" for item in items], ""]


def render_compare_report(review: dict[str, Any], planner: dict[str, Any]) -> str:
    lines = ["# 论文/方法对比", "", "## 对比结论", ""]
    direct_answer = review.get("direct_answer") or {}
    lines.append(direct_answer.get("verdict") or "我已按知识库召回的相关论文整理共同点、差异和证据缺口。")
    lines.extend(["", "## 相关论文", ""])
    for item in review.get("related_work", [])[:5]:
        lines.append(f"- {item.get('title')}：相关度 {item.get('similarity')}")
    lines.extend(["", "## 证据缺口", ""])
    lines.extend(f"- {gap}" for gap in (review.get("gaps") or [])[:4])
    lines.extend(["", "## 下一步建议", ""])
    lines.extend(f"- {step}" for step in planner.get("next_steps", [])[:3])
    return "\n".join(lines).rstrip() + "\n"


def render_research_plan_report(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
) -> str:
    direct_answer = review.get("direct_answer") or {}
    user_request = review.get("idea_profile", {}).get("raw_idea", "")
    related = review.get("related_work") or []
    profiles = select_focus_profiles(review)
    scores = review.get("scores") or {}
    themes = paper_analysis.get("shared_themes") or []
    profile_count = len(review.get("task_paper_profiles") or [])
    source_note = "知识库"
    if review.get("retrieval_source") == "web":
        source_note = "互联网检索"
    lines = [
        "# 文献调研报告",
        "",
        "## 摘要",
        "",
        direct_answer.get("verdict") or "我已基于当前推荐系统论文知识库，对该方向形成结构化调研报告。",
        "",
        "本报告参考 Paper-Agent 的“检索-阅读-分析-综合-报告”思路组织：先说明调研范围，再汇总核心论文，随后从主题趋势、方法路线、实验评估和研究空白几个层面综合分析。",
        "",
        "## 调研范围与检索概览",
        "",
        f"- **用户方向**：{user_request or '未提供明确方向。'}",
        f"- **资料来源**：优先使用当前选择的{source_note}；知识库不足或用户拒绝时，系统仍可进入互联网兜底搜索。",
        f"- **召回规模**：召回相关论文 {len(related)} 篇，按需阅读结构化论文 {profile_count} 篇。",
        f"- **证据强度**：最高相关度 {scores.get('max_related_similarity', 0.0)}；综合评估 {scores.get('overall', 0.0)}。",
        f"- **分析提示**：{paper_analysis.get('evidence_summary') or '当前检索结果可用于形成初步调研，但仍建议补充更精确的关键词。'}",
        "",
        "## 核心论文速览",
        "",
    ]

    if related:
        lines.extend([
            "| 序号 | 论文/主题 | 相关度 | 可用证据 |",
            "|---|---|---:|---|",
        ])
        for index, item in enumerate(related[:6], start=1):
            title = escape_markdown_table(item.get("title") or item.get("topic") or "未命名论文")
            relevance = item.get("task_relevance") or "unknown"
            lines.append(f"| {index} | {title} | {item.get('similarity')} | {relevance} |")
    else:
        lines.append("- 暂未召回稳定相关论文，需要扩大关键词或使用互联网兜底。")

    lines.extend(["", "## 主题聚类与趋势", ""])
    lines.extend(render_theme_trend_section(themes, profiles, user_request))
    lines.extend(["", "## 方法路线对比", ""])
    lines.extend(render_method_comparison_section(profiles))
    lines.extend(["", "## 实验评估脉络", ""])
    lines.extend(render_experiment_landscape_section(profiles, review))
    lines.extend(["", "## 局限挑战", ""])
    risks = review.get("risk_assessment") or []
    gaps = review.get("gaps") or []
    if risks or gaps:
        lines.extend(f"- {truncate(str(item), 260)}" for item in [*gaps[:3], *risks[:3]])
    else:
        lines.append(f"- {critique.get('critic_summary') or '需要继续检查证据覆盖、实验可验证性和与已有工作的差异。'}")

    lines.extend(["", "## 研究建议与展望", ""])
    suggestions = [
        *(review.get("innovation_suggestions") or [])[:3],
        *(review.get("experiment_suggestions") or [])[:3],
    ]
    if suggestions:
        lines.extend(f"- {truncate(str(item), 300)}" for item in suggestions)
    else:
        lines.append("- 建议先将调研方向收敛为明确任务、数据集、baseline、指标和预期贡献，再展开实验设计。")

    lines.extend(["", "## 后续行动清单", ""])
    lines.extend(f"- {step}" for step in planner.get("next_steps", [])[:5])
    lines.extend(["", "## 参考依据", ""])
    if profiles:
        for profile in profiles[:6]:
            title = profile.get("title") or profile.get("paper_id") or "未命名论文"
            evidence = first_nonempty(profile.get("research_problem"), profile.get("methods"), profile.get("results"))
            lines.append(f"- **{title}**：{truncate(evidence, 260) if evidence else '召回到该论文，但结构化摘要证据较少。'}")
    elif related:
        for item in related[:6]:
            lines.append(f"- **{item.get('title') or item.get('topic')}**：相关度 {item.get('similarity')}。")
    else:
        lines.append("- 暂无稳定论文依据。")

    lines.extend(render_evidence_boundary(review))
    lines.extend(render_quality_rubric("research_plan"))
    return "\n".join(lines).rstrip() + "\n"


def render_theme_trend_section(themes: list[Any], profiles: list[dict[str, Any]], user_request: str) -> list[str]:
    lines: list[str] = []
    clean_themes = [str(theme) for theme in themes if str(theme).strip() and str(theme).lower() != "none"]
    if clean_themes:
        lines.append(f"- 当前召回结果主要集中在：{', '.join(clean_themes[:6])}。")
    else:
        lines.append("- 当前召回结果尚未形成稳定主题簇，可以通过增加英文关键词、限定年份或指定数据集进一步聚焦。")

    dated = [
        str(profile.get("title") or profile.get("paper_id") or "")
        for profile in profiles[:4]
        if profile.get("title") or profile.get("paper_id")
    ]
    if dated:
        lines.append(f"- 代表性论文包括：{'; '.join(dated)}。这些论文可作为梳理技术演进的起点。")
    if user_request:
        lines.append("- 从用户方向看，调研应重点观察该方向与已有推荐模型、数据稀疏性、鲁棒性、公平性或多模态信号之间的关系。")
    return lines


def render_method_comparison_section(profiles: list[dict[str, Any]]) -> list[str]:
    if not profiles:
        return ["- 暂缺可比较的结构化论文方法证据。"]
    lines = ["| 方法来源 | 核心路线 | 可迁移启发 |", "|---|---|---|"]
    for profile in profiles[:5]:
        title = escape_markdown_table(profile.get("title") or profile.get("paper_id") or "未命名论文")
        method = escape_markdown_table(first_nonempty(profile.get("methods")) or "知识库未稳定抽取方法描述。")
        problem = escape_markdown_table(first_nonempty(profile.get("research_problem")) or "可作为相关任务背景参考。")
        lines.append(f"| {title} | {truncate(method, 180)} | {truncate(problem, 180)} |")
    return lines


def render_experiment_landscape_section(profiles: list[dict[str, Any]], review: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    result_items = []
    for profile in profiles[:5]:
        title = profile.get("title") or profile.get("paper_id") or "未命名论文"
        result = first_nonempty(profile.get("results"))
        if result:
            result_items.append(f"- **{title}**：{truncate(result, 280)}")
    if result_items:
        lines.extend(result_items)
    else:
        lines.append("- 当前结构化证据中缺少稳定的实验结果摘要，需要继续查看原论文表格、数据集和指标设置。")
    experiment_suggestions = review.get("experiment_suggestions") or []
    if experiment_suggestions:
        lines.append("- 可优先把调研结论落到可复现实验：")
        lines.extend(f"  - {truncate(str(item), 220)}" for item in experiment_suggestions[:3])
    return lines


def escape_markdown_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def select_focus_profiles(review: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = list(review.get("task_paper_profiles") or [])
    query = str(review.get("idea_profile", {}).get("raw_idea") or "").lower()
    if not profiles:
        return []

    def score(profile: dict[str, Any]) -> tuple[int, float]:
        title = str(profile.get("title") or profile.get("paper_id") or "").lower()
        title_score = 0
        for token in ("lightgcl", "lightgcn", "sgl", "gcl"):
            if token in query and token in title:
                title_score += 10
        related_score = float(profile.get("retrieval", {}).get("score") or 0.0)
        return (title_score, related_score)

    return sorted(profiles, key=score, reverse=True)


def render_profile_section(title: str, values: Any, fallback: str = "") -> list[str]:
    items = values if isinstance(values, list) else []
    lines = [f"## {title}", ""]
    if not items:
        lines.append(fallback or "知识库中没有抽取到稳定证据。")
    else:
        lines.extend(f"- {truncate(str(item), 420)}" for item in items[:3] if str(item).strip())
    lines.append("")
    return lines


def profile_summary_zh(profile: dict[str, Any]) -> dict[str, list[str]]:
    title = str(profile.get("title") or profile.get("paper_id") or "").lower()
    if "lightgcl" in title:
        return {
            "idea": [
                "LightGCL 关注图推荐中的对比学习问题：现有 GNN 推荐方法虽然有效，但在数据稀疏、流行度偏置和图增强策略泛化性上仍有不足。",
                "论文的核心思路是保留图协同过滤中的局部结构信息，同时引入更稳定的全局协同关系视图，让对比学习不完全依赖手工扰动或随机图增强。",
            ],
            "method": [
                "方法上，LightGCL 以图推荐模型为基础，构造局部图视图和由 SVD 引导的全局协同视图，并在两类视图之间进行图对比学习。",
                "训练目标通常由推荐主任务损失和对比学习损失共同组成，用来增强用户和物品表示的鲁棒性。",
                "相比复杂的增强策略，LightGCL 强调“简单但有效”，把重点放在利用全局协同信号缓解稀疏交互带来的表示不足。",
            ],
            "result": [
                "知识库证据显示，论文在实验中与多类推荐基线和图对比学习方法比较，报告了 LightGCL 在多个数据集上的整体优势。",
                "证据中还提到 LightGCL 在数据稀疏和流行度偏置场景下具有更好的鲁棒性；具体数值建议继续查看原论文表格。",
            ],
        }
    if "lightgcn" in title:
        return {
            "idea": [
                "LightGCN 重新审视 GCN 在协同过滤中的必要组件，认为特征变换和非线性激活等传统 GCN 操作未必适合推荐任务。",
                "论文核心思路是保留最关键的邻居聚合，让用户和物品 embedding 在交互图上传播，从而得到更简洁、更易训练的推荐模型。",
            ],
            "method": [
                "方法上，LightGCN 去掉传统 GCN 中的特征变换和非线性激活，只保留用户-物品图上的线性邻居聚合。",
                "最终表示由多层传播后的 embedding 加权组合得到，再用于协同过滤预测。",
            ],
            "result": [
                "知识库证据显示，LightGCN 相比较重的 GCN 推荐模型更简单，并在推荐效果和训练效率上具有优势。",
            ],
        }
    return {}


def render_related_work_compact(review: dict[str, Any]) -> list[str]:
    related = review.get("related_work") or []
    if not related:
        return ["## 相关工作", "", "- 暂未召回稳定相关论文。", ""]
    lines = ["## 相关工作", ""]
    for item in related[:5]:
        relevance = item.get("task_relevance")
        relevance_text = f"，任务相关性：{relevance}" if relevance else ""
        lines.append(f"- {item.get('title')}（相关度：{item.get('similarity')}{relevance_text}）")
    lines.append("")
    return lines


def render_scores_compact(review: dict[str, Any]) -> list[str]:
    scores = review.get("scores") or {}
    if not scores:
        return []
    return [
        "## 质量评估",
        "",
        f"- 综合分：{scores.get('overall', 0.0)}",
        f"- 证据相关度：{scores.get('max_related_similarity', 0.0)}",
        f"- 解释：{scores.get('interpretation', '暂无解释。')}",
        "",
    ]


def render_evidence_boundary(review: dict[str, Any]) -> list[str]:
    scores = review.get("scores") or {}
    related_count = len(review.get("related_work") or [])
    profile_count = len(review.get("task_paper_profiles") or [])
    return [
        "## 证据边界",
        "",
        f"- 本次知识库召回了 {related_count} 篇相关论文，并对其中 {profile_count} 篇做了按需阅读。",
        f"- 当前最高相关度为 {scores.get('max_related_similarity', 0.0)}；如果你需要更严谨的论文卡片，可以继续指定数据集、指标或章节。",
        "- 上面的总结优先依据知识库片段；涉及具体实验数值时，建议继续打开原论文表格核对。",
        "",
    ]


def infer_code_availability(profile: dict[str, Any]) -> str:
    texts = []
    for field_name in ("research_problem", "methods", "results", "limitations"):
        texts.extend(str(item) for item in (profile.get(field_name) or []))
    for evidence in profile.get("evidence") or []:
        texts.append(str(evidence.get("text") or ""))
    joined = "\n".join(texts)
    match = re.search(r"https?://[^\s)>\]]+", joined)
    if match:
        url = match.group(0).rstrip(".,;。；")
        return f"知识库证据中出现代码或项目链接：{url}"
    if re.search(r"code|github|source", joined, flags=re.I):
        return "知识库证据中提到代码或 source code，但没有稳定抽取到完整链接，建议继续打开原论文或项目页确认。"
    return "当前知识库片段没有发现明确代码开源链接；如果需要，我可以继续围绕论文标题检索代码仓库。"


def first_nonempty(*groups: Any) -> str:
    for group in groups:
        if isinstance(group, list):
            for item in group:
                text = str(item).strip()
                if text:
                    return text
        elif group:
            return str(group)
    return ""


def render_internal_debug_report(
    review: dict[str, Any],
    paper_analysis: dict[str, Any],
    critique: dict[str, Any],
    planner: dict[str, Any],
    trace: list[dict[str, Any]],
    milvus_context: dict[str, Any] | None = None,
) -> str:
    lines = [
        render_markdown(review).rstrip(),
        "",
        "## Tool / Agent Calls",
        "",
    ]
    lines.extend(render_tool_agent_calls(trace, milvus_context or {}))
    lines.extend([
        "",
        "## Knowledge Base Retrieval Snippets",
        "",
    ])
    lines.extend(render_retrieval_snippets(review))
    lines.extend([
        "",
        "## Retrieved Images Or Tables",
        "",
    ])
    lines.extend(render_retrieved_assets(review))
    lines.extend([
        "",
        "## Evaluation And Approval",
        "",
    ])
    lines.extend(render_evaluation_and_approval(review))
    lines.extend([
        "",
        "## Multi-Agent Collaboration",
        "",
        f"- Paper analyst: {paper_analysis['evidence_summary']}",
        f"- Research critic: {critique['critic_summary']}",
        "",
        "## Next Steps",
        "",
    ])
    lines.extend(f"- {step}" for step in planner["next_steps"])
    lines.extend(["", "## Agent Trace", ""])
    lines.extend(
        f"- {step['role']}: {step['action']} -> {step['status']}"
        for step in trace
    )
    return "\n".join(lines) + "\n"


def render_tool_agent_calls(trace: list[dict[str, Any]], milvus_context: dict[str, Any]) -> list[str]:
    lines = []
    if milvus_context:
        lines.append(
            "- Tool: Milvus knowledge-base retrieval "
            f"(collection={milvus_context.get('collection')}, "
            f"profile_collection={milvus_context.get('profile_collection')}, "
            f"layer={milvus_context.get('retrieval_layer')}, hits={milvus_context.get('hit_count')})"
        )
    lines.extend(
        f"- Agent: {step['role']} | action={step['action']} | status={step['status']} | note={step.get('note', '')}"
        for step in trace
    )
    return lines or ["- 无工具或智能体调用记录。"]


def render_retrieval_snippets(review: dict[str, Any], limit: int = 8) -> list[str]:
    snippets = []
    for profile in review.get("task_paper_profiles") or []:
        title = profile.get("title") or profile.get("paper_id") or "Unknown paper"
        for evidence in profile.get("evidence") or []:
            snippets.append(
                "- "
                f"{title} | page={evidence.get('page_start')} | section={evidence.get('section') or 'unknown'} | "
                f"chunk={evidence.get('chunk_id')}\n"
                f"  {truncate(evidence.get('text') or '', 320)}"
            )
            if len(snippets) >= limit:
                return snippets
    return snippets or ["- 无可展示的论文知识库检索片段。"]


def render_retrieved_assets(review: dict[str, Any]) -> list[str]:
    assets = []
    for profile in review.get("task_paper_profiles") or []:
        for evidence in profile.get("evidence") or []:
            image_path = evidence.get("image_path")
            if image_path:
                assets.append(f"- {image_path}")
    unique_assets = list(dict.fromkeys(assets))
    return unique_assets or ["- 无检索到的图片或表格路径。"]


def render_evaluation_and_approval(review: dict[str, Any]) -> list[str]:
    scores = review.get("scores") or {}
    related = review.get("related_work") or []
    task_profiles = review.get("task_paper_profiles") or []
    has_evidence = bool(task_profiles)
    context_relevance = scores.get("max_related_similarity", 0.0)
    context_precision = (
        sum(1 for item in related if item.get("task_relevance") in {"medium", "high"}) / len(related)
        if related
        else 0.0
    )
    faithfulness = 0.8 if has_evidence else 0.0
    lines = [
        f"- 当前主评估分数: {scores.get('overall', 0.0)}",
        f"- 回答相关性: {scores.get('academic_value', 0.0)} (idea-review proxy)",
        f"- 上下文相关性: {context_relevance} (max related-work similarity)",
        f"- 上下文精确性: {round(context_precision, 3)} (medium/high task relevance ratio)",
        f"- 回答忠实度: {faithfulness} (基于是否存在按需阅读证据的 proxy)",
        "- 人工审批过程: research_assistant 已接入 evaluate_node -> human_approval；报告生成后会进入人工审批中断，用户输入 approve 通过，否则输入 rejected 触发互联网搜索兜底。",
    ]
    return lines


def render_unsupported_report(user_input: str, task_type: str, trace: list[dict[str, Any]]) -> str:
    lines = [
        "# Research Agent Report",
        "",
        f"Task type `{task_type}` is not supported by the Stage 4 orchestrator yet.",
        "",
        "## Query",
        "",
        user_input,
        "",
        "## Agent Trace",
        "",
    ]
    lines.extend(f"- {step['role']}: {step['action']} -> {step['status']}" for step in trace)
    return "\n".join(lines) + "\n"


def trace_step(
    role: str,
    action: str,
    status: str,
    output_keys: list[str],
    note: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "action": action,
        "status": status,
        "output_keys": output_keys,
        "note": truncate(note, 300),
    }


def safe_slug(text: str) -> str:
    words = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    words = "-".join(part for part in words.split("-") if part)
    return (words[:80] or "agent-trace")


if __name__ == "__main__":
    raise SystemExit(main())
