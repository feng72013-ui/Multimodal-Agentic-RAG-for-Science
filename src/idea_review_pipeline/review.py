#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from idea_review_pipeline.on_demand_reader import build_task_paper_profiles
from vector_ingest_pipeline.utils import iter_jsonl, truncate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PROFILE_PATH = DATA_ROOT / "processed_rag" / "paper_profiles.jsonl"
DEFAULT_CHUNKS_PATH = DATA_ROOT / "processed_rag" / "chunks.jsonl"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_rag" / "idea_reviews"


TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "tasks": (
        "recommendation",
        "recommender",
        "next item",
        "sequential recommendation",
        "session",
        "multimodal recommendation",
        "fairness",
        "privacy",
        "watermark",
        "evaluation",
        "benchmark",
        "ranking",
        "retrieval",
        "explanation",
    ),
    "methods": (
        "rag",
        "retrieval augmented generation",
        "llm",
        "large language model",
        "hypergraph",
        "hypergraph neural network",
        "agent",
        "graph neural network",
        "gnn",
        "graph transformer",
        "contrastive learning",
        "causal",
        "alignment",
        "watermarking",
        "collaborative filtering",
        "transformer",
    ),
    "datasets": (
        "movielens",
        "amazon",
        "yelp",
        "gowalla",
        "lastfm",
        "steam",
        "netflix",
        "taobao",
    ),
    "metrics": (
        "recall",
        "ndcg",
        "hit rate",
        "mrr",
        "auc",
        "precision",
        "fairness",
        "robustness",
    ),
}

TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "推荐系统": ("recommendation", "recommender system"),
    "序列推荐": ("sequential recommendation", "next item recommendation"),
    "会话推荐": ("session recommendation",),
    "多模态推荐": ("multimodal recommendation",),
    "检索增强生成": ("rag", "retrieval augmented generation"),
    "检索增强": ("rag", "retrieval augmented generation"),
    "大模型": ("llm", "large language model", "foundation model"),
    "大语言模型": ("llm", "large language model"),
    "大预语言模型": ("llm", "large language model"),
    "语言模型": ("language model",),
    "超图": ("hypergraph", "hypergraph neural network"),
    "图神经网络": ("graph neural network", "gnn"),
    "知识图谱": ("knowledge graph",),
    "对比学习": ("contrastive learning",),
    "因果": ("causal",),
    "公平性": ("fairness",),
    "鲁棒性": ("robustness",),
    "可解释": ("explanation", "explainable recommendation"),
    "召回": ("recall",),
    "排序": ("ranking",),
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "about",
    "idea",
    "research",
    "paper",
    "system",
    "systems",
    "model",
    "models",
    "method",
    "methods",
    "请",
    "帮我",
    "一个",
    "这个",
    "是否",
    "如何",
    "研究",
    "系统",
    "方法",
    "模型",
}


@dataclass
class IdeaProfile:
    raw_idea: str
    normalized: str
    terms: dict[str, list[str]]
    tokens: list[str]
    query_variants: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a research idea against generated paper profiles.")
    parser.add_argument("--idea", required=True, help="Research idea text.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--on-demand-top-k", type=int, default=0)
    parser.add_argument("--use-llm-reader", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-max-chars",
        type=int,
        default=10000,
        help="Maximum characters to print in dry-run mode. Use 0 for full JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = list(iter_jsonl(args.profiles))
    chunks = list(iter_jsonl(args.chunks)) if args.chunks.exists() and args.on_demand_top_k > 0 else []
    review = review_idea(
        args.idea,
        profiles,
        top_k=args.top_k,
        chunks=chunks,
        on_demand_top_k=args.on_demand_top_k,
        use_llm_reader=args.use_llm_reader,
    )

    if args.dry_run:
        output = json.dumps(review, ensure_ascii=False, indent=2)
        if args.dry_run_max_chars > 0:
            output = output[: args.dry_run_max_chars]
        print(output)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.idea)
    json_path = args.output_json or args.output_dir / f"{slug}.json"
    md_path = args.output_md or args.output_dir / f"{slug}.md"
    json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(review), encoding="utf-8")
    print(f"Idea review JSON: {json_path}")
    print(f"Idea review Markdown: {md_path}")
    return 0


def review_idea(
    idea: str,
    profiles: list[dict[str, Any]],
    task_type: str = "idea_review",
    top_k: int = 5,
    *,
    chunks: list[dict[str, Any]] | None = None,
    on_demand_top_k: int = 0,
    use_llm_reader: bool = False,
) -> dict[str, Any]:
    idea_profile = analyze_idea(idea)
    ranked = rank_profiles(idea_profile, profiles, top_k=top_k)
    task_paper_profiles = build_task_paper_profiles(
        idea,
        ranked,
        chunks or [],
        top_k=on_demand_top_k,
        use_llm=use_llm_reader,
    )
    if task_paper_profiles:
        ranked = enrich_related_work_with_task_profiles(ranked, task_paper_profiles)
    scores = score_idea(idea_profile, ranked)
    return {
        "task_type": task_type,
        "idea_profile": {
            "raw_idea": idea_profile.raw_idea,
            "terms": idea_profile.terms,
            "query_variants": idea_profile.query_variants,
        },
        "related_work": ranked,
        "task_paper_profiles": task_paper_profiles,
        "similarity_matrix": build_similarity_matrix(idea_profile, ranked),
        "scores": scores,
        "direct_answer": generate_direct_answer(idea_profile, ranked, task_type),
        "gaps": infer_research_gaps(idea_profile, ranked),
        "innovation_suggestions": suggest_innovations(idea_profile, ranked, scores),
        "experiment_suggestions": suggest_experiments(idea_profile, ranked),
        "risk_assessment": assess_risks(idea_profile, ranked, scores),
    }


def enrich_related_work_with_task_profiles(
    related_work: list[dict[str, Any]],
    task_paper_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_by_key = {
        f"{profile.get('topic') or ''}::{profile.get('paper_id') or ''}": profile
        for profile in task_paper_profiles
    }
    enriched = []
    for item in related_work:
        key = f"{item.get('topic') or ''}::{item.get('paper_id') or ''}"
        task_profile = profile_by_key.get(key)
        if not task_profile:
            enriched.append(item)
            continue
        evidence = []
        for field_name in ("research_problem", "methods", "results", "limitations"):
            evidence.extend(task_profile.get(field_name) or [])
        item = dict(item)
        item["task_relevance"] = task_profile.get("task_relevance")
        item["evidence"] = [truncate(text, 350) for text in evidence[:4]] or item.get("evidence", [])
        enriched.append(item)
    return enriched


def analyze_idea(idea: str) -> IdeaProfile:
    normalized = _normalize(idea)
    expanded = expand_query_text(normalized)
    tokens = _tokens(expanded)
    terms = {
        group: [term for term in terms if term in expanded]
        for group, terms in TERM_GROUPS.items()
    }
    query_variants = build_query_variants(idea, expanded, terms)
    return IdeaProfile(
        raw_idea=idea,
        normalized=expanded,
        terms=terms,
        tokens=tokens,
        query_variants=query_variants,
    )


def build_query_variants(idea: str, expanded: str, terms: dict[str, list[str]]) -> list[str]:
    variants = [idea, expanded]
    for group in ("tasks", "methods", "datasets", "metrics"):
        if terms[group]:
            variants.append(" ".join(terms[group]))
    if terms["tasks"] and terms["methods"]:
        variants.append(" ".join(terms["methods"] + terms["tasks"]))
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))


def expand_query_text(text: str) -> str:
    """Rewrite Chinese research queries into English first, then append alias fallback terms."""
    llm_rewrite = rewrite_query_with_glm4_flash(text)
    aliases = []
    for source, targets in TERM_ALIASES.items():
        if source in text:
            aliases.extend(targets)
    return " ".join([text, *llm_rewrite.get("query_variants", []), *aliases])


@lru_cache(maxsize=256)
def rewrite_query_with_glm4_flash(text: str) -> dict[str, Any]:
    """Use glm4_flash to convert a user query into English retrieval variants."""
    if not text.strip():
        return {}
    if not os.getenv("ZHIPU_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
    if not os.getenv("ZHIPU_API_KEY"):
        return {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            from project.recommendate_project.myllm import glm4_flash
        except ModuleNotFoundError:
            from myllm import glm4_flash

        system_prompt = (
            "You rewrite Chinese research-assistant queries into English retrieval queries for an English "
            "recommender-system paper database. Return strict JSON only. Do not answer the user. "
            "Schema: {\"english_query\": string, \"keywords\": string[], \"query_variants\": string[]}."
        )
        user_prompt = (
            "Rewrite this query for academic paper retrieval. Preserve technical meaning, expand common "
            "abbreviations, and include recommender-system terms when relevant.\n\n"
            f"Query: {text}"
        )
        response = glm4_flash.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        parsed = parse_json_object(str(response.content))
        english_query = clean_rewrite_text(parsed.get("english_query"))
        keywords = clean_rewrite_list(parsed.get("keywords"))
        variants = clean_rewrite_list(parsed.get("query_variants"))
        query_variants = [english_query, *keywords, *variants]
        query_variants = [item for item in dict.fromkeys(query_variants) if item]
        return {
            "english_query": english_query,
            "keywords": keywords,
            "query_variants": query_variants,
        }
    except Exception:
        return {}


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clean_rewrite_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip())


def clean_rewrite_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        cleaned = clean_rewrite_text(item)
        if cleaned:
            output.append(cleaned)
    return output[:8]


def rank_profiles(idea: IdeaProfile, profiles: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for profile in profiles:
        score_parts = aspect_scores(idea, profile)
        total = (
            score_parts["overall"] * 0.45
            + score_parts["problem"] * 0.20
            + score_parts["method"] * 0.20
            + score_parts["data_metric"] * 0.10
            + score_parts["topic"] * 0.05
        )
        if total <= 0:
            continue
        scored.append((total, score_parts, profile))

    ranked = []
    seen_papers: set[str] = set()
    for score, score_parts, profile in sorted(scored, key=lambda item: item[0], reverse=True):
        paper_key = _paper_key(profile)
        if paper_key in seen_papers:
            continue
        seen_papers.add(paper_key)
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "similarity": round(score, 3),
                "aspect_scores": {key: round(value, 3) for key, value in score_parts.items()},
                "paper_id": profile.get("paper_id"),
                "title": profile.get("title"),
                "topic": profile.get("topic"),
                "evidence": evidence_snippets(idea, profile),
                "source_pdf": profile.get("source_pdf"),
            }
        )
        if len(ranked) >= top_k:
            break
    return ranked


def aspect_scores(idea: IdeaProfile, profile: dict[str, Any]) -> dict[str, float]:
    overall_text = profile_text(profile)
    problem_text = " ".join(profile.get("research_problem") or [])
    method_text = " ".join(profile.get("methods") or [])
    data_metric_text = " ".join((profile.get("datasets") or []) + (profile.get("metrics") or []))
    topic_text = f"{profile.get('topic', '')} {profile.get('title', '')}"

    return {
        "overall": text_overlap(idea.tokens, overall_text),
        "problem": text_overlap(idea.tokens, problem_text),
        "method": text_overlap(idea.tokens, method_text),
        "data_metric": text_overlap(idea.tokens, data_metric_text),
        "topic": text_overlap(idea.tokens, topic_text),
    }


def build_similarity_matrix(idea: IdeaProfile, related_work: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for item in related_work:
        aspect = item["aspect_scores"]
        matrix.append(
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "problem_similarity": aspect["problem"],
                "method_similarity": aspect["method"],
                "data_metric_similarity": aspect["data_metric"],
                "overall_similarity": item["similarity"],
            }
        )
    return matrix


def score_idea(idea: IdeaProfile, related_work: list[dict[str, Any]]) -> dict[str, Any]:
    max_similarity = max((item["similarity"] for item in related_work), default=0.0)
    avg_top3 = sum(item["similarity"] for item in related_work[:3]) / min(len(related_work), 3) if related_work else 0.0
    has_method = bool(idea.terms["methods"])
    has_task = bool(idea.terms["tasks"])
    has_data_or_metric = bool(idea.terms["datasets"] or idea.terms["metrics"])

    novelty = clamp(1.0 - max_similarity * 0.85)
    feasibility = clamp(0.35 + 0.2 * has_task + 0.2 * has_method + 0.15 * has_data_or_metric + 0.1 * bool(related_work))
    academic_value = clamp(0.35 + 0.2 * has_task + 0.15 * has_method + 0.15 * avg_top3 + 0.1 * bool(idea.terms["metrics"]))
    risk = clamp(0.25 + 0.45 * max_similarity + 0.15 * (not has_data_or_metric) + 0.1 * (not has_method))
    overall = clamp((novelty + feasibility + academic_value + (1 - risk)) / 4)

    return {
        "novelty": round(novelty, 3),
        "feasibility": round(feasibility, 3),
        "academic_value": round(academic_value, 3),
        "risk": round(risk, 3),
        "overall": round(overall, 3),
        "max_related_similarity": round(max_similarity, 3),
        "interpretation": interpret_scores(novelty, feasibility, academic_value, risk),
    }


def infer_research_gaps(idea: IdeaProfile, related_work: list[dict[str, Any]]) -> list[str]:
    gaps = []
    if not related_work:
        gaps.append("知识库中未找到明显相似工作，需要扩大检索范围或联网补充。")
    if not idea.terms["datasets"]:
        gaps.append("用户请求尚未明确目标数据集，后续需要指定 MovieLens/Amazon/Yelp 等实验场景。")
    if not idea.terms["metrics"]:
        gaps.append("用户请求尚未明确评价指标，建议补充 Recall/NDCG/robustness/fairness 等可衡量目标。")
    if related_work and related_work[0]["similarity"] >= 0.55:
        gaps.append("已有工作相似度较高，需要把创新点落到新任务、新约束、新数据或新评测协议上。")
    if not idea.terms["methods"]:
        gaps.append("当前方法假设还不够具体，需要明确模型结构、检索方式或训练目标。")
    return gaps or ["当前请求有基本任务和方法信号，下一步应通过更细粒度 chunk 检索确认证据。"]


def suggest_innovations(idea: IdeaProfile, related_work: list[dict[str, Any]], scores: dict[str, Any]) -> list[str]:
    suggestions = []
    if scores["novelty"] < 0.55:
        suggestions.append("围绕最相似论文增加明确差异：新约束、新任务定义、新数据设置或新评价指标。")
    else:
        suggestions.append("保留当前组合方向，但补充系统性 related work 检索来确认 novelty。")
    if idea.terms["methods"]:
        suggestions.append("把方法贡献拆成可验证模块，并设计 ablation 证明每个模块的必要性。")
    else:
        suggestions.append("先确定核心技术路线，例如 RAG、LLM agent、GNN、contrastive learning 或 causal modeling。")
    if not idea.terms["metrics"]:
        suggestions.append("把创新点转化为指标提升或新评测协议，否则学术价值会较难证明。")
    if related_work:
        suggestions.append(f"优先精读相似度最高的论文：{related_work[0]['title']}。")
    return suggestions


def suggest_experiments(idea: IdeaProfile, related_work: list[dict[str, Any]]) -> list[str]:
    datasets = idea.terms["datasets"] or ["MovieLens", "Amazon", "Yelp"]
    metrics = idea.terms["metrics"] or ["Recall@K", "NDCG@K"]
    baselines = [item["title"] for item in related_work[:3] if item.get("title")]
    experiments = [
        f"在 {', '.join(datasets[:3])} 上构建主实验，至少报告 {', '.join(metrics[:3])}。",
        "设计 ablation study，拆分 idea 中的核心模块和普通 baseline 的差异。",
        "增加效率、鲁棒性或公平性分析，避免只报告准确率提升。",
    ]
    if baselines:
        experiments.append("把相似工作作为 baseline 或 related work anchor：" + "；".join(baselines[:3]))
    return experiments


def generate_direct_answer(idea: IdeaProfile, related_work: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    llm_answer = generate_direct_answer_with_glm4_flash(idea, related_work, task_type)
    if llm_answer:
        return llm_answer
    return fallback_direct_answer(idea, related_work, task_type)


def generate_direct_answer_with_glm4_flash(
    idea: IdeaProfile,
    related_work: list[dict[str, Any]],
    task_type: str,
) -> dict[str, Any]:
    if not os.getenv("ZHIPU_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
    if not os.getenv("ZHIPU_API_KEY"):
        return {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            from project.recommendate_project.myllm import glm4_flash
        except ModuleNotFoundError:
            from myllm import glm4_flash

        evidence = [
            {
                "title": item.get("title"),
                "similarity": item.get("similarity"),
                "topic": item.get("topic"),
                "evidence": item.get("evidence", [])[:2],
            }
            for item in related_work[:5]
        ]
        system_prompt = (
            "你是推荐系统研究助手。请根据用户 idea、识别出的英文检索词和已召回论文证据，"
            "生成一个可泛化的中文 direct answer。不要套固定模板，不要声称没有检索。"
            "如果证据较弱，要明确说“知识库证据较弱”。"
            "只返回 JSON，schema: {\"verdict\": string, \"design_options\": string[], "
            "\"key_decisions\": string[], \"validation_plan\": string[], \"evidence_note\": string}."
        )
        user_prompt = json.dumps(
            {
                "user_idea": idea.raw_idea,
                "task_type": task_type,
                "query_variants": idea.query_variants,
                "detected_terms": idea.terms,
                "retrieved_evidence": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
        response = glm4_flash.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        parsed = parse_json_object(str(response.content))
        return normalize_direct_answer(parsed)
    except Exception:
        return {}


def normalize_direct_answer(value: dict[str, Any]) -> dict[str, Any]:
    verdict = clean_rewrite_text(value.get("verdict"))
    design_options = clean_rewrite_list(value.get("design_options"))
    key_decisions = clean_rewrite_list(value.get("key_decisions"))
    validation_plan = clean_rewrite_list(value.get("validation_plan"))
    evidence_note = clean_rewrite_text(value.get("evidence_note"))
    if not verdict:
        return {}
    return {
        "verdict": verdict,
        "design_options": design_options,
        "key_decisions": key_decisions,
        "validation_plan": validation_plan,
        "evidence_note": evidence_note,
        "source": "glm4_flash",
    }


def fallback_direct_answer(idea: IdeaProfile, related_work: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    evidence_note = (
        f"当前知识库召回了 {len(related_work)} 篇候选论文，可作为 related work 起点。"
        if related_work
        else "当前知识库没有召回强相关论文，需要扩展检索词或联网补充。"
    )
    task_verdicts = {
        "qa": "可以回答，但需要把结论限定在已检索到的论文证据范围内。",
        "literature_summary": "可以总结，但需要围绕研究问题、方法、实验、结果和局限组织证据。",
        "paper_compare": "可以对比，但需要先固定比较维度，并说明每篇论文证据是否充分。",
        "research_plan": "可以制定研究计划，但需要把路线落到数据集、baseline、指标和消融实验。",
        "idea_review": "这个方向可以继续推进，但需要先把用户提出的技术方向转成明确任务、模型位置和评价指标。",
    }
    return {
        "verdict": task_verdicts.get(task_type, task_verdicts["idea_review"]),
        "design_options": [
            "先确定应用场景，例如序列推荐、冷启动、多模态推荐、生成式召回或可解释推荐。",
            "把每个技术实体分配到模型流程中的具体位置，例如召回、表征、重排序、解释或数据增强。",
            "保留传统推荐 baseline，证明新增模块带来的增益不是来自额外参数或数据泄漏。",
        ],
        "key_decisions": [
            "目标数据集和推荐任务是什么。",
            "核心技术模块解决什么痛点。",
            "相似论文中哪些方法可作为 baseline 或 related work anchor。",
        ],
        "validation_plan": [
            "主指标使用 Recall@K、NDCG@K 或任务相关指标。",
            "为每个新增模块设计 ablation。",
            "补充效率、鲁棒性或可解释性分析。",
        ],
        "evidence_note": evidence_note,
        "source": "fallback",
    }


def assess_risks(idea: IdeaProfile, related_work: list[dict[str, Any]], scores: dict[str, Any]) -> list[str]:
    risks = []
    if scores["max_related_similarity"] >= 0.55:
        risks.append("新颖性风险：知识库中已有高度相似方向，需要更明确地区分贡献。")
    if not idea.terms["datasets"]:
        risks.append("可复现风险：缺少目标数据集会导致实验设计不够落地。")
    if not idea.terms["metrics"]:
        risks.append("评价风险：缺少指标会让 idea 难以证明有效性。")
    if scores["feasibility"] < 0.6:
        risks.append("实现风险：任务、方法或数据约束不完整，建议先缩小问题定义。")
    return risks or ["主要风险可控，下一步重点是细化 related work 差异和实验协议。"]


def evidence_snippets(idea: IdeaProfile, profile: dict[str, Any], limit: int = 3) -> list[str]:
    candidates = []
    for field_name in ("research_problem", "methods", "results", "limitations", "abstract"):
        value = profile.get(field_name)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, str):
            candidates.append(value)

    scored = []
    for text in candidates:
        score = text_overlap(idea.tokens, text)
        if score > 0:
            scored.append((score, text))
    return [truncate(text, 350) for _, text in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def profile_text(profile: dict[str, Any]) -> str:
    parts = [
        profile.get("title", ""),
        profile.get("abstract", ""),
        " ".join(profile.get("research_problem") or []),
        " ".join(profile.get("methods") or []),
        " ".join(profile.get("datasets") or []),
        " ".join(profile.get("metrics") or []),
        " ".join(profile.get("results") or []),
        " ".join(profile.get("limitations") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def text_overlap(query_tokens: list[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = set(_tokens(_normalize(text)))
    if not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = len(query_set & text_tokens)
    return clamp(overlap / math.sqrt(len(query_set) * len(text_tokens)))


def render_markdown(review: dict[str, Any]) -> str:
    task_type = review.get("task_type", "idea_review")
    idea = review["idea_profile"]["raw_idea"]
    scores = review["scores"]
    direct_answer = review.get("direct_answer") or {}
    lines = [
        "# Research Assistant Report",
        "",
        "## Task",
        "",
        task_type,
        "",
        "## User Request",
        "",
        idea,
        "",
        "## Direct Answer",
        "",
        direct_answer.get("verdict", "可以继续探索，但需要先明确任务、方法位置和可验证贡献。"),
        "",
        "### Design Options",
        "",
    ]
    lines.extend(f"- {item}" for item in direct_answer.get("design_options", []))
    lines.extend(["", "### Key Decisions", ""])
    lines.extend(f"- {item}" for item in direct_answer.get("key_decisions", []))
    lines.extend(["", "### Validation Plan", ""])
    lines.extend(f"- {item}" for item in direct_answer.get("validation_plan", []))
    if direct_answer.get("evidence_note"):
        lines.extend(["", f"> {direct_answer['evidence_note']}", ""])
    lines.extend([
        "## Task Scores",
        "",
        f"- Novelty: {scores['novelty']}",
        f"- Feasibility: {scores['feasibility']}",
        f"- Academic value: {scores['academic_value']}",
        f"- Risk: {scores['risk']}",
        f"- Overall: {scores['overall']}",
        f"- Interpretation: {scores['interpretation']}",
        "",
        "## Related Work",
        "",
    ])
    for item in review["related_work"]:
        relevance = f", task relevance: {item.get('task_relevance')}" if item.get("task_relevance") else ""
        lines.append(f"- [{item['rank']}] {item['title']} ({item['similarity']}{relevance})")
    if review.get("task_paper_profiles"):
        lines.extend(["", "## On-Demand Paper Reading", ""])
        for profile in review["task_paper_profiles"]:
            lines.append(f"### {profile.get('title')}")
            if profile.get("research_problem"):
                lines.append("- Problem: " + " ".join(profile["research_problem"][:2]))
            if profile.get("methods"):
                lines.append("- Method: " + " ".join(profile["methods"][:2]))
            if profile.get("results"):
                lines.append("- Evidence/results: " + " ".join(profile["results"][:2]))
            if profile.get("limitations"):
                lines.append("- Limits: " + " ".join(profile["limitations"][:1]))
            lines.append("")
    lines.extend(["", "## Evidence Gaps", ""])
    lines.extend(f"- {gap}" for gap in review["gaps"])
    lines.extend(["", "## Research Suggestions", ""])
    lines.extend(f"- {suggestion}" for suggestion in review["innovation_suggestions"])
    lines.extend(["", "## Experiment / Follow-Up Suggestions", ""])
    lines.extend(f"- {experiment}" for experiment in review["experiment_suggestions"])
    lines.extend(["", "## Risks", ""])
    lines.extend(f"- {risk}" for risk in review["risk_assessment"])
    return "\n".join(lines) + "\n"


def interpret_scores(novelty: float, feasibility: float, academic_value: float, risk: float) -> str:
    if novelty < 0.45:
        return "相关工作非常接近，需要显著强化差异点。"
    if feasibility < 0.55:
        return "idea 还偏概念化，建议先收敛任务、数据和实验指标。"
    if academic_value >= 0.65 and risk <= 0.55:
        return "具备继续推进价值，建议进入细粒度 related work 和实验设计。"
    return "可以作为探索方向，但需要补充证据、baseline 和可验证贡献。"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", text)
    phrase_tokens = []
    for group_terms in TERM_GROUPS.values():
        for term in group_terms:
            if term in text:
                phrase_tokens.append(term)
    output = []
    for token in tokens + phrase_tokens:
        token = token.lower().strip()
        if token and token not in STOPWORDS and len(token) >= 2:
            output.append(token)
    return list(dict.fromkeys(output))


def _slug(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    slug = "-".join(words[:10]) if words else "idea-review"
    return slug[:80]


def _paper_key(profile: dict[str, Any]) -> str:
    title = str(profile.get("title") or "").lower()
    if title:
        return re.sub(r"\W+", "", title)
    return str(profile.get("paper_id") or "")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    raise SystemExit(main())
