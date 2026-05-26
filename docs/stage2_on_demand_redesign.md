# Stage 2 Redesign: Lightweight Index + On-Demand Reading

Stage 2 is now split into two layers.

## 2A. Offline Lightweight Paper Index

`processed_rag/paper_profiles.jsonl` remains as a coarse paper-level index. It is used to quickly find candidate papers for an idea.

This layer should stay cheap and deterministic:

- title, authors, topic, paper id, source PDF
- abstract or short profile text
- coarse method/dataset/metric hints
- rule-based quality/risk signals

It is not treated as a fully trusted final structured profile.

## 2B. Online On-Demand Paper Reading

When the user asks for an idea review:

1. The system analyzes the user idea.
2. It retrieves related papers from the lightweight profile index.
3. Only the retrieved Top-K papers are matched to `processed_rag/chunks.jsonl`.
4. `idea_review_pipeline.on_demand_reader` extracts task-specific structure from those chunks.
5. The final idea review uses these task-specific profiles for evidence, comparison, and planning.

Default behavior uses deterministic extraction. LLM reading is opt-in:

```bash
python3 -m idea_review_pipeline.review \
  --idea "用 LLM agent 做推荐系统论文综述生成，并评估 citation coverage 和 factuality" \
  --top-k 5 \
  --on-demand-top-k 5
```

Opt-in LLM reader:

```bash
python3 -m idea_review_pipeline.review \
  --idea "..." \
  --top-k 5 \
  --on-demand-top-k 5 \
  --use-llm-reader
```

The Stage 4 orchestrator enables on-demand deterministic reading by default:

```bash
python3 -m research_agents.orchestrator \
  --query "..." \
  --top-k 5 \
  --on-demand-top-k 5 \
  --dry-run
```

## Why

This avoids full-library LLM processing. LLM or deeper structure extraction is only needed for papers that are actually relevant to a user task. It keeps cost bounded when the library grows or when the project moves to another research field.

