# Stage 0 Baseline Report

本报告记录阶段 0 检索基线。基线命令均应在项目目录 `/home/lf/mount/LLM/project/recommendate_project` 下运行。

## Runtime

正式基线环境：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python
```

环境确认：

- Python: `3.12.12`
- `sentence_transformers`: `5.1.2`
- `DASHSCOPE_API_KEY`: 未配置，因此阶段 0 跳过 `multimodal_dense` 正式评测。

不要使用系统 `/usr/bin/python3` 跑正式基线；该环境缺少 `sentence_transformers`，会导致 `text_dense` 和 `hybrid` 失败。

## Baseline Commands

非联网检查：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --dry-run
```

正式检索基线：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --modes sparse text_dense hybrid --top-k 5
```

多模态可选基线：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --modes multimodal_dense --top-k 5
```

`multimodal_dense` 只有在 `DASHSCOPE_API_KEY` 配置后才运行。

## Baseline Artifacts

最新完整基线输出：

- Results: `test_retrieval_eval/results/retrieval_results_20260522_142849.jsonl`
- Summary: `test_retrieval_eval/results/summary_20260522_142849.md`
- Sample rows: `test_retrieval_eval/results/sample_rows_20260522_142849.json`

Collection:

- Database: `recommendation_system`
- Collection: `recsys_paper_rag`
- Rows: `12072`
- Sample rows checked: `5`
- Baseline queries: `10`
- Top K: `5`

## Mode Overview

| Mode | Query groups | Non-empty | Errors | Avg expected term recall |
|---|---:|---:|---:|---:|
| `sparse` | 10 | 9 | 0 | 0.875 |
| `text_dense` | 10 | 10 | 0 | 0.975 |
| `hybrid` | 10 | 10 | 0 | 1.000 |

阶段 0 采用 `hybrid` 作为主检索基线。

## Hybrid Top-1 Results

| Query ID | Top-1 title | Category | Topic | Term recall |
|---|---|---|---|---:|
| `kw_green_red_watermark` | `3 GREW: Green-Red Watermarking for Recommendation --> 3.1 Semantic-Consistent Hashing` | `text` | `2026-05-19_causal_fairness_privacy` | 1.0 |
| `kw_sigformer` | `1 INTRODUCTION` | `text` | `2026-05-19_rag_recommender` | 1.0 |
| `semantic_rag_recommender` | `SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION --> 2 RELATED WORK` | `text` | `2026-05-19_rag_recommender` | 1.0 |
| `semantic_fairness_privacy` | `7 Conclusion` | `text` | `2026-05-19_causal_fairness_privacy` | 1.0 |
| `classic_lightgcn` | `4 EXPERIMENTS --> 4.4 Ablation and Effectiveness Analyses` | `text` | `2026-05-19_classic_recommender` | 1.0 |
| `classic_bert4rec` | `References` | `text` | `2026-05-19_hypergraph_recommender` | 1.0 |
| `image_framework` | `Figure 1` | `image` | `2026-05-19_rag_recommender` | 1.0 |
| `table_benchmark` | `Table 1` | `table` | `2026-05-19_rag_recommender` | 1.0 |
| `multimodal_recommender` | `Methodology --> General Feature Enhancement` | `text` | `2026-05-19_rag_recommender` | 1.0 |
| `llm_recommender` | `An Embarrassingly Simple Graph Heuristic Reveals Shortcut-Solvable Benchmarks for Sequential Recommendation --> 3 Problem Setup and Diagnostic Heuristic --> 3.1 Problem Formulation` | `text` | `2026-05-19_sequential_session_recommender` | 1.0 |

## Image and Table Baseline

普通 `hybrid` 对图表类 query 已能返回图表记录：

| Query ID | Expected | Hybrid top-1 | Category | Topic | Term recall |
|---|---|---|---|---|---:|
| `image_framework` | model framework / architecture figure | `Figure 1` | `image` | `2026-05-19_rag_recommender` | 1.0 |
| `table_benchmark` | benchmark / dataset / metrics table | `Table 1` | `table` | `2026-05-19_rag_recommender` | 1.0 |

后续仍建议补跑 `multimodal_dense`，用于验证多模态向量字段是否优于普通文本 hybrid。

## Fixes Made During Stage 0

修复了 `test_retrieval_eval/retrievers.py::normalize_hits` 对 pymilvus `Hit` 对象的字段抽取问题。

修复前：

- `hits` 只有 `id`、`score`、`rank`。
- Summary 中 Top Results 显示为空：` (, )`。

修复后：

- 优先读取 `hit.fields`，正确保留 `title`、`category`、`topic`、`paper_id` 等 output fields。
- Summary 可显示 top-1 标题、类别和 topic。

此修复只影响评测报告输出，不改变主 RAG 工作流行为。

## Known Issues

1. `multimodal_dense` 未跑正式基线
   - 原因：当前环境未配置 `DASHSCOPE_API_KEY`。
   - 影响：阶段 0 已验证 image/table 可被 ordinary hybrid 召回，但尚未单独验证多模态向量检索质量。

2. 图片描述中存在 API 限流文本
   - `processed_rag/image_descriptions.jsonl` 中出现 `Zhipu API HTTP 429`。
   - 影响：部分图表描述包含模型失败信息，可能干扰图表检索和回答质量。

3. 当前 query 集太小
   - `sample_queries.jsonl` 只有 10 条。
   - 影响：只能作为最小检索基线，不能覆盖科研 idea 查重、论文总结、方法比较、研究计划生成。

4. `expected_term_recall` 是弱指标
   - 当前指标只检查 expected terms 是否出现在返回文本中。
   - 影响：不能完全代表科研答案质量、事实一致性或 idea novelty 判断质量。

5. RAGAS 当前只在回答后评估
   - 当前评估尚未覆盖任务路由、idea 分析、论文结构化抽取等未来任务。

## Stage 1 Inputs

建议阶段 1 将任务意图扩展为：

| Intent | Meaning |
|---|---|
| `qa` | 普通知识库问答。 |
| `idea_review` | 科研 idea 查重、相关工作分析和创新性评估。 |
| `literature_summary` | 单篇或多篇文献总结。 |
| `paper_compare` | 多篇论文方法、实验、结论对比。 |
| `research_plan` | 基于现有工作生成研究路线和实验计划。 |

后续 baseline query 扩展建议：

- 保留当前 10 条检索 query。
- 新增 10 条 idea review query。
- 新增 10 条 literature summary query。
- 新增 10 条 paper compare query。
- 新增 5-10 条 research plan query。

