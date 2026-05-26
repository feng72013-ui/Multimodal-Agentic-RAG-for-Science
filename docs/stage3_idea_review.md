# Stage 3 Idea Review

阶段 3 的目标是实现“给一个科研 idea，检索相关工作、判断相似研究、给出创新建议和可行性初评”。

当前实现优先复用阶段 2 的 `paper_profiles.jsonl`，不依赖 Milvus 和 LLM。这样可以稳定评测 idea review 的核心结构，后续再接入 Milvus profile collection 或 LLM critic。

## Inputs

输入：

- 用户科研 idea 文本。
- `processed_rag/paper_profiles.jsonl`。

示例命令：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m idea_review_pipeline.review \
  --idea "用 RAG 增强 next item recommendation，并比较不同检索粒度对 Recall 和 NDCG 的影响"
```

## Outputs

默认输出目录：

- `processed_rag/idea_reviews/*.json`
- `processed_rag/idea_reviews/*.md`

当前样例输出：

- `processed_rag/idea_reviews/rag-next-item-recommendation-recall-ndcg.json`
- `processed_rag/idea_reviews/rag-next-item-recommendation-recall-ndcg.md`

报告包含：

- `idea_profile`
- `related_work`
- `similarity_matrix`
- `scores`
- `gaps`
- `innovation_suggestions`
- `experiment_suggestions`
- `risk_assessment`

## Implementation

核心入口：

- `idea_review_pipeline/review.py`

在线工作流接入：

- `graph/idea_review_node.py`
- `graph/all_route.py`
- `graph/workflow.py`

当阶段 1 分类器识别到 `task_type == "idea_review"` 时，工作流会进入 `idea_review_node`，直接基于 paper profiles 生成 Markdown idea review 报告。其他任务仍走原 RAG 路径。

## Algorithm

当前是确定性启发式算法，流程如下：

1. `analyze_idea`
   - 从 idea 中抽取任务、方法、数据集和指标信号。
   - 生成 query variants，例如原始 idea、方法词、任务词、方法+任务组合。

2. `rank_profiles`
   - 遍历 `paper_profiles.jsonl`。
   - 对每篇论文计算多维相似度：overall、problem、method、data_metric、topic。
   - 按加权相似度排序，并按论文标题去重。

3. `build_similarity_matrix`
   - 输出相似工作矩阵，展示 problem/method/data_metric/overall similarity。

4. `score_idea`
   - 输出 novelty、feasibility、academic_value、risk、overall。

5. `suggest_*`
   - 基于 idea 缺失项和相似工作，生成研究 gap、创新建议、实验建议和风险评估。

## Scoring Rubric

| Score | Meaning |
|---|---|
| `novelty` | 与最相似工作越远，分数越高。 |
| `feasibility` | idea 是否包含任务、方法、数据/指标信号，以及是否能找到相关工作支撑。 |
| `academic_value` | idea 是否有明确任务和方法，并能连接已有研究问题。 |
| `risk` | 相似工作过近、缺少数据集、缺少指标或方法过抽象会提高风险。 |
| `overall` | 综合 novelty、feasibility、academic value 和反向 risk。 |

注意：当前评分是工程启发式初评，不是最终学术评价。它适合做第一轮筛选和面试展示，后续应接入 LLM critic、人工标注和更细粒度 evidence。

## Evaluation

评测目录：

- `test_idea_review_eval/sample_ideas.jsonl`
- `test_idea_review_eval/evaluate.py`
- `test_idea_review_eval/README.md`

运行命令：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_idea_review_eval.evaluate
```

最新结果：

| Metric | Value |
|---|---:|
| ideas | 20 |
| passed | 20/20 |
| pass_rate | 1.000 |

验收标准：

- 默认 pass rate >= `0.80`。
- 每个报告必须包含 related work、similarity matrix、scores、gaps、innovation suggestions、experiment suggestions 和 risk assessment。
- 窄领域 idea 可在样例中设置 `min_related`，例如 watermark 方向当前只有一篇强相关 profile，因此设置 `min_related=1`。

## Current Limits

- 当前相似度是 token overlap，不是 embedding semantic search。
- 当前不会直接查 Milvus chunk，也不会做联网补充。
- novelty 分数只基于当前本地 paper profiles，不代表全网 novelty。
- 对中文短语和复杂混合 idea 的解析仍偏粗糙。
- 评分 rubric 是启发式，后续需要 LLM critic 或人工评测校准。

## Interview Notes

阶段 3 可以这样讲：

我们先不直接让 LLM 对 idea 做主观判断，而是把 idea 拆成任务、方法、数据和指标，再用阶段 2 的论文级 profile 做相似工作检索。系统先输出 evidence-backed related work 和相似矩阵，再基于固定 rubric 计算 novelty、feasibility、academic value 和 risk。这样做的好处是可解释、可复现、可评测；缺点是语义召回和评分质量还受启发式规则限制，后续可以接 embedding retrieval、Milvus profile collection 和 LLM critic。

