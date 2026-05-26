# Stage 2 Paper Profiles

阶段 2 的目标是把 chunk 级知识库进一步提炼成论文级结构化知识，支撑后续 `literature_summary`、`idea_review`、`paper_compare` 和 `research_plan`。

当前阶段不调用 LLM、不写入 Milvus，先基于现有 `processed_rag` 产物生成可复现的离线结构化文件。

## Inputs

输入文件：

- `processed_rag/chunks.jsonl`
- `processed_rag/image_descriptions.jsonl`

`chunks.jsonl` 提供论文文本片段、页码、标题路径和 metadata。

`image_descriptions.jsonl` 提供图片/表格 caption、上下文、图表路径和页码。

## Outputs

生成文件：

- `processed_rag/paper_profiles.jsonl`
- `processed_rag/citation_edges.jsonl`
- `processed_rag/paper_profile_summary.json`

生成命令：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m paper_profile_pipeline.extract
```

快速预览：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m paper_profile_pipeline.extract --limit-papers 3 --dry-run
```

## Profile Schema

每条 `paper_profiles.jsonl` 表示一篇论文：

| Field | Meaning |
|---|---|
| `topic` | 论文所在主题目录。 |
| `paper_id` | 论文 ID。 |
| `title` | 论文标题，优先来自 metadata。 |
| `authors` | 作者列表。 |
| `year` | 年份，来自 metadata。 |
| `venue` | 会议/期刊，来自 metadata。 |
| `doi` | DOI，来自 metadata。 |
| `arxiv_id` | arXiv ID，来自 metadata。 |
| `source_pdf` | 原 PDF 路径。 |
| `abstract` | 摘要文本。 |
| `research_problem` | 研究问题/动机相关句子。 |
| `methods` | 方法、模型、框架相关句子。 |
| `datasets` | 启发式识别出的数据集名称。 |
| `metrics` | 启发式识别出的评价指标。 |
| `results` | 实验结果、提升、验证相关句子。 |
| `conclusions` | 结论相关句子；若未识别到 conclusion section，会用后部文本或摘要 fallback。 |
| `limitations` | 局限、future work、threat 相关句子。 |
| `key_visuals` | 关键图片/表格路径、caption 和页码。 |
| `quality` | 字段完整性质量分。 |
| `coverage` | chunk 数、图表数、页码范围、识别到的 section。 |

## Citation Edges

`citation_edges.jsonl` 是引用关系初版，当前只从 References section 中抽取显式引用。

字段：

| Field | Meaning |
|---|---|
| `source_topic` | 引用方论文主题。 |
| `source_paper_id` | 引用方论文 ID。 |
| `citation_marker` | references 中的编号。 |
| `target_title` | 启发式猜测出的被引论文标题。 |
| `target_raw` | 原始 reference 文本。 |
| `source` | 当前固定为 `references_section`。 |

当前 citation graph 还没有做 DOI/arXiv 对齐，也没有把引用边解析到库内具体论文 ID。这个会放到后续增强。

## Extraction Strategy

当前实现是确定性启发式抽取，入口在 `paper_profile_pipeline/extract.py`。

主要步骤：

1. 按 `(topic, paper_id)` 聚合所有 chunks 和图表记录。
2. 根据 `title_path` 和正文开头判断 section：abstract、introduction、method、experiment、conclusion、limitation、references。
3. 从不同 section 中按关键词抽取研究问题、方法、结果、结论和局限句子。
4. 从全文中识别常见 dataset 和 metric。
5. 从图表描述中选取关键 visuals。
6. 从 references section 抽取 citation edges。
7. 计算 profile quality score。

## Evaluation

验收脚本：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_paper_profile_eval.evaluate
```

默认验收线：

- required field fill rate >= `0.80`
- average quality score >= `0.60`
- citation edge count >= `1`

最新结果：

| Metric | Value |
|---|---:|
| profiles | 156 |
| citation_edges | 5508 |
| avg_quality | 0.957 |
| title fill rate | 1.000 |
| abstract fill rate | 1.000 |
| research_problem fill rate | 0.942 |
| methods fill rate | 1.000 |
| results fill rate | 1.000 |
| conclusions fill rate | 1.000 |
| quality fill rate | 1.000 |

## Current Limits

- 句子抽取是启发式的，有些字段会包含较粗糙或不够精炼的句子。
- 长篇 thesis 或 survey 的目录、表格文本可能会干扰 results/conclusions。
- dataset 和 metric 目前是词表匹配，不是开放抽取。
- citation edges 还没有做去重、实体对齐或库内 paper matching。
- 当前产物还没有写回 Milvus，后续可以设计 `paper_profiles` 独立 collection。

## Interview Notes

阶段 2 可以这样讲：

先前 RAG 只是在 chunk 层面检索，适合问答，但不适合做 idea 评估和论文级分析。因此我们增加了离线 paper profile 层，把同一篇论文的 chunk、图表和 references 聚合成结构化记录。这样在线系统后续可以先检索论文级 profile，再钻取 chunk 证据，既提升回答结构，也为 novelty search、paper comparison 和 research planning 准备更稳定的知识单元。

