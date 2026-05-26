# Stage 1 Intent Routing

阶段 1 的目标是让科研助手在进入检索和回答前，先识别用户请求属于哪类科研任务，并把任务类型写入 LangGraph 状态。当前阶段不实现完整多 Agent，只建立稳定的任务路由基础。

## Task Types

| Task type | Meaning | Current behavior |
|---|---|---|
| `qa` | 普通知识库问答、找论文、找图表、解释概念 | 沿用原 RAG 路径，按证据回答。 |
| `idea_review` | 科研 idea 查重、相关工作分析、创新性和可行性初评 | 沿用原检索路径，但回答按相关工作、差异、创新点和风险组织。 |
| `literature_summary` | 单篇或多篇文献阅读与结构化总结 | 沿用原检索路径，但回答按研究问题、方法、实验、结果、局限组织。 |
| `paper_compare` | 多篇论文或方法对比 | 沿用原检索路径，但回答按比较维度、共同点、差异和适用场景组织。 |
| `research_plan` | 研究计划、技术路线、实验设计、开题 proposal | 沿用原检索路径，但回答按研究目标、依据、路线、实验和风险组织。 |

## Implementation

新增模块：

- `graph/intent_node.py`
  - `classify_task(text, input_type)`：纯规则、可测试的任务分类函数。
  - `classify_intent_node(state)`：LangGraph 节点，写入任务字段。

扩展状态：

- `task_type`
- `task_confidence`
- `task_reason`
- `task_signals`

工作流变化：

```text
START
  -> process_input
  -> classify_intent
  -> first_chatbot or retriever_node
  -> existing RAG / evaluation / fallback flow
```

只有图片输入仍然进入图表/论文检索路径，任务类型默认回退为 `qa`。

## Routing Strategy

当前分类器是确定性规则分类器，不调用 LLM。选择这个实现有三个原因：

- 阶段 1 可以稳定复现，不依赖外部模型质量或 API。
- 可以用离线样例集快速评测。
- 后续可平滑替换为 LLM classifier 或 hybrid classifier，只要保持状态字段不变。

分类依据包括：

- 中英文关键词，例如 `idea`、`创新点`、`summarize`、`对比`、`research plan`。
- 少量模式规则，例如 “idea + 可行/创新/已有” 判为 `idea_review`。
- 任务优先级，用于处理重叠表达：`idea_review` > `research_plan` > `paper_compare` > `literature_summary` > `qa`。

## Evaluation

评测目录：

- `test_intent_eval/sample_intents.jsonl`
- `test_intent_eval/evaluate.py`
- `test_intent_eval/README.md`

运行命令：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_intent_eval.evaluate
```

最新结果：

| Task type | Correct | Accuracy |
|---|---:|---:|
| `qa` | 10/10 | 1.000 |
| `idea_review` | 10/10 | 1.000 |
| `literature_summary` | 10/10 | 1.000 |
| `paper_compare` | 10/10 | 1.000 |
| `research_plan` | 10/10 | 1.000 |
| Overall | 50/50 | 1.000 |

验收线：overall accuracy >= 0.85。

## Current Limits

- 当前是规则分类，不理解深层语义；真实用户表达更复杂时可能需要 LLM classifier。
- 当前只改变任务识别和回答结构，不新增独立 idea review Agent、paper profile 抽取或多 Agent 调度。
- 当前所有任务仍复用同一套知识库检索节点，尚未按任务类型生成多 query 检索计划。

## Next Step Inputs

阶段 2 可以在这个基础上继续做：

- `literature_summary` 的论文级结构化抽取。
- `idea_review` 的 idea profile、相似工作矩阵和 novelty scoring。
- `research_plan` 的实验设计模板和证据追踪。
- 将规则分类器升级为 “规则 + LLM 校验” 的 hybrid classifier。

