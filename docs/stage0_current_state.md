# Stage 0 Current State

本文档固化当前科研助手的现状。阶段 0 原始版本只做梳理、基线和风险记录；当前主工作流已经演进为统一科研助手入口。

## System Positioning

当前系统是面向推荐系统论文知识库的 RAG 助手，已经具备论文片段检索、图表检索、基于检索上下文回答、回答质量评估、人工审批和互联网兜底能力。

当前系统定位为完整科研助手，而不是单一 idea 生成器。统一支持：

- 文献问答：围绕推荐系统论文、方法、实验、图表进行证据化回答。
- 文献阅读与总结：围绕研究问题、方法、数据集、指标、结果和局限组织结构化总结。
- 论文/方法对比：按任务、模型、数据、指标、实验结论和适用场景对比。
- idea 讨论与优化：检索相关工作，分析相似点、差异、可行性、风险和实验设计。
- 研究计划：生成技术路线、baseline、指标、消融实验和阶段产出。

## Main Workflow

入口：`graph/workflow.py`

现有 LangGraph 节点：

| Node | Responsibility |
|---|---|
| `process_input` | 解析用户文本和图片输入，写入 `input_type`、`input_text`、`input_image`、`user`。 |
| `classify_intent` | 识别 `qa`、`literature_summary`、`paper_compare`、`idea_review`、`research_plan`、`direct_response`。 |
| `research_assistant_node` | 统一科研助手工作流。所有科研任务进入同一报告结构，不同任务只替换内部策略。 |
| `first_chatbot` | 绑定 `search_context` 工具，先尝试检索当前用户历史上下文。 |
| `search_context` | 从 `recsys_context_history` collection 检索相似历史问答。 |
| `retriever_node` | 从论文知识库检索文本、图片、表格相关上下文。 |
| `second_chatbot` | 当历史上下文足够时，直接基于当前消息生成回答。 |
| `third_chatbot` | 基于论文检索片段和图表路径生成中文 Markdown 回答。 |
| `evaluate_node` | 使用 RAGAS 指标评估回答质量，当前包括 relevancy、context relevance、context precision、faithfulness。 |
| `human_approval` | 在低分或需确认时中断，等待人工审批。 |
| `fourth_chatbot` | 人工拒绝或知识库回答未通过时，调用互联网搜索兜底。 |
| `web_search_node` | 执行 `my_search` 互联网搜索工具。 |

文字版数据流：

```text
User input
  -> process_input
  -> classify_intent
  -> research_assistant_node, for qa/literature_summary/paper_compare/idea_review/research_plan
  -> END
```

统一科研助手报告结构：

```text
Research Assistant Report
  -> Task
  -> User Request
  -> Direct Answer
  -> Task Scores
  -> Related Work
  -> On-Demand Paper Reading
  -> Evidence Gaps
  -> Research Suggestions
  -> Experiment / Follow-Up Suggestions
  -> Risks
  -> Tool / Agent Calls
  -> Knowledge Base Retrieval Snippets
  -> Retrieved Images Or Tables
  -> Evaluation And Approval
  -> Multi-Agent Collaboration
  -> Next Steps
  -> Agent Trace
```

不同任务共享这套外层结构，内部策略不同：

| Task | Internal strategy |
|---|---|
| `qa` | 检索论文证据，直接回答并标注边界。 |
| `literature_summary` | 按问题、方法、数据、指标、结果、局限组织文献总结。 |
| `paper_compare` | 固定比较维度，输出共同点、差异和证据缺口。 |
| `idea_review` | 讨论并优化科研想法，分析相关工作、创新空间、风险和实验设计。 |
| `research_plan` | 生成研究路线、baseline、消融、指标和阶段产出。 |

特殊路径：

- 只有图片输入：`process_input -> retriever_node -> third_chatbot -> END`，跳过 RAGAS 文本评估。
- 问候、身份和能力说明：`process_input -> classify_intent -> first_chatbot -> END`，不检索知识库。
- 历史上下文足够：`search_context -> second_chatbot -> END`。
- 知识库回答未通过：人工拒绝后进入互联网兜底，并说明信息来自公开互联网搜索。

## State Fields

核心状态定义在 `graph/my_state.py`：

| Field | Purpose |
|---|---|
| `messages` | LangChain 消息列表。 |
| `input_type` | `has_text` 或 `only_image`。 |
| `input_text` | 用户文本输入。 |
| `input_image` | 用户图片路径或 data URL。 |
| `user` | 当前用户名，用于历史上下文过滤。 |
| `context_retrieved` | 论文知识库检索片段。 |
| `images_retrieved` | 检索到的图片或表格路径。 |
| `evaluate_score` | 当前主评估分数。 |
| `response_relevancy` | 回答相关性。 |
| `context_relevance` | 上下文相关性。 |
| `context_precision` | 上下文精确性。 |
| `faithfulness` | 回答忠实度。 |
| `human_answer` | 人工审批结果。 |
| `agent_trace` | 统一科研助手中的智能体执行轨迹。 |
| `agent_outputs` | Supervisor、Retriever、PaperAnalyst、ResearchCritic、ResearchPlanner、Writer 的中间输出。 |
| `related_work` | 当前任务召回的相关论文。 |
| `task_paper_profiles` | 针对当前任务按需阅读后的论文结构化证据。 |

## Knowledge Base Shape

Milvus 主 collection：`recsys_paper_rag`

数据库：`recommendation_system`

阶段 0 基线统计：`12072` rows。

当前 schema 支持字段：

| Field | Purpose |
|---|---|
| `doc_id` | chunk、图片或表格的稳定标识。 |
| `category` | `text`、`image`、`table`。 |
| `topic` | 论文主题目录。 |
| `paper_id` | 论文标识。 |
| `title` | 标题路径、图号或表号。 |
| `text` | chunk 文本，或图片/表格描述文本。 |
| `filename` | 原 PDF 或图片路径。 |
| `filetype` | 文件类型。 |
| `image_path` | 图片或表格截图路径。 |
| `page_start` | 起始页。 |
| `page_end` | 结束页。 |
| `metadata_json` | OCR、paper metadata 和来源信息。 |
| `sparse` | BM25 sparse vector。 |
| `text_dense` | 本地 Qwen text embedding。 |
| `multimodal_dense` | DashScope multimodal embedding。 |

## Retrieval Modes

现有检索能力由 `test_retrieval_eval/retrievers.py` 和 `graph/search_node.py` 使用：

| Mode | Implementation | Notes |
|---|---|---|
| `sparse` | client-side BM25 sparse query over Milvus `sparse` field | 关键词检索效果稳定。 |
| `text_dense` | local Qwen text embedding over `text_dense` | 需要 `sentence_transformers`，应使用 `my_ocr_env`。 |
| `hybrid` | sparse + dense reciprocal-rank fusion | 当前主推荐基线模式。 |
| `multimodal_dense` | DashScope text/image embedding over `multimodal_dense` | 需要 `DASHSCOPE_API_KEY`。 |

## Data Pipeline

现有数据链路：

```text
papers/
  -> ocr_dots_vllm_batch.py / dots_ocr
  -> processed_ocr/
  -> post_ocr_pipeline/
  -> processed_rag/chunks.jsonl
  -> processed_rag/image_descriptions.jsonl
  -> vector_ingest_pipeline/ingest.py
  -> Milvus recsys_paper_rag
```

文本记录来自 `vector_ingest_pipeline/records.py::build_text_records`。

图片和表格记录来自 `build_image_records`，其中 `category == "Table"` 会映射为 `table`，其他视觉元素映射为 `image`。

## Current Boundaries

当前系统可以：

- 回答推荐系统论文、方法、实验、图表相关问题。
- 检索用户历史上下文并复用历史回答。
- 检索文本 chunk、图片和表格描述。
- 基于检索上下文生成带来源线索的回答。
- 对回答做 RAGAS 自动评估。
- 在知识库不足或人工拒绝时调用互联网搜索兜底。

当前系统还不能稳定完成：

- 建立引用图谱和关键引用关系。
- 对所有任务使用同一套真实 RAGAS 评估和人工审批中断；统一科研助手目前输出代理评估指标，尚未完全接入 `evaluate_node -> human_approval`。
- 对所有论文标题和章节标题进行完全清洗；当前已在运行时过滤部分章节标题，但历史入库数据仍可能需要重建。
