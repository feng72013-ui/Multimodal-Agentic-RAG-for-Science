# 科研助手智能体分阶段开发计划

## 1. 总体定位

基于现有 `graph` 工作流、Milvus 知识库、OCR 后处理、向量入库与检索评测模块，逐步升级为“多智能体协作式科研助手”。

当前基础能力：
- 已有 LangGraph 工作流：输入处理、历史上下文检索、论文知识库检索、RAG 回答、RAGAS 评估、人工审批、互联网兜底。
- 已有知识库：Milvus 存储文本、图片、表格、多模态向量、BM25 稀疏向量和元数据。
- 已有文献处理链路：PDF/OCR、chunk、图片表格描述、向量入库、检索评测。

目标能力：
- 科研 idea 生成、查重、扩展、价值评估。
- 文献自动解析、结构化摘要、引用关系抽取、质量评估。
- 多智能体协作：检索、文献分析、创新评估、方法设计、批判审稿、报告生成。

## 2. 分阶段开发计划表

| 阶段 | 周期 | 目标 | 关键任务 | 核心模块 | 测试与验收 | 交付物 |
|---|---:|---|---|---|---|---|
| 阶段 0：现状梳理与基线固化 | 1 周 | 固化当前 RAG 基线，明确后续扩展边界 | 梳理 `graph` 工作流节点；确认 Milvus schema、字段覆盖、检索模式；整理已有 OCR/入库/评测链路；建立 baseline 查询集 | 现有 RAG 工作流、知识库连接、检索评测脚本 | 20-50 条科研问答样例可稳定跑通；hybrid 检索 Top-5 命中率有基线报告；失败样例分类完成 | 当前架构说明、baseline 评测报告、数据字段清单 |
| 阶段 1：科研意图识别与任务路由 | 1-2 周 | 让系统识别用户是在问知识、评估 idea、读论文、找创新点还是写报告 | 扩展状态字段；新增 intent classifier；设计任务类型：`qa`、`idea_review`、`literature_summary`、`paper_compare`、`research_plan`；替换当前单一路由逻辑 | 意图识别 Agent、任务路由器、状态 schema 扩展 | 每类任务 10 条样例，路由准确率 >= 85%；错误路由可回退到普通 RAG | 任务路由模块、测试集、路由评测报告 |
| 阶段 2：文献处理与按需结构化知识提炼 | 2-3 周 | 建立轻量论文级索引，并在用户任务触发时只精读相关 Top-K 文献 | 2A 离线生成轻量 paper profile：题名、摘要、主题、粗粒度方法/数据/指标候选、质量风险标签；2B 在 idea review 等任务中，对检索命中的 Top-K 论文基于 chunk 做按需结构化阅读：研究问题、方法、数据集、指标、结果、局限、证据片段；LLM 精读改为可选或高风险触发 | Lightweight Paper Index、On-demand Paper Reader、Structure Extractor、Risk/Quality Filter | 轻量索引覆盖率 >= 90%；在线 Top-K 结构化字段可用率 >= 80%；默认流程不依赖全库 LLM；抽样或高风险 LLM 复核通过率 >= 80% | 轻量论文索引、按需 task paper profiles、证据片段、引用关系初版 |
| 阶段 3：科研 idea 检索、查重与评估 | 2-3 周 | 实现“给一个 idea，判断已有工作、分析相似研究、给出创新建议” | 将用户 idea 拆解为问题、方法、任务、数据、贡献点；多路检索相关论文；生成相似工作矩阵；评估 novelty、feasibility、academic value、risk；输出扩展方向 | Idea Analyzer Agent、Novelty Search Agent、Research Gap Agent、Idea Scorer | 20 个 idea 样例，输出必须包含相似工作、差异点、创新建议、可行性评分；人工验收通过率 >= 80% | idea 评估报告模板、评分 rubric、相似论文分析模块 |
| 阶段 4：多智能体协作工作流 | 2 周 | 将单链 RAG 升级为可协作、可审查、可回退的多 Agent 系统 | 设计 Supervisor 调度；拆分检索、阅读、创新评估、实验设计、批判审稿、报告写作 Agent；定义共享状态、任务输入输出协议、失败重试策略；引入自评和交叉评审 | Supervisor Agent、Retriever Agent、Paper Analyst Agent、Idea Critic Agent、Planner Agent、Writer Agent | 多 Agent 任务可端到端完成；每个 Agent 输出结构化 JSON；失败时可降级到普通 RAG；日志可追踪每一步决策 | 多 Agent LangGraph 工作流、Agent I/O 协议、协作日志 |
| 阶段 5：产品化接口与体验完善 | 2 周 | 提供稳定可用的科研助手接口 | 封装 CLI/API；支持上传 PDF、输入 idea、选择任务类型；生成 Markdown/JSON 报告；支持用户历史上下文；增加配置项和错误提示 | API/CLI 层、报告生成器、会话记忆、任务配置 | 端到端场景测试：上传论文、总结论文、评估 idea、生成研究计划；平均响应可控；异常信息清晰 | 可用 Demo、API 文档、用户操作说明 |
| 阶段 6：评测、优化与持续迭代 | 持续 | 提升可靠性、检索质量和科研判断质量 | 建立 gold set；跟踪检索召回、事实一致性、引用正确性、idea 评分一致性；优化 chunk、rerank、prompt、Agent 协作策略；加入人工反馈闭环 | Retrieval Eval、RAGAS、人工评分平台、反馈学习机制 | 检索 Top-5 召回持续提升；faithfulness >= 0.75；报告事实错误率低于人工阈值 | 周期性评测报告、失败案例库、优化记录 |

## 3. 核心功能设计

### 科研创意生成与评估

核心流程：
1. 用户输入初步 idea。
2. Idea Analyzer 抽取结构化要素：研究问题、目标任务、方法假设、数据需求、预期贡献。
3. Retriever Agent 先查询轻量论文索引、Milvus 文本、图表、表格和历史上下文。
4. On-demand Paper Reader 只对检索命中的 Top-K 论文做任务相关结构化阅读。
5. Novelty Search Agent 找相似论文，并按“问题相似、方法相似、场景相似、实验相似”分组。
6. Research Gap Agent 总结已有工作的不足。
7. Idea Critic Agent 从学术价值、可行性、创新性、风险、实验成本五个维度评分。
8. Writer Agent 生成结构化报告。

建议输出格式：
- idea 摘要
- 相关研究列表
- 与已有工作的相同点和差异点
- 潜在创新点
- 可行实验设计
- 风险与改进建议
- 综合评分

### 文献处理与知识提炼

核心流程：
1. 沿用现有 PDF/OCR/chunk/入库链路。
2. 离线只生成轻量论文级索引，不对全库做昂贵 LLM 深度抽取。
3. 每篇论文生成 coarse `paper_profile`：
   - title
   - authors
   - year / venue
   - research_problem
   - method
   - datasets
   - metrics
   - results
   - conclusion
   - limitations
   - key_figures
   - references
   - quality_score
4. 对图片、表格继续使用现有多模态字段，但补充“图表作用”和“实验结论”描述。
5. 引用关系优先从 references section、正文 citation pattern、metadata 三路抽取。

### 多智能体协作机制

采用 Supervisor + Specialist 模式：

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| Supervisor Agent | 判断任务、拆分步骤、调度子 Agent | 用户请求、当前状态 | 执行计划、路由决策 |
| Retriever Agent | 检索论文 chunk、图表、历史上下文 | 查询、任务类型、过滤条件 | 证据片段、来源、得分 |
| Paper Analyst Agent | 阅读论文并结构化总结 | 论文片段、metadata | paper profile、摘要 |
| Citation Agent | 抽取引用关系 | 论文文本、metadata | citation graph records |
| Idea Analyzer Agent | 解析用户 idea | 用户 idea | 结构化 idea |
| Novelty Agent | 判断相关工作与新颖性 | idea、检索结果 | 相似工作矩阵、novelty 判断 |
| Critic Agent | 批判性评估 | idea、相关工作、证据 | 风险、漏洞、评分 |
| Research Planner Agent | 设计扩展方向和实验计划 | idea、gap、约束 | 研究路线、实验设计 |
| Writer Agent | 汇总最终报告 | 所有中间结果 | Markdown/JSON 报告 |

共享状态建议新增：
- `task_type`
- `idea_profile`
- `paper_profiles`
- `citation_edges`
- `evidence_pack`
- `agent_traces`
- `quality_scores`
- `final_report`

## 4. 知识库集成方案

保留现有 Milvus 主 collection，用于 chunk、图片、表格检索。

新增或扩展：
- `paper_profiles`：存论文级结构化信息。
- `citation_edges`：存引用边，字段包括 `source_paper_id`、`target_title`、`target_doi/arxiv_id`、`context`。
- `idea_reviews`：存用户 idea 评估历史，支持后续上下文召回。
- `agent_runs`：存多智能体执行轨迹，便于调试和评测。

检索策略：
- 普通问答：沿用 hybrid 检索。
- idea 评估：idea 拆解后多 query 检索，并按问题、方法、数据集、任务分别召回。
- 文献总结：优先按 `paper_id/title/filename` 精确过滤，再做 chunk 聚合。
- 图表问题：沿用 multimodal_dense，并结合图表描述。
- 质量评估：结合 metadata、引用信息、实验完整性和 LLM rubric。

## 5. 测试策略与验收标准

| 测试类型 | 内容 | 验收标准 |
|---|---|---|
| 单元测试 | intent 分类、结构化抽取、评分 rubric、引用抽取 | 核心函数覆盖主要分支 |
| 检索测试 | sparse、dense、hybrid、multimodal | Top-5 召回有稳定 baseline，新增功能不低于旧版 |
| Agent 流程测试 | idea 评估、论文总结、研究计划生成 | 每类任务至少 10 条端到端样例通过 |
| 事实一致性测试 | 回答是否基于证据 | RAGAS faithfulness 目标 >= 0.75 |
| 人工评测 | 科研价值、创新点质量、报告可读性 | 人工通过率 >= 80% |
| 回归测试 | 旧问答能力是否受影响 | 当前普通 RAG 查询保持可用 |
| 异常测试 | 无检索结果、OCR 缺失、模型失败、API key 缺失 | 明确降级或提示，不生成伪造结论 |

## 6. 资源需求估算

人力：
- 后端/Agent 工程师：1-2 人
- RAG/检索工程师：1 人
- 数据处理/OCR 工程师：0.5-1 人
- 科研领域评测人员：1 人兼职
- 产品/交互设计：0.5 人，可后置

技术栈：
- LangGraph / LangChain
- Milvus
- Qwen3 text embedding
- DashScope multimodal embedding
- RAGAS
- DotsOCR + vLLM
- Pydantic / JSON Schema
- pytest
- 可选 reranker：bge-reranker、Qwen reranker 或 API rerank

工具与数据：
- 已有 `papers/`、`processed_ocr/`、`processed_rag/`
- 现有 `test_retrieval_eval`
- 新增 idea 评测样例集、论文结构化 gold set、人工评审表

## 7. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| OCR 质量不稳定 | 文献摘要和引用抽取错误 | 增加 OCR 质量评分；低质量页提示人工复核 |
| 检索召回不足 | idea 查重遗漏相关工作 | 多 query 检索、hybrid + rerank、论文级 profile 检索 |
| LLM 生成幻觉 | 科研报告不可信 | 强制 evidence pack；无证据时明确说明不足 |
| 多 Agent 流程过慢 | 用户体验差 | 先串行实现，再并行检索/分析；缓存 paper profile |
| 引用关系抽取困难 | citation graph 不完整 | 第一版只做显式引用抽取，后续再做 DOI/arXiv 对齐 |
| 评分主观性强 | idea 评价不稳定 | 固定 rubric；保存评分理由；引入人工校准集 |
| Milvus schema 频繁变化 | 数据迁移成本高 | 新增 collection 优先，减少破坏性修改 |
| 外部 API 失败 | 多模态或联网搜索不可用 | 保留本地文本检索降级路径 |

## 8. 阶段性交付物清单

- 阶段 0：现状架构文档、baseline 检索评测报告、字段清单。
- 阶段 1：任务路由器、intent 测试集、路由准确率报告。
- 阶段 2：paper profile 生成器、结构化论文 JSONL、引用抽取初版、文献摘要模板。
- 阶段 3：idea 评估 Agent、相似工作矩阵、创新点评分报告模板。
- 阶段 4：多 Agent LangGraph 工作流、Agent 输入输出协议、执行 trace。
- 阶段 5：CLI/API Demo、用户文档、报告导出能力。
- 阶段 6：持续评测集、失败案例库、优化报告。

## 9. 默认假设

- 第一版聚焦推荐系统论文知识库，后续再扩展到通用科研领域。
- 优先复用当前 `graph`、Milvus、OCR、入库和检索评测模块。
- 新能力优先通过新增 Agent、状态字段和 collection 实现，避免破坏现有 RAG 闭环。
- 首个可用版本目标周期为 8-10 周，完整评测和优化持续迭代。
