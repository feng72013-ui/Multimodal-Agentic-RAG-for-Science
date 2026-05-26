# 阶段 4：多智能体协作工作流

## 目标

阶段 4 在不改变 Milvus schema、不新增入库流程的前提下，为科研助手增加可测试的多智能体编排层。当前实现优先服务 `idea_review`，复用阶段 2 的论文画像和阶段 3 的 idea 评估能力，把职责拆成六个确定性 agent，并输出统一的 `agent_trace`。

## Agent 角色

| 角色 | 职责 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| Supervisor | 判断任务、制定执行计划和验收项 | 用户 query、task_type | `supervisor_plan` |
| Retriever | 检索相关论文画像并生成相似度矩阵 | query、paper profiles | `related_work`、`similarity_matrix` |
| PaperAnalyst | 汇总相关工作的主题和证据 | related work | `top_papers`、`shared_themes` |
| IdeaCritic | 评估新颖性、可行性、学术价值和风险 | review scores、related work | `scores`、`risk_assessment` |
| ResearchPlanner | 生成创新点、实验建议和下一步计划 | review、paper analysis | `next_steps` |
| Writer | 汇总最终报告和执行轨迹 | 各 agent 输出 | `final_report` |

## 当前实现

- 新增 `research_agents/orchestrator.py`
- 新增 `test_multi_agent_eval/`
- `graph/idea_review_node.py` 已切换为调用 `run_research_workflow`
- `graph/my_state.py` 新增：
  - `agent_trace`
  - `agent_outputs`

当前版本是确定性 agent，不依赖额外 LLM API。这样做的目的不是最终形态，而是先把协作协议、状态结构、验收指标固化下来，后续可以逐步把某个角色替换成 LLM 或工具调用。

## 协作机制

执行顺序：

1. Supervisor 生成任务计划。
2. Retriever 调用阶段 3 的 `review_idea`，基于阶段 2 的 `paper_profiles.jsonl` 检索相关论文。
3. PaperAnalyst 对 top related work 做证据摘要。
4. IdeaCritic 根据分数和相似工作判断风险。
5. ResearchPlanner 生成实验建议和下一步计划。
6. Writer 渲染 Markdown 报告，并附带 agent trace。

降级路径：

- 如果论文画像不存在或检索不到相关论文，Retriever 标记为 `degraded`。
- PaperAnalyst 会提示需要扩大检索范围。
- Writer 仍输出报告，避免主流程中断。

## 验收命令

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m py_compile research_agents/orchestrator.py graph/idea_review_node.py graph/my_state.py test_multi_agent_eval/evaluate.py
/home/lf/conda/envs/my_ocr_env/bin/python -m test_multi_agent_eval.evaluate
```

建议回归：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_idea_review_eval.evaluate
/home/lf/conda/envs/my_ocr_env/bin/python -m test_intent_eval.evaluate
/home/lf/conda/envs/my_ocr_env/bin/python -m test_paper_profile_eval.evaluate
```

## 验收标准

- `agent_trace` 包含六个角色：Supervisor、Retriever、PaperAnalyst、IdeaCritic、ResearchPlanner、Writer。
- `agent_outputs` 包含每个角色的结构化输出。
- `final_report` 非空，且包含 related work、scores、risks、next steps。
- 阶段 3 的 idea review 评测仍通过。
- 阶段 1/2 的回归评测仍通过。

## 当前边界

- 只完整接入 `idea_review`；`literature_summary`、`paper_compare`、`research_plan` 后续阶段再扩展。
- 多智能体目前是确定性规则编排，不是多 LLM 并发对话。
- 相关工作来自 `processed_rag/paper_profiles.jsonl`，还没有把论文画像写回 Milvus。
- 当前评测只覆盖 5 条 agent workflow 样例，后续应扩展到 30-50 条任务级 baseline。
