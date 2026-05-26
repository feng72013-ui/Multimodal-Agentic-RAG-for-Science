# Stage 0 Acceptance Checklist

本清单用于验收“现状梳理与基线固化”阶段。

## Documentation

- [x] 创建 `docs/stage0_current_state.md`。
- [x] 创建 `docs/stage0_baseline_report.md`。
- [x] 创建 `docs/stage0_acceptance_checklist.md`。
- [x] 文档说明当前系统已经能做什么。
- [x] 文档说明当前系统还不能做什么。
- [x] 文档记录阶段 1 输入建议。

## Architecture Baseline

- [x] 梳理 `graph.workflow` 主要节点。
- [x] 梳理路由条件和失败回退路径。
- [x] 记录文字版数据流。
- [x] 记录当前状态字段。
- [x] 明确当前系统定位是推荐系统论文 RAG 助手。

## Knowledge Base Baseline

- [x] 记录 Milvus database：`recommendation_system`。
- [x] 记录 Milvus collection：`recsys_paper_rag`。
- [x] 记录 collection rows：`12072`。
- [x] 记录文本、图片、表格三类数据。
- [x] 记录 schema 关键字段。
- [x] 记录已支持检索模式：`sparse`、`text_dense`、`hybrid`、`multimodal_dense`。

## Retrieval Baseline

- [x] 使用 `my_ocr_env` 运行 dry-run。
- [x] 使用 `my_ocr_env` 运行 `sparse text_dense hybrid` 正式基线。
- [x] 生成 results 文件：`test_retrieval_eval/results/retrieval_results_20260522_142849.jsonl`。
- [x] 生成 summary 文件：`test_retrieval_eval/results/summary_20260522_142849.md`。
- [x] 生成 sample rows 文件：`test_retrieval_eval/results/sample_rows_20260522_142849.json`。
- [x] 记录各模式 non-empty 数、error 数、平均 expected term recall。
- [x] 记录 image/table query 在普通 hybrid 下的表现。
- [ ] 运行 `multimodal_dense` 正式基线。

`multimodal_dense` 未完成原因：当前未配置 `DASHSCOPE_API_KEY`。该项不阻塞阶段 0 验收，但应作为阶段 1 或阶段 6 的补充基线。

## Evaluation Tooling

- [x] 修复 `test_retrieval_eval/retrievers.py::normalize_hits` 对 pymilvus `Hit` 的字段抽取。
- [x] 修复后 summary 能显示 top-1 标题、类别和 topic。
- [x] 修复不影响主 RAG workflow。

## Risks Recorded

- [x] 记录图表描述中存在 `Zhipu API HTTP 429` 数据质量风险。
- [x] 记录当前 query 集只有 10 条，覆盖不足。
- [x] 记录 `expected_term_recall` 只是弱指标。
- [x] 记录 RAGAS 还不是任务级验收体系。
- [x] 记录 `multimodal_dense` 依赖外部 API key。

## Acceptance Result

阶段 0 验收状态：通过，带一个非阻塞遗留项。

非阻塞遗留项：

- 配置 `DASHSCOPE_API_KEY` 后补跑：

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --modes multimodal_dense --top-k 5
```

正式复现命令：

```bash
cd /home/lf/mount/LLM/project/recommendate_project
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --dry-run
/home/lf/conda/envs/my_ocr_env/bin/python -m test_retrieval_eval.evaluate --modes sparse text_dense hybrid --top-k 5
```

