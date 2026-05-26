# FastAPI Service

本目录把推荐系统论文科研助手发布为 FastAPI 服务。服务层复用现有 LangGraph 工作流和
`research_agents.orchestrator.run_research_workflow`，不改变原有命令行入口。

## Install

```bash
cd /home/lf/mount/LLM/project/recommendate_project
python3 -m pip install -r requirements.txt
```

如果项目依赖安装在特定环境中，请先激活该环境再启动服务。

## Start

```bash
cd /home/lf/mount/LLM/project/recommendate_project
bash scripts/run_fastapi.sh
```

或者手动启动：

```bash
cd /home/lf/mount/LLM/project/recommendate_project
PYTHONPATH=src RECSYS_TEXT_DEVICE=cpu /home/lf/conda/envs/my_ocr_env/bin/python \
  -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## Main Endpoints

### Health

```bash
curl http://127.0.0.1:8000/api/health
```

### Task List

```bash
curl http://127.0.0.1:8000/api/tasks
```

### Structured Research Workflow

适合前端直接展示结构化科研报告、相关论文、评分和 agent trace。

```bash
curl -X POST http://127.0.0.1:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请评估一个把 RAG 用于推荐系统冷启动解释的 idea",
    "task_type": "idea_review",
    "top_k": 5,
    "on_demand_top_k": 5,
    "use_milvus": true
  }'
```

### LangGraph Chat Workflow

适合做完整聊天体验，支持会话续接和人工审批。

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你能帮我总结 LightGCN 的核心贡献吗？",
    "user_name": "ZS"
  }'
```

响应中会返回 `session_id`。后续同一会话继续传这个值：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<previous-session-id>",
    "message": "approve",
    "user_name": "ZS"
  }'
```

### Image Upload Chat

```bash
curl -X POST http://127.0.0.1:8000/api/chat/upload \
  -F "file=@/path/to/image.png" \
  -F "message=请分析这张论文图表" \
  -F "user_name=ZS"
```

### File Access

用于前端读取项目目录或 `/tmp` 下的图片、表格截图等文件。

```text
GET /api/files?path=/home/lf/mount/LLM/project/recommendate_project/data/processed_rag/assets/example.png
```

同时挂载了静态目录：

```text
/assets        -> data/processed_rag/assets
/processed_ocr -> data/processed_ocr
```
