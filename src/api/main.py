from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from graph.my_state import InvalidInputError

from .schemas import (
    ChatRequest, ChatResponse, HealthResponse, ResearchRequest, TaskType,
    ResearchResponse, TaskInfo, ModelConfig, ConfigResponse,
    KnowledgeBaseCreateRequest, KnowledgeBaseResponse, KnowledgeBaseListResponse,
    KnowledgeBaseIngestRequest, KnowledgeBaseJobResponse,
)
from .services import PROJECT_ROOT, TASKS, run_chat_api, run_research_api, stream_chat_api
from .knowledge_bases import (
    create_knowledge_base,
    get_job,
    get_knowledge_base,
    list_knowledge_bases,
    start_ingest_job,
    upload_document,
)


SERVICE_VERSION = "1.0.0"
UPLOAD_DIR = Path("/tmp/recommendate_project_uploads")
CONFIG_FILE = PROJECT_ROOT / "config.json"

# 全局配置存储
_current_config: ModelConfig = ModelConfig()


def load_config() -> ModelConfig:
    """从文件加载配置"""
    global _current_config
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _current_config = ModelConfig(**data)
        except Exception:
            _current_config = ModelConfig()
    return _current_config


def save_config(config: ModelConfig) -> None:
    """保存配置到文件"""
    global _current_config
    _current_config = config
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, ensure_ascii=False, indent=2)


# 初始化加载配置
load_config()


app = FastAPI(
    title="MARS Scholar API",
    description="FastAPI service for multimodal agentic RAG and scientific research workflows.",
    version=SERVICE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mount_static_dir(route: str, directory: Path, name: str) -> None:
    if directory.exists():
        app.mount(route, StaticFiles(directory=str(directory)), name=name)


mount_static_dir("/assets", PROJECT_ROOT / "data" / "processed_rag" / "assets", "assets")
mount_static_dir("/processed_ocr", PROJECT_ROOT / "data" / "processed_ocr", "processed_ocr")


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(_, exc: InvalidInputError):
    return JSONResponse(status_code=exc.error_code, content={"detail": exc.message})


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="mars-scholar", version=SERVICE_VERSION)


@app.get("/api/tasks", response_model=list[TaskInfo])
async def tasks() -> list[TaskInfo]:
    return [TaskInfo(**task) for task in TASKS]


@app.get("/api/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """获取当前模型配置"""
    return ConfigResponse(
        config=load_config(),
        message="获取配置成功"
    )


@app.post("/api/config", response_model=ConfigResponse)
async def update_config(config: ModelConfig) -> ConfigResponse:
    """更新模型配置"""
    save_config(config)
    return ConfigResponse(
        config=config,
        message="配置已保存"
    )


@app.get("/api/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def knowledge_bases() -> KnowledgeBaseListResponse:
    return KnowledgeBaseListResponse(knowledge_bases=list_knowledge_bases())


@app.post("/api/knowledge-bases", response_model=KnowledgeBaseResponse)
async def create_kb(request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        knowledge_base=create_knowledge_base(request.name, request.description),
        message="知识库已创建",
    )


@app.get("/api/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_kb(knowledge_base_id: str) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        knowledge_base=get_knowledge_base(knowledge_base_id),
        message="获取知识库成功",
    )


@app.post("/api/knowledge-bases/{knowledge_base_id}/documents", response_model=KnowledgeBaseResponse)
async def upload_kb_document(
    knowledge_base_id: str,
    file: UploadFile = File(...),
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        knowledge_base=upload_document(knowledge_base_id, file),
        message="PDF 已上传",
    )


@app.post("/api/knowledge-bases/{knowledge_base_id}/ingest", response_model=KnowledgeBaseJobResponse)
async def ingest_kb(
    knowledge_base_id: str,
    request: KnowledgeBaseIngestRequest,
) -> KnowledgeBaseJobResponse:
    return KnowledgeBaseJobResponse(job=start_ingest_job(knowledge_base_id, request))


@app.get("/api/knowledge-bases/jobs/{job_id}", response_model=KnowledgeBaseJobResponse)
async def get_kb_job(job_id: str) -> KnowledgeBaseJobResponse:
    return KnowledgeBaseJobResponse(job=get_job(job_id))


@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    try:
        return ResearchResponse(**await run_research_api(request))
    except InvalidInputError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"科研工作流执行失败：{exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(**await run_chat_api(request))
    except InvalidInputError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图工作流执行失败：{exc}") from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in stream_chat_api(request):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except InvalidInputError as exc:
            yield json.dumps({"event": "error", "detail": exc.message}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "detail": f"图工作流执行失败：{exc}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/chat/upload", response_model=ChatResponse)
async def chat_upload(
    file: UploadFile = File(...),
    message: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    knowledge_base_id: str | None = Form(default=None),
    task_type_hint: TaskType | None = Form(default=None),
    user_name: str = Form(default="ZS"),
) -> ChatResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with upload_path.open("wb") as writer:
        shutil.copyfileobj(file.file, writer)
    request = ChatRequest(
        message=message,
        session_id=session_id,
        user_name=user_name,
        knowledge_base_id=knowledge_base_id,
        task_type_hint=task_type_hint,
        image_path=str(upload_path),
    )
    try:
        return ChatResponse(**await run_chat_api(request))
    except InvalidInputError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图工作流执行失败：{exc}") from exc


@app.get("/api/files")
async def get_file(path: str):
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    try:
        resolved = file_path.resolve()
        project_root = PROJECT_ROOT.resolve()
        tmp_root = Path("/tmp").resolve()
        if not (resolved.is_relative_to(project_root) or resolved.is_relative_to(tmp_root)):
            raise HTTPException(status_code=403, detail="只允许访问项目目录或 /tmp 下的文件。")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(str(resolved))
