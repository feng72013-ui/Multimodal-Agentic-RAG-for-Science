from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal["qa", "idea_review", "literature_summary", "paper_compare", "research_plan"]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TaskInfo(BaseModel):
    task_type: str
    label: str
    description: str


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    task_type: TaskType = "idea_review"
    top_k: int = Field(default=5, ge=1, le=20)
    on_demand_top_k: int = Field(default=5, ge=0, le=20)
    use_llm_reader: bool = False
    use_milvus: bool = True


class ResearchResponse(BaseModel):
    task_type: str
    status: str
    retrieval_source: str
    final_report: str
    related_work: list[dict[str, Any]] = Field(default_factory=list)
    task_paper_profiles: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    gaps: list[Any] = Field(default_factory=list)
    innovation_suggestions: list[Any] = Field(default_factory=list)
    experiment_suggestions: list[Any] = Field(default_factory=list)
    risk_assessment: list[Any] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, description="Text message or approval text.")
    session_id: str | None = None
    user_name: str = "ZS"
    knowledge_base_id: str | None = Field(default=None, description="Knowledge base selected for retrieval.")
    task_type_hint: TaskType | None = Field(default=None, description="Task selected by the active frontend workspace.")
    image_path: str | None = Field(default=None, description="Server-local image path.")
    image_data_url: str | None = Field(default=None, description="Base64 data URL image.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    requires_approval: bool = False
    task_type: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class MultimodalLLMConfig(BaseModel):
    """多模态 LLM 模型配置"""
    model_name: str = Field(default="glm-4v-flash", description="模型名称")
    base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4/", description="API 基础 URL")
    api_key: str = Field(default="", description="API Key")
    temperature: float = Field(default=0.3, ge=0, le=2, description="温度参数")


class MultimodalEmbeddingConfig(BaseModel):
    """多模态向量模型配置"""
    model_name: str = Field(default="qwen3-vl-embedding", description="模型名称")
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", description="API 基础 URL")
    api_key: str = Field(default="", description="API Key")
    embedding_dim: int = Field(default=2560, description="向量维度")


class ModelConfig(BaseModel):
    """完整的模型配置"""
    multimodal_llm: MultimodalLLMConfig = Field(default_factory=MultimodalLLMConfig)
    multimodal_embedding: MultimodalEmbeddingConfig = Field(default_factory=MultimodalEmbeddingConfig)


class ConfigResponse(BaseModel):
    """配置响应"""
    config: ModelConfig
    message: str


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class KnowledgeBaseDocument(BaseModel):
    id: str
    filename: str
    stored_path: str
    size_bytes: int
    status: str = "uploaded"
    created_at: str


class KnowledgeBaseInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    collection_name: str
    status: str = "empty"
    document_count: int = 0
    chunk_count: int = 0
    image_asset_count: int = 0
    created_at: str
    updated_at: str
    documents: list[KnowledgeBaseDocument] = Field(default_factory=list)


class KnowledgeBaseListResponse(BaseModel):
    knowledge_bases: list[KnowledgeBaseInfo]


class KnowledgeBaseResponse(BaseModel):
    knowledge_base: KnowledgeBaseInfo
    message: str


class KnowledgeBaseIngestRequest(BaseModel):
    force_ocr: bool = False
    create_collection: bool = True
    drop_existing: bool = False
    use_model_descriptions: bool = False


class KnowledgeBaseJobInfo(BaseModel):
    id: str
    knowledge_base_id: str
    status: str
    current_step: str
    progress: float = 0.0
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str
    updated_at: str


class KnowledgeBaseJobResponse(BaseModel):
    job: KnowledgeBaseJobInfo
