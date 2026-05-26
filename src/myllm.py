from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_ROOT / "test"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
load_dotenv(PROJECT_ROOT / ".env")

for _path in (PROJECT_ROOT / "src", PROJECT_ROOT, TEST_ROOT, WORKSPACE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

DEFAULT_ALIBABA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL_PATH = Path("/home/lf/mount/LLM/model/Qwen3-Embedding-0.6B")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _api_key() -> str:
    return (
        _env("ALIBABA_API_KEY")
        or _env("DASHSCOPE_API_KEY")
        or _env("OPENAI_API_KEY")
        or "EMPTY"
    )


def _base_url() -> str:
    return _env("ALIBABA_BASE_URL") or _env("OPENAI_BASE_URL") or DEFAULT_ALIBABA_BASE_URL


def _zhipu_api_key() -> str:
    return _env("ZHIPU_API_KEY") or "EMPTY"


def _zhipu_base_url() -> str:
    return _env("ZHIPU_BASE_URL") or "https://open.bigapi.cn/v1"


llm = ChatOpenAI(
    model=_env("RECSYS_LLM_MODEL", "qwen-plus"),
    temperature=float(_env("RECSYS_LLM_TEMPERATURE", "0.3")),
    api_key=_api_key(),
    base_url=_base_url(),
)

multiModal_llm = ChatOpenAI(
    model=_env("RECSYS_MULTIMODAL_LLM_MODEL", "glm-4v-flash"),
    temperature=float(_env("RECSYS_MULTIMODAL_LLM_TEMPERATURE", "0.3")),
    api_key=_zhipu_api_key(),
    base_url=_zhipu_base_url(),
)

glm4_flash = ChatOpenAI(
    model=_env("RECSYS_GLM4_FLASH_MODEL", "glm-4-flash-250414"),
    temperature=float(_env("RECSYS_GLM4_FLASH_TEMPERATURE", "0.3")),
    api_key=_zhipu_api_key(),
    base_url=_zhipu_base_url(),
)


class LocalQwenLangchainEmbeddings(Embeddings):
    """LangChain embedding wrapper around the project's local Qwen3 embedder."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        from vector_ingest_pipeline.embeddings import LocalQwenTextEmbedder

        self.model_path = Path(model_path or _env("RECSYS_TEXT_MODEL_PATH") or DEFAULT_TEXT_MODEL_PATH)
        self.embedder = LocalQwenTextEmbedder(self.model_path, device=device or _env("RECSYS_TEXT_DEVICE") or None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts, batch_size=int(_env("RECSYS_EMBED_BATCH_SIZE", "8")))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


embedding = LocalQwenLangchainEmbeddings()
