"""Project-local Python startup hooks.

This makes DotsOCR's custom vLLM model registration visible to worker
subprocesses without patching the installed `vllm` entrypoint.
"""

from __future__ import annotations

import importlib
import os


def _register_dotsocr_vllm_model() -> None:
    if os.environ.get("DOTSOCR_VLLM_AUTO_REGISTER") != "1":
        return

    model_package_name = os.environ.get("DOTSOCR_MODEL_PACKAGE_NAME", "DotsOCR")
    importlib.import_module(f"{model_package_name}.modeling_dots_ocr_vllm")


_register_dotsocr_vllm_model()
