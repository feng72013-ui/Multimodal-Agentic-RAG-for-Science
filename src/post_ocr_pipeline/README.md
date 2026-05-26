# Post-OCR Pipeline: Steps 6-9

This directory contains the post-OCR preparation code for the recommendation-paper RAG corpus.

Inputs:

- `data/processed_ocr/`: DotsOCR output.
- `data/papers/`: source paper folders and metadata.

Outputs:

- `data/processed_rag/clean_pages.jsonl`: cleaned page-level text.
- `data/processed_rag/chunks.jsonl`: RAG-ready text chunks.
- `data/processed_rag/image_assets.jsonl`: extracted image/table assets.
- `data/processed_rag/image_descriptions.jsonl`: image assets with descriptions.
- `data/processed_rag/assets/images/`: extracted images.

Run a small test:

```bash
cd /home/lf/mount/LLM/project/recommendate_project
PYTHONPATH=src python3 -m post_ocr_pipeline.run_steps_6_9 --only-topic 2026-05-19_rag_recommender --limit 2
```

Run all OCR outputs:

```bash
PYTHONPATH=src python3 -m post_ocr_pipeline.run_steps_6_9
```

Optional multimodal descriptions with Zhipu GLM-4.6V-Flash:

```bash
export ZHIPU_API_KEY="your-api-key"
PYTHONPATH=src python3 -m post_ocr_pipeline.run_steps_6_9 \
  --use-model-descriptions \
  --description-provider zhipu
```

The default model is `glm-4.6v-flash`, and the default endpoint is:

```text
https://open.bigmodel.cn/api/paas/v4/chat/completions
```

If no multimodal model is configured, image descriptions are generated from OCR captions, paper-level figure/table references, and nearby text.

Figure/table grounding:

- Extracts candidate labels such as `Fig. 2`, `Figure 3`, and `Table 1`.
- Prefers nearby OCR caption cells when available.
- Searches the full paper for sentences that cite the same figure/table label.
- Sends `image + caption + cited sentences + local context` to the multimodal model.
