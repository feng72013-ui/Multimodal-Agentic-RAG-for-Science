# Intent Routing Evaluation

This directory validates the Stage 1 task router without calling Milvus or any LLM.

Run from the project root:

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_intent_eval.evaluate
```

The default sample set contains 50 queries:

- 10 `qa`
- 10 `idea_review`
- 10 `literature_summary`
- 10 `paper_compare`
- 10 `research_plan`

The acceptance threshold is `0.85` overall accuracy.

