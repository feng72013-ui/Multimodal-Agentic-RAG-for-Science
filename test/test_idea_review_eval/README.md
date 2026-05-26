# Idea Review Evaluation

This directory validates Stage 3 idea review reports without calling Milvus or any LLM.

Run from the project root:

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_idea_review_eval.evaluate
```

The default sample set contains 20 research ideas. A report passes when it includes related work, a similarity matrix, scoring fields, gaps, innovation suggestions, experiment suggestions, and risk assessment.

