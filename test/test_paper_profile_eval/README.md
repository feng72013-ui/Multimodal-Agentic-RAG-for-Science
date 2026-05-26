# Paper Profile Evaluation

This directory validates Stage 2 paper-level extraction outputs.

Generate profiles:

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m paper_profile_pipeline.extract
```

Evaluate generated artifacts:

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_paper_profile_eval.evaluate
```

Default acceptance thresholds:

- required field fill rate >= `0.80`
- average quality score >= `0.60`
- citation edge count >= `1`

