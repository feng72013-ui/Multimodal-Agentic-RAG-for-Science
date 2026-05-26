# Stage 4 Multi-Agent Evaluation

This evaluator checks the deterministic multi-agent workflow introduced in Stage 4.

Run:

```bash
/home/lf/conda/envs/my_ocr_env/bin/python -m test_multi_agent_eval.evaluate
```

The checks require:

- all six roles appear in `agent_trace`
- final report is generated
- related work and similarity matrix are populated
- score fields are present and in `[0, 1]`
- planner returns concrete next steps

The workflow is deterministic by design. It validates the agent contract before later replacing individual roles with LLM-backed agents.
