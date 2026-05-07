# Competition Updates

## 2026-05-07: task logs format specification

Update summary: the organizers announced that recent submissions showed traces of manual Agent intervention. To support review, the required format for `task1_logs.log` and `task2_logs.log` has been standardized. The scoring system now checks this format, and non-compliant log files can cause a score of 0.

Local materials: `task_log_sample/`

Risk: every submitted task must include a log file that follows the official sample format. A structurally valid prediction can still score 0 if `task{N}_logs.log` is malformed or incomplete.

Project response:

- Add task log sample inventory: `docs/task_log_sample_inventory.md`
- Add format analysis: `docs/task_log_format_analysis.md`
- Add task log validator: `src/submission/validate_task_logs.py`
- Update dummy Task 1 log generation to emit JSON Lines records.
- Update submission validation so task log compliance is a hard check.
- Add `task-log-compliance` to Agent skills.
- Require all future training, experiment, and submission workflows to produce compliant task logs.

## 2026-05-07: Task 2 additional task and stronger log compliance

The preliminary round now includes Task 2, a multi-physics Burgers prediction task. Task 1 remains fixed at `Nu=0.001`. Task 2 training data provides `Nu` values, but Task 2 test data does not provide `Nu`; inference must predict only from the initial condition. Training may explore how to use `Nu`, such as conditioning or augmentation, but test-time prediction must not depend on hidden `Nu`.

Task 2 constraints:

- Task 2 training time is not scored, but total training duration must stay within 12 hours.
- Task 2 inference must finish within 2 minutes.
- Task 2 must not use Task 1 checkpoints.
- Task 2 must not use Task 1 data.
- Task 2 must not use pretrained models.
- Task 2 must be trained from scratch.
- Do not generate extra training data with numerical solvers.
- Use only official competition data.

The log requirement is now treated as a hard provenance and format constraint:

- Every line of `task{N}_logs.log` must be valid JSON.
- Every line must contain `timestamp` and `elapsed_seconds`.
- `timestamp` must be ISO 8601 and include timezone.
- The log should record the complete response from each Agent LLM call.
- The first-to-last timestamp span must not exceed 12 hours.
- Submitted `code/` must be traceable to the recorded LLM call history.
- The official local proxy is provided at `task_log_sample/openai-log/proxy.py` and can be run as `python proxy.py --port 8080 --target https://api.openai.com --log-dir ./logs`.

Project response:

- Strengthen `src/submission/validate_task_logs.py` to enforce JSONL, required fields, timezone, non-negative elapsed time, content fields, placeholder rejection, and 12-hour span.
- Treat provenance as a documented risk: structural validation cannot prove that `code/` fully corresponds to LLM call history.
- Mark logs from `src/agent/task_log_writer.py` as `development_summary_log`, not full API proxy captures.
- Prefer `api_proxy_llm_log` from the official proxy or another complete export of LLM call records for final competition submission.
- Document Task 2 isolation before any Task 2 implementation.
