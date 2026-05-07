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
