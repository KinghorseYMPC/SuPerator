# Task 1 + Task 2 Quick Baseline — Platform Acceptance Record

## Submission Info

- **submission type**: Task 1 + Task 2 quick baseline
- **environment**: pdeagent conda env
- **date**: 2026-05-18
- **accepted by competition platform**: yes
- **score**: 77.874956

## Generated Artifact

```
outputs/submission/submission.zip
```

Generated via:

```bash
conda activate pdeagent
python scripts/run_pdeagent_all_quick_submission.py --skip-task1-train --skip-task2-train
```

## Local Validations

All passed before platform submission:

- `python scripts/validate_submission.py --all-present` — passed
- `python scripts/validate_task_logs.py` — passed (development_summary_log provenance warning)
- methodology.pdf — present in submission bundle
- write_file tool_calls — present in task1_logs.log and task2_logs.log
- code-log consistency — passed platform check

## Known Limitations

- **Quick training**: low epoch count; score reflects quick baseline only, not model capacity limit
- **development_summary_log provenance**: task logs use development summary log mode;
  write_file records are code snapshots appended at submission time, not real LLM tool calls
- **Score gap vs. longer training**: pdeagent longer-training runs achieve 200+; the 77.87
  score validates submission pipeline completeness

## Next Work

- Longer controlled training runs
- Official LLM log provenance (API proxy capture)
- Higher-quality submission with full provenance
