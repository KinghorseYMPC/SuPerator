# Code Inventory and Cleanup Candidates

This document inventories the main scripts and source modules, marks their
status, identifies potential duplication, flags low-coverage areas, and lists
candidates for consolidation or deprecation. **No code will be deleted in this
stage.** All suggestions are for future consideration.

## Script Inventory

| Script | Status | Notes |
|---|---|---|
| `check_text_encoding.py` | active | Core validation tool |
| `pre_push_audit.py` | active | Core validation tool |
| `validate_task_logs.py` | active | Core validation tool |
| `validate_submission.py` | active | Core validation tool (thin wrapper) |
| `make_dummy_task1_submission.py` | active | Core workflow, thin wrapper |
| `make_task1_trained_submission.py` | active | Trained submission generator, thin wrapper |
| `train_task1_minimal.py` | active | Local training entry, thin wrapper |
| `one_batch_train_task1.py` | active | Quick forward-pass smoke test |
| `smoke_fno1d_forward.py` | active | FNO model smoke test |
| `inspect_project.py` | active | Project inventory smoke check |
| `inspect_task1_hdf5.py` | active | HDF5 data inspection |
| `inspect_task_log_sample.py` | active | Log sample inspection |
| `evaluate_persistence_task1.py` | active | Task 1 persistence evaluation |
| `check_compute_environment.py` | active | Compute environment check |
| `run_task1_auto_loop.py` | active | A5 controller, thin wrapper |
| `summarize_task1_auto_loop.py` | active | A5 summary reporter |
| `run_task1_experiment_suite.py` | active | A6 controller |
| `compare_task1_results.py` | active | A6 result comparison |
| `finalize_best_task1_result.py` | active | A6 finalization |
| `run_task1_full_auto_experiment.py` | active | A7 controller |
| `summarize_task1_full_auto.py` | active | A7 summary reporter |
| `create_remote_manifest.py` | active | SLURM manifest generation |
| `create_remote_package_plan.py` | active | SLURM package plan |
| `render_slurm_jobs.py` | active | SLURM job rendering |
| `parse_slurm_min_train_result.py` | active | SLURM result parsing |
| `create_kaggle_dataset_package.py` | active | Kaggle dataset staging |
| `create_kaggle_kernel_package.py` | active | Kaggle kernel staging |
| `run_kaggle_task1_min_train.py` | active | Kaggle full orchestration |
| `parse_kaggle_min_train_output.py` | active | Kaggle output parsing |
| `adopt_kaggle_task1_result.py` | active | Kaggle result adoption |
| `finalize_kaggle_task1_submission.py` | active | Kaggle submission finalization |
| `scripts/kaggle/run_task1_min_train.py` | candidate for consolidation | Duplicate of `run_kaggle_task1_min_train.py` entry point; may be an older or alternative path |
| `scripts/knowledge/audit_kb_compliance.py` | active | Knowledge-base compliance audit |
| `scripts/knowledge/create_concept_entry.py` | active | Concept entry generation |
| `scripts/knowledge/create_literature_metadata.py` | active | Literature metadata creation |
| `scripts/knowledge/generate_literature_card.py` | active | Literature card generation |
| `scripts/knowledge/validate_metadata_examples.py` | active | Metadata example validation |
| `scripts/knowledge/validate_taxonomy_usage.py` | active | Taxonomy usage validation |

## Source Module Inventory

| Module | Status | Notes |
|---|---|---|
| `src/data/hdf5_utils.py` | active | HDF5 read/write helpers |
| `src/data/task1_dataset.py` | active | Task 1 PyTorch dataset |
| `src/models/fno1d.py` | active | FNO-1D model architecture |
| `src/train/checkpointing.py` | active | Checkpoint save/load |
| `src/train/train_task1_minimal.py` | active | Minimal training loop |
| `src/infer/rollout.py` | active | Autoregressive rollout inference |
| `src/eval/task1_metrics.py` | active | Evaluation metrics |
| `src/experiment/registry.py` | active | Experiment registry (JSONL) |
| `src/experiment/backend_config.py` | active | Backend config parsing |
| `src/experiment/remote_manifest.py` | active | Remote manifest generation |
| `src/experiment/remote_package_plan.py` | active | Remote package planning |
| `src/experiment/kaggle_package_plan.py` | active | Kaggle package planning |
| `src/experiment/kaggle_adoption.py` | active | Kaggle result adoption |
| `src/experiment/task1_auto_loop.py` | active | A5 auto-loop logic |
| `src/experiment/config_generation.py` | active | A6 config generation |
| `src/experiment/backend_selector.py` | active | A6 backend selector |
| `src/experiment/result_comparison.py` | active | A6 result comparison |
| `src/experiment/command_runner.py` | active | A7 subprocess runner |
| `src/experiment/slurm_executor.py` | active | A7 SLURM executor |
| `src/experiment/kaggle_executor.py` | active | A7 Kaggle executor |
| `src/experiment/local_executor.py` | active | A7 local executor |
| `src/experiment/full_auto_controller.py` | active | A7 full-auto controller |
| `src/submission/make_dummy_task1_submission.py` | active | Dummy submission generator |
| `src/submission/make_task1_trained_submission.py` | active | Trained submission generator |
| `src/submission/validate_submission.py` | active | Submission format validator |
| `src/submission/validate_task_logs.py` | active | Task log validator |
| `src/submission/package_submission.py` | active | Submission packaging |
| `src/agent/task_log_writer.py` | active | Task log writer utility |
| `src/knowledge/__init__.py` | active | Knowledge package init |
| `src/knowledge/literature_metadata.py` | active | Literature metadata handling |
| `src/knowledge/literature_card.py` | active | Literature card generation |
| `src/knowledge/concept_entry.py` | active | Concept entry generation |
| `src/knowledge/taxonomy.py` | active | Taxonomy handling |
| `src/knowledge/metadata_schema.py` | active | Metadata schema |

## Potential Duplication Analysis

### Task 1 Controllers: auto loop vs. experiment suite vs. full auto

| Aspect | A5 Auto Loop | A6 Experiment Suite | A7 Full Auto |
|---|---|---|---|
| Backend selection | Kaggle preferred | SLURM > Kaggle > local | SLURM > Kaggle > local |
| Config generation | None | Yes, from suite YAML | No |
| Result comparison | No | Yes | Yes (shares A6 comparison) |
| Finalization | Internal | Separate script | Separate script |
| Overlap | Kaggle execution duplicated across all three | Backend selection logic similar to A7 | Wraps both A5 and A6 patterns |

**Recommendation**: The three controllers represent an evolution (A5 → A6 → A7)
but share substantial backend orchestration, adoption, and validation logic.
Consider consolidating into a single controller with mode selection after A7
stabilizes. Do not remove A5/A6 until A7 covers all use cases.

### Kaggle Parse / Adopt / Finalize

`parse_kaggle_min_train_output.py`, `adopt_kaggle_task1_result.py`, and
`finalize_kaggle_task1_submission.py` are called sequentially by the A5, A6, and
A7 controllers. The three-step pipeline is intentional (parse → adopt →
finalize) but the orchestration logic is duplicated across controllers.

### SLURM Package Plan / Manifest / Job Rendering

`create_remote_manifest.py`, `create_remote_package_plan.py`, and
`render_slurm_jobs.py` are distinct steps, each with its own script. The A7
SLURM executor orchestrates them internally. The standalone scripts remain useful
for manual dry-runs.

### Multiple Submission Scripts

`make_dummy_task1_submission.py` and `make_task1_trained_submission.py` serve
different purposes (dummy validation vs. trained submission). The underlying
`src/submission/` modules share packaging logic.

### `scripts/kaggle/run_task1_min_train.py` vs `scripts/run_kaggle_task1_min_train.py`

The `scripts/kaggle/` subdirectory contains `run_task1_min_train.py` which
appears to be an older or alternative entry point. The root-level
`scripts/run_kaggle_task1_min_train.py` is the primary entry point referenced in
documentation.

## Test Coverage Gaps

| Area | Test Count | Coverage Assessment |
|---|---|---|
| Core model/data/training | 6 test files | Good |
| Submission validation | 5 test files | Good |
| Experiment/A5/A6/A7 | 15 test files | Good |
| SLURM backend | 4 test files | Adequate (remote execution not tested) |
| Kaggle backend | 6 test files | Adequate (API calls mocked) |
| Knowledge-base | 6 test files | Adequate (content quality not tested) |
| Script-level entry points | ~15 test files | Good (thin wrappers, core tested separately) |
| `scripts/knowledge/audit_kb_compliance.py` | Covered by `test_knowledge_compliance_audit.py` | Adequate |
| `scripts/pre_push_audit.py` | Covered by `test_pre_push_audit.py` | Good |
| **Evaluation metrics** (`src/eval/`) | 1 test file | Light — `evaluate_persistence_task1.py` script not separately tested |
| **HDF5 inspection scripts** | No dedicated tests | Light — inspection scripts are read-only, low risk |
| **`check_compute_environment.py`** | 1 test file | Adequate |
| **`inspect_project.py`** | No dedicated test | Light — read-only inventory script |

## Candidates for Future Consolidation

| Candidate | Reason | Priority |
|---|---|---|
| Three Task 1 controllers (A5, A6, A7) | Shared backend orchestration, adoption, validation | P1 (after A7 proves stable) |
| `scripts/kaggle/run_task1_min_train.py` | Possible duplicate of root-level script | P2 |
| Kaggle adoption pipeline (parse/adopt/finalize) | Three-step pipeline could be a single module with modes | P2 |
| Multiple thin wrapper scripts | Many scripts are 5-15 line wrappers around `src/` modules; could consolidate | P3 |

## Candidates for Future Deprecation

| Candidate | Reason | Priority |
|---|---|---|
| `scripts/kaggle/run_task1_min_train.py` | Redundant with root-level `run_kaggle_task1_min_train.py` | P2 |
| Individual SLURM preparation scripts | A7 SLURM executor already orchestrates these internally | P3 (keep for manual dry-runs) |

## No Deletion Policy

**This stage does not delete any code.** All candidates listed above are for
discussion and future planning only. Before any removal:

1. Verify no active workflow references the script.
2. Check git history for the script's original purpose.
3. Ensure test coverage exists for the replacement path.
4. Archive the script reference in `docs/project_stage_history.md`.
