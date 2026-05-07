# SuPerator

SuPerator 是面向“世界科学智能大赛 AI4S 智能体 CNS 挑战赛：任务 4 神经算子 PDE 智能体”的工程项目。本项目目标不是单独训练一个神经网络，而是构建一个面向 PDE 神经算子的科研 Agent，使其围绕规则理解、数据盘点、实验设计、模型训练、推理验证、提交生成和结论记录形成闭环。

## 比赛任务摘要

当前赛题围绕 1D Burgers PDE 神经算子预测任务。提交至少需要覆盖一个任务。每个任务必须同时提供：

- `task{N}_pred.hdf5`
- `task{N}_time.csv`
- `task{N}_logs.log`

预测文件 shape 必须为 `(N, 200, 256)`，前 10 个时间步必须与输入初始条件一致。推理时间是硬约束，应重点控制在 2 分钟以内。提交包中的 `code/` 目录不能为空。

## 当前阶段

当前处于 A3.7 阶段：基于外部自动化科研资源的只读调研，吸收通用科研 Agent 工作流能力，并继续保持预置上下文合规边界。

## 目录结构

```text
SuPerator/
├── .agents/
│   ├── external_skill_intake_log.md
│   ├── skill_registry.yaml
│   └── skills/
│       ├── README.md
│       ├── debug-and-fix/
│       ├── external-skill-intake/
│       ├── git-workflow/
│       ├── project-onboarding/
│       ├── safe-code-change/
│       ├── skill-maintenance/
│       └── testing-checklist/
├── AGENTS.md
├── README.md
├── guideline.md
├── requirements.txt
├── configs/
├── data_and_sample_submission/   # 官方材料，本地保留，不提交 git
├── docs/
├── experiments/                  # 实验工作区，仅保留 .gitkeep
├── outputs/                      # 运行输出，仅保留 .gitkeep
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── submission/
├── scripts/
├── src/
│   ├── agent/
│   ├── data/
│   ├── eval/
│   ├── experiment/
│   ├── infer/
│   ├── models/
│   ├── submission/
│   └── train/
└── tests/
```

## Agent 技能体系

项目工作技能统一位于 `.agents/skills/`。根目录 `SKILL.md` 已迁移并删除，不再作为主技能文件使用。

- `.agents/skills/README.md`：skill 总索引，说明当前 skill 列表、用途、更新规则和外部引入规则。
- `.agents/skill_registry.yaml`：skill 注册表，记录 skill 名称、路径、用途、状态、来源和 review 日期。
- `.agents/external_skill_intake_log.md`：外部 skill 调研和吸收日志，记录候选来源、许可证、决策和本地适配结果。
- `.agents/skills/project-onboarding/SKILL.md`：项目接管，确认阶段、规则、数据、commit、测试和下一步。
- `.agents/skills/safe-code-change/SKILL.md`：安全代码修改，保证小步、可回滚、可测试。
- `.agents/skills/debug-and-fix/SKILL.md`：调试和修复，覆盖报错复现、定位、补丁、回归测试和根因记录。
- `.agents/skills/testing-checklist/SKILL.md`：测试清单，定义通用项目、日志和提交校验命令。
- `.agents/skills/git-workflow/SKILL.md`：git 工作流，定义 status、staged 检查、commit 粒度和大文件禁提交规则。
- `.agents/skills/skill-maintenance/SKILL.md`：维护本地 skill，保证项目阶段推进时规程同步更新。
- `.agents/skills/external-skill-intake/SKILL.md`：安全吸收外部 skill，要求候选记录、许可证检查、项目适配、测试和提交。
- `.agents/skills/task-log-compliance/SKILL.md`：通用 JSONL task log 格式与 provenance 校验流程。
- `.agents/skills/data-checkpoint-isolation/SKILL.md`：通用数据、checkpoint、配置、输出和推理输入隔离流程。
- `.agents/skills/research-agent-loop/SKILL.md`：通用科研 Agent 闭环，覆盖观察、假设、实验、评估和记录。
- `.agents/skills/experiment-recording/SKILL.md`：通用实验记录规范，覆盖假设、配置、diff、指标、失败和结论。
- `.agents/skills/external-research-review/SKILL.md`：只读审查外部自动化科研资源，覆盖许可证、适配和拒绝规则。

## 快速开始

```bash
python scripts/inspect_project.py
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
pytest -q
```

`scripts/inspect_project.py` 会扫描当前项目结构和官方材料元信息，并生成 `docs/project_inventory.md`。`scripts/make_dummy_task1_submission.py` 会生成本地 dummy submission；`scripts/validate_submission.py` 会校验提交结构、HDF5 shape、前 10 步一致性、`time.csv`、JSONL log 和 code bundle 禁止项。

## 数据放置说明

官方数据和示例提交材料应放在 `data_and_sample_submission/` 下。该目录包含大体积 HDF5/H5 文件，已被 `.gitignore` 排除。后续新增数据、checkpoint、预测文件、压缩包和日志输出也不得提交到 git。

## 后续计划

后续阶段应由 Agent 在合规上下文内根据比赛规则和运行结果自行生成执行计划。README 只记录项目结构、运行入口和治理边界，不预置任务执行策略。

## 注意事项

- 不修改官方原始数据。
- 不提交数据集、checkpoint、预测文件、日志输出或大型压缩包。
- 不无审查复制外部 skill。
- 不执行外部仓库脚本。
- 不把外部仓库整体 vendor 到本项目。
- 每个实验必须留下配置、日志、指标和结论。
- 路径应使用配置或相对路径管理，避免写死个人机器路径。
- 修改提交文件生成逻辑后必须运行 submission 校验。
- 修改后尽量运行相关测试。

## A2.5 task log compliance

On 2026-05-07 the competition announced a strict format requirement for `task1_logs.log` and `task2_logs.log`. Local official samples must be placed in `task_log_sample/`. This directory is treated as local official material and is ignored by git.

Before packaging or submitting, run:

```bash
python scripts/validate_task_logs.py
python scripts/validate_submission.py
```

All submission-related work must follow `.agents/skills/task-log-compliance/SKILL.md`.

## A3.5 competition rule hardening

A3.5 records the latest task log compliance updates. See
`docs/log_compliance_strategy.md` for the distinction between
`development_summary_log` and `api_proxy_llm_log`.

## A3.6 compliance boundary

A3.6 defines the preloaded context boundary for project governance.

- `docs/preloaded_context_policy.md` explains what may live in skills, competition clarification docs, wiki pages, Agent-generated artifacts, and final submission code bundles.
- `docs/competition_clarifications.md` records neutral rules and format requirements only.
- `.agents/skills/` is reserved for generic work skills and must not contain task-specific execution strategy.
- `docs/wiki/`, if extended, is reserved for broad PDE, neural operator, and scientific computing knowledge.
- Submission `code/` bundles exclude human-preloaded strategy documents by default.

## A3.7 external auto-research resource review

A3.7 records read-only review of external automated research Agent resources. See
`docs/external_auto_research_tools_intake.md` for the reviewed source, accepted
generic workflow ideas, and rejected or deferred reuse. External resources may
improve generic skills only; they must not become task execution strategy.
