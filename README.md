# SuPerator

SuPerator 是面向“世界科学智能大赛 AI4S 智能体 CNS 挑战赛：任务 4 神经算子 PDE 智能体”的工程项目。本项目目标不是单独训练一个神经网络，而是构建一个面向 PDE 神经算子的科研 Agent，使其围绕规则理解、数据盘点、实验设计、模型训练、推理验证、提交生成和结论记录形成闭环。

## 比赛任务摘要

当前赛题围绕 1D Burgers PDE 神经算子预测任务。提交至少需要覆盖一个任务。每个任务必须同时提供：

- `task{N}_pred.hdf5`
- `task{N}_time.csv`
- `task{N}_logs.log`

预测文件 shape 必须为 `(N, 200, 256)`，前 10 个时间步必须与输入初始条件一致。推理时间是硬约束，应重点控制在 2 分钟以内。提交包中的 `code/` 目录不能为空。

## 当前阶段

当前处于 A1.6 / A 阶段：建立 skill 自演化与外部 skill 引入机制。A0 工程基线、A1 Task 1 dummy submission pipeline 和 A1.5 skill 迁移已完成；本阶段不训练模型，不实现 FNO，不修改官方数据。

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
- `.agents/skills/testing-checklist/SKILL.md`：测试清单，定义 A0、A1、提交校验和未来 A2 测试要求。
- `.agents/skills/git-workflow/SKILL.md`：git 工作流，定义 status、staged 检查、commit 粒度和大文件禁提交规则。
- `.agents/skills/skill-maintenance/SKILL.md`：维护本地 skill，保证项目阶段推进时规程同步更新。
- `.agents/skills/external-skill-intake/SKILL.md`：安全吸收外部 skill，要求候选记录、许可证检查、项目适配、测试和提交。

## 快速开始

```bash
python scripts/inspect_project.py
python scripts/make_dummy_task1_submission.py
python scripts/validate_submission.py
pytest -q
```

`scripts/inspect_project.py` 会扫描当前项目结构和官方材料元信息，并生成 `docs/project_inventory.md`。`scripts/make_dummy_task1_submission.py` 会生成 Task 1 persistence dummy submission；`scripts/validate_submission.py` 会校验提交结构、HDF5 shape、前 10 步一致性和 `time.csv`。

## 数据放置说明

官方数据和示例提交材料应放在 `data_and_sample_submission/` 下。该目录包含大体积 HDF5/H5 文件，已被 `.gitignore` 排除。后续新增数据、checkpoint、预测文件、压缩包和日志输出也不得提交到 git。

## 后续计划

- A1.6：建立 skill 自演化与外部 skill 引入机制。
- A2：建立 Task 1 baseline，形成可复现实验配置、训练日志、推理指标和提交文件。
- B：围绕 Task 1 做模型结构、损失函数、推理速度和长时稳定性改进。
- C：建设科研 Agent 闭环，包括自动实验规划、结果分析、假设迭代和报告生成。

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
