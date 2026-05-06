# SuPerator

SuPerator 是面向“世界科学智能大赛 AI4S 智能体 CNS 挑战赛：任务 4 神经算子 PDE 智能体”的工程项目。本项目目标不是单独训练一个神经网络，而是构建一个面向 PDE 神经算子的科研 Agent，使其能够围绕规则理解、数据盘点、实验设计、模型训练、推理验证、提交生成和结论记录形成闭环。

## 比赛任务摘要

当前赛题围绕 1D Burgers PDE 神经算子预测任务。提交至少需要覆盖一个任务。每个任务必须同时提供：

- `task{N}_pred.hdf5`
- `task{N}_time.csv`
- `task{N}_logs.log`

预测文件 shape 必须为 `(N, 200, 256)`，前 10 个时间步必须与输入初始条件一致。推理时间是硬约束，应重点控制在 2 分钟以内。提交包中的 `code/` 目录不能为空。

## 当前阶段

当前处于 A0 / A 阶段：项目接管与工程基线。此阶段只建立目录、文档、盘点脚本、测试和依赖说明，不开始训练模型。

## 目录结构

```text
SuPerator/
├── AGENTS.md
├── README.md
├── SKILL.md
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

## 快速开始

```bash
python scripts/inspect_project.py
pytest -q
```

`scripts/inspect_project.py` 会扫描当前项目结构和官方材料元信息，并生成 `docs/project_inventory.md`。

## 数据放置说明

官方数据和示例提交材料应放在 `data_and_sample_submission/` 下。该目录包含大体积 HDF5/H5 文件，已被 `.gitignore` 排除。后续新增数据、checkpoint、预测文件、压缩包和日志输出也不得提交到 git。

## 后续计划

- A1：构建 dummy submission，验证提交格式、shape、初始条件复制和打包流程。
- A2：建立 Task 1 baseline，形成可复现实验配置、训练日志、推理指标和提交文件。
- B：围绕 Task 1 做模型结构、损失函数、推理速度和长时稳定性改进。
- C：建设科研 Agent 闭环，包括自动实验规划、结果分析、假设迭代和报告生成。

## 注意事项

- 不修改官方原始数据。
- 不提交数据集、checkpoint、预测文件、日志输出或大型压缩包。
- 每个实验必须留下配置、日志、指标和结论。
- 路径应使用配置或相对路径管理，避免写死个人机器路径。
- 修改后尽量运行相关测试。
