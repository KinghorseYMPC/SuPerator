# Baseline Adapter (A9.4)

## 为什么选择 pdeagent baseline

- pdeagent code-ref 是比赛方官方开源示例，其 ChunkedFNO1d 架构和评分对齐已获比赛官网基础分（Task 1 ~66.3, Task 2 ~62.9）
- 该 baseline 正确实现了：自回归 chunk rollout、分段评分对齐、Task 2 FiLM/nu_estimator
- SuPerator 与 pdeagent 使用**相同数据**，不存在数据路径或 HDF5 key 不兼容问题
- 采用 pdeagent baseline 作为后续提分和 Agent 闭环的基础代码是最低风险路径

## 数据相同前提

- 两个项目均使用 `data_and_sample_submission/train_val_test_init/` 下的相同 HDF5 文件
- 输入 shape 一致：(N, 10, 256) → (N, 190, 256) → 提交 (N, 200, 256)
- 数据路径通过 config 传入，适配器不硬编码数据路径

## 本阶段范围

- 仅创建 **smoke-compatible baseline adapter**（model + dataset + inference）
- 不替换 SuPerator 主训练/推理/submission 流程
- 不做长训练
- 不生成正式 submission
- 不与 experiment suite 接驳

## Baseline Adapter 组成

| 模块 | 文件 | 功能 |
|---|---|---|
| Model Adapter | `src/adapters/pdeagent/model_adapter.py` | clean-room 最小 ChunkedFNO1d-compatible 模型 |
| Dataset Adapter | `src/adapters/pdeagent/dataset_adapter.py` | 窗口索引生成、HDF5 shape 探查、Normalizer |
| Inference Adapter | `src/adapters/pdeagent/inference_adapter.py` | 自回归 rollout 推理 |
| Smoke Config | `configs/pdeagent_baseline_smoke.yaml` | smoke test 参数 |
| Smoke Script | `scripts/smoke_pdeagent_baseline_adapter.py` | 端到端 smoke 验证 |

## 与 pdeagent code-ref 的关系

- **不 import external_references** 中的模块
- Model adapter 是 pdeagent ChunkedFNO1d 的 clean-room 最小复刻
- 使用相同的数学结构：SpectralConv1d + FNOBlock + chunked forward
- 当前实现为 smoke-compatible skeleton，后续可升级为完整 ChunkedFNO1d

## 不使用的内容

- pdeagent `config.yaml` — 使用 SuPerator 的 YAML configs/ 层级
- pdeagent `pack_submission.py` — 不生成 submission
- pdeagent `AGENTS.md` / `AGENT_CODE_GUIDE.md` — 含竞赛策略

## 后续接入计划

- A9.5：完整迁入 ChunkedFNO1d + dataset full + infer 完整版
- A10：将 baseline adapter 接入 experiment suite（可选使用 adapter score）

## 当前限制

- Model adapter 为最小 skeleton（SpectralConv1d + FNOBlock），非完整 ChunkedFNO1d
- Dataset adapter 仅提供窗口索引和 shape 探查，不是完整训练 DataLoader
- Inference adapter 基于简单 step-by-step rollout，未使用 chunk 优化
- 无 Task 2 FiLM / nu_estimator 支持（A9.6）
- 无 checkpoint 保存/加载
