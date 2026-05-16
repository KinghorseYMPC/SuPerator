# Model and Scoring Capability Comparison

## Task 1 模型

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 模型架构 | FNO1D（基础 SpectralConv1d + FNOBlock） | ChunkedFNO1d（SpectralConv1d + FNOBlock + chunk rollout） | pdeagent 更先进 |
| 参数量 | 取决于配置，~300k+ | 425k params (modes=24, width=64, depth=4) | pdeagent 已标定 |
| 自回归策略 | 直接 10→190 步映射 | chunk_size=10 逐步 rollout，共预测 190 步 | pdeagent 更符合 Markov 结构 |
| 残差连接 | 基础实现 | last_frame 残差连接（增强时间一致性） | pdeagent 有优化 |
| 空间坐标 | 未确认 | lift 层拼接 x-coordinate | pdeagent 有坐标增强 |
| 训练数据 | task1_val.hdf5 (100 trajectories) | task1_val.hdf5（完整 100 条） + sliding window | pdeagent 数据利用更充分 |
| 滑动窗口 | 无 | WindowedBurgersDataset（stride=1，大幅增加样本） | pdeagent 独有 |
| 验证方式 | 一步预测 loss | 完整 190 步自回归 rollout（非 teacher forcing） | pdeagent 更诚实 |

## Task 2 模型

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Task 2 支持 | **不支持** | ChunkedFNO1d + FiLM conditioning + nu_estimator | pdeagent 独有 |
| 参数量 | N/A | 492k params（FiLM 增加 ~67k） | pdeagent 已标定 |
| FiLM 条件化 | 无 | Feature-wise Linear Modulation 将 nu 注入每层 FNO block | pdeagent 独有 |
| Nu 推断 | N/A | CNN nu_estimator: Conv1d→GELU→AdaptiveAvgPool1d→Linear→log(nu) | pdeagent 独有 |
| 训练数据 | N/A | 3×1000=3000 多 nu 样本（task2_part{0,1,2}_train.h5） | pdeagent 已配置 |
| 验证数据 | N/A | 100 样本（含已知 nu） | pdeagent 已配置 |
| 测试时 Nu | N/A | 测试数据无 nu，通过 nu_estimator 从初始条件推断 | **正确匹配比赛规则** |

## 评分对齐

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| compute_segment_scores | 基础实现 | 完整实现，完全按比赛规则： | pdeagent 更准确 |
| S1 (0-47, 25%) | 可能存在 | `100 × exp(-20 × Rel-MSE)` | pdeagent 已验证 |
| S2 (47-95, 25%) | 可能存在 | `100 × exp(-10 × Rel-MSE)` | pdeagent 已验证 |
| S3 (95-190, 50%) | 可能存在 | `max(100/(1+10×RMSE), 50×exp(-FD²))` | pdeagent 实现 Lorentzian + Frechet |
| Rel-MSE clamp | 未确认 | clamp(max=5.0) 逐样本逐时间步 | pdeagent 正确 |
| Frechet distance | 未确认 | 基于 spatial mean/std 统计量的轻量实现 | pdeagent 已实现 |
| 总分公式 | 未确认 | `0.25×S1 + 0.25×S2 + 0.5×S3` | pdeagent 正确 |

## GPU / CPU 支持

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| GPU 自动检测 | 基础 `torch.cuda.is_available()` | `train.py` 内置 GPU 检测 + `--device` 参数 | 两者都有 |
| CPU fallback | 未明确测试 | train.py 支持 `--device cpu` | pdeagent 已验证 |
| CUDA 版本 | 未锁定 | PyTorch 2.6.0+cu124, CUDA 12.4 | pdeagent 环境明确 |
| 设备参数 | argparse 支持 | `--device` default=`"cuda" if torch.cuda.is_available() else "cpu"` | 一致 |

## Checkpoint 评估

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Checkpoint 保存 | `best_checkpoint.pt` | `best_checkpoint.pt` | 一致 |
| 独立 eval 工具 | `scripts/evaluate_persistence_task1.py` | `code-ref/eval_checkpoint.py` | 两者都有 |
| 多 checkpoint 管理 | registry 记录路径 | 按 iter 目录组织 | 两者都不同但可工作 |

## Smoke Test

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| 模型 smoke test | `scripts/smoke_fno1d_forward.py` | `agent/tools.py` 中 `quick_test_model` 工具 | 两者都有 |
| 测试内容 | 前向传播验证 | 创建小 batch，验证 input→output shape | pdeagent 的作为 Agent 工具更整合 |

## 物理信息损失

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| physics_loss | 未实现 | `train.py` 支持 `--use_physics_loss --physics_weight 1e-5` | pdeagent 独有 |
| Burgers 残差 | 无 | `burgers_residual()` 接口预留 | pdeagent 已预留 |
| Pushforward | 无 | `--use_pushforward` + schedule `[10, 30, 60]` | pdeagent 独有 |

## 官网基础分

| 维度 | SuPerator | pdeagent | 评估 |
|---|---|---|---|
| Task 1 baseline score | None（未提交） | ~66.3 (quick mode, 20 epochs) | pdeagent 已验证 |
| Task 2 baseline score | None（不支持） | ~62.9 (quick mode, 20 epochs) | pdeagent 已验证 |
| Task 1 满分潜力 | 未知 | 113/150（预估完整训练） | pdeagent 有预估 |
| Task 2 满分潜力 | N/A | 94/150（预估完整训练） | pdeagent 有预估 |

## 关键代码位置

| 模块 | pdeagent | SuPerator |
|---|---|---|
| 模型定义 | `code-ref/model.py` (11046 bytes) | `src/models/fno1d.py` (4390 bytes) |
| 数据集 | `code-ref/dataset.py` (12473 bytes) | `src/data/` (基础 HDF5 工具) |
| 训练脚本 | `code-ref/train.py` (17925 bytes) | `src/train/` (最小 scaffold) |
| 推理脚本 | `code-ref/infer.py` (7188 bytes) | `src/infer/` (基础) |
| 评分工具 | `code-ref/utils.py` (7916 bytes) | `src/eval/` (基础) |
| Checkpoint eval | `code-ref/eval_checkpoint.py` (8991 bytes) | `scripts/evaluate_persistence_task1.py` |

## 总结

- **pdeagent 在模型能力上全面领先**：ChunkedFNO1d、FiLM、nu_estimator、正确的评分函数、滑动窗口训练、物理损失
- **SuPerator 的模型是最小化基础实现**，不支持 Task 2，缺少关键优化
- **pdeagent 已获得比赛官网基础分**，验证了 pipeline 的可行性
- 如果以 SuPerator 为主，必须迁移 pdeagent 的模型和评分代码
