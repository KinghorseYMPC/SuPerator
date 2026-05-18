# Task 2 Adapter (A10.1)

## 本阶段目标

将 pdeagent Task 2 baseline 的 adapter 结构迁移到 SuPerator，完成 Task 2 模型、
数据、Nu 条件、推理接口的静态适配与 smoke 测试准备。

本阶段只做 Task 2 adapter 结构迁移、shape smoke、配置、测试和文档。
不做真实训练，不生成正式 submission。

## Task 2 规则约束

1. Task 2 不得使用 Task 1 checkpoint
2. Task 2 不得使用 Task 1 数据
3. Task 2 不得使用预训练模型
4. Task 2 必须从头训练
5. Task 2 训练数据提供 Nu
6. Task 2 测试数据不提供 Nu
7. Task 2 推理时只能基于初始条件预测，不能依赖测试 Nu
8. 不允许使用数值求解器生成额外训练数据
9. 只能使用官方提供的数据
10. Task 2 推理时间必须控制在 2 分钟以内
11. Task 2 训练总时长需控制在 12 小时以内

## Task 2 与 Task 1 的隔离要求

| 维度 | Task 1 | Task 2 |
|---|---|---|
| 数据源 | task1_val.hdf5, task1_test.hdf5 | task2_part{0,1,2}_train.h5, task2_val.h5, task2_test.h5 |
| Checkpoint | Task 1 独立 | Task 2 独立，不加载 Task 1 checkpoint |
| Nu 条件 | 固定 nu=0.001 | 训练时有 Nu，测试时无 Nu |
| 模型 | FNOForecast1d (无 FiLM) | FNOForecast1d + FiLM + NuEstimator1d |
| Config | pdeagent_task1_adapter_smoke.yaml | pdeagent_task2_adapter_smoke.yaml |
| Adapter 文件 | model_adapter, dataset_adapter, inference_adapter (Task 1 部分) | model_adapter (Task 2 扩展), task2_dataset_adapter, task2_inference_adapter |

## 训练数据可用 Nu，测试数据不可用 Nu

- 训练阶段：HDF5 中包含 `nu` 字段，dataset adapter 在训练模式下返回 `nu`
- 推理阶段：测试 HDF5 中无 `nu` 字段，推理时使用 NuEstimator1d 从初始条件估计替代
- 模型在 `condition_source="provided_nu"` 模式下使用给定 Nu（训练），
  在 `condition_source="estimated_nu"` 模式下估计 Nu（推理）

## Task 2 推理时必须只使用初始条件

- 测试时只传入 `test.h5` 的 `tensor`（初始条件 shape (N, 10, 256)）
- 不传入任何 Nu 信息
- NuEstimator1d 从初始条件推断 Nu
- rollout 仅依赖 `model(x, nu=None)` → 内部调用 `nu_estimator(x)`

## pdeagent Task 2 baseline 资产来源

来自 `external_references/pdeagent_code_ref/code-ref/`：
- model.py: FNOForecast1d（含 nu_estimator）+ FiLM + ChunkedFNO1d
- dataset.py: WindowedBurgersDataset（支持 nu 字段）
- train.py: 训练循环（multi_step_rollout_loss）
- infer.py: checkpoint 推理（test 模式不传 cond）

## 本阶段只做 adapter smoke，不训练

- model adapter: 静态 shape 测试（synthetic input）
- dataset adapter: fake HDF5 测试
- inference adapter: 接口验证
- smoke 脚本: 端到端 shape 验证

## 后续 A10.2 quick train / predict / submission 计划

- A10.2: 真实数据训练（smoke 级别）
- A10.3: 推理 + submission 生成
- A10.4: 接入 experiment suite

## 禁止事项

- 不使用 Task 1 checkpoint
- 不使用 Task 1 数据
- 不使用 pdeagent config.yaml
- 不使用 pdeagent pack_submission.py
- 不生成额外数据
- 不调用数值求解器
- 不调用 Kaggle / SLURM / LLM API
- 不训练模型
