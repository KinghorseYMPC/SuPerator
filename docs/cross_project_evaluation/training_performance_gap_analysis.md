# Training Performance Gap Analysis

本文档基于静态代码对比分析 SuPerator quick baseline（score: 77.874956）与 pdeagent
更长训练最高分（score: 200+）之间的差距来源。

本文件只提出工程差距和需要验证的因素。不写成具体调参攻略，不写出具体训练参数推荐。

## 1. 当前 SuPerator Quick Baseline Score

| 指标 | 值 |
|---|---|
| Score | 77.874956 |
| 提交类型 | Task 1 + Task 2 combined quick baseline |
| 训练配置 | `pdeagent_task1_adapter_smoke.yaml` / `pdeagent_task2_adapter_quick.yaml` |
| 训练轮次 | quick/smoke 级别（极低） |
| 平台验收 | 通过（2026-05-18） |

## 2. pdeagent 最新分数

| 指标 | 值 |
|---|---|
| 最高分 | 200+ |
| 训练轮次 | 更长训练（epochs 规模远超 quick） |
| 训练策略 | pushforward + scheduled sampling + sliding window + physics loss（可选） |

具体差值约 122 分。以下是可能导致差距的工程因素分析。

## 3. 当前 SuPerator Quick Config 限制

- **epochs**：极低（smoke/quick 级别，通常 1-20 轮）
- **max_batches**：极小（quick 模式通常限制 2-5 个 batch）
- **数据样本**：quick 模式可能仅使用部分样本
- **训练步数**：quick 模式总优化步数远低于完整训练

## 4. 可能导致分数差距的工程因素

### 4.1 训练轮次（epochs / optimization steps）

- **状态**：SuPerator quick config 的 epochs 远低于 pdeagent 默认 220
- **影响**：直接影响模型收敛程度
- **评估**：这是最可能的差距主要来源

### 4.2 Chunked Rollout 训练

- **状态**：SuPerator model adapter 已实现 ChunkedFNO1d，训练也使用 chunked 模式
- **影响**：chunk rollout 优于 direct 10→190 映射，SuPerator 已采用
- **评估**：已吸收，不是差距来源

### 4.3 Sliding Window 数据扩增

- **状态**：SuPerator dataset adapter 已实现 window-based 数据加载
- **影响**：quick 模式样本量可能受限，不足以覆盖完整的数据分布
- **评估**：部分因素 — window 逻辑已就绪，quick 样本数不足

### 4.4 Task 2 Nu Conditioning

- **状态**：SuPerator Task 2 adapter 已实现 provided_nu 训练 + estimated_nu 推理
- **影响**：Nu conditioning 对 Task 2 性能有显著影响
- **评估**：已吸收，但 quick 训练轮次限制了 nu_estimator 的准确性

### 4.5 Checkpoint Selection

- **状态**：SuPerator 目前使用 quick-cycle 的最后一个 checkpoint 或简单
  选择，没有 pdeagent 风格的 best_metric checkpoint 对比
- **影响**：非最优 checkpoint 可能降低分数
- **评估**：未吸收 — eval_checkpoint.py 是差距因素之一

### 4.6 Scoring-Aligned Evaluation

- **状态**：SuPerator scoring adapter 完整实现了 segment scores
- **影响**：训练时的 evaluation metric 与最终评分对齐
- **评估**：已吸收，不是差距来源

### 4.7 Inference Implementation

- **状态**：SuPerator inference adapter 支持 chunked rollout 或 step-by-step fallback
- **影响**：推理实现差异可能影响预测质量
- **评估**：已吸收，但 rollout detach 策略可能需要验证

### 4.8 Pushforward / Scheduled Sampling

- **状态**：SuPerator 未实现 multi_step_rollout_loss 或 scheduled sampling
- **影响**：可能限制模型的长程 rollout 稳定性
- **评估**：未吸收 — 需要实验验证其影响

### 4.9 Auxiliary Loss（spectral gradient, temporal difference）

- **状态**：SuPerator 未实现这些辅助损失
- **影响**：可能改善 shock 区域的保真度
- **评估**：未吸收 — 需要实验验证

### 4.10 Physics Loss

- **状态**：SuPerator 未实现 burgers_residual physics loss
- **影响**：可能增强物理一致性
- **评估**：未吸收 — 需要确认合规性后再评估

## 5. 已迁移因素

| 因素 | 状态 |
|---|---|
| ChunkedFNO1d 模型结构 | 已迁移 |
| WindowedBurgersDataset | 已迁移 |
| 评分对齐 (compute_segment_scores) | 已迁移 |
| Task 2 FiLM + nu_estimator | 已迁移 |
| Inference (autoregressive rollout) | 已迁移 |
| methodology.pdf | 已迁移 |
| code-log consistency | 已迁移 |
| Quick submission pipeline | 已迁移 |

## 6. 仍未迁移因素

| 因素 | 状态 | 优先级 |
|---|---|---|
| 完整训练轮次配置 | 未迁移（quick 限制） | P1-next |
| eval_checkpoint / checkpoint comparison | 未迁移 | P1-next |
| pushforward / scheduled sampling | 未迁移 | P2-later |
| auxiliary loss (spectral_gradient, temporal_difference) | 未迁移 | P2-later |
| physics_loss (burgers_residual) | 未迁移 | P2-later |
| LLM provenance (真实 API log) | 未迁移 | P1-next |

## 7. 需要实验验证的因素

以下因素需要通过受控实验验证其对分数的影响，不应仅凭静态分析做出结论：

1. **训练轮次**：验证从 quick epochs 增加到中等 epochs 时的分数提升
2. **训练样本量**：验证 full window dataset vs quick subset 的影响
3. **Checkpoint selection**：验证 eval_checkpoint-based best checkpoint
   vs last checkpoint
4. **Multi-step rollout loss**：验证 pushforward 对长期 rollout 稳定性的改善
5. **Auxiliary losses**：验证 gradient/temporal losses 对 shock 保真度的影响

## 8. 风险说明

- 本文不包含具体训练参数推荐（epochs 值、lr 值等）
- 本文不包含竞赛得分优化指引或具体训练参数调优指引
- 本文不包含模型架构选择建议
- 所有因素需要在受控实验中验证，不应假设其一定有效
- 训练轮次低是当前最可能的主要差距来源，但需要实验确认
