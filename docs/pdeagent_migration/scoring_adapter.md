# Scoring Adapter (A9.3)

## 为什么先适配 scoring

- scoring 是所有下游实验（训练验证、checkpoint 评估、结果对比）的基础
- pdeagent 的 `compute_segment_scores` 完整实现了 S3 的 `max(Lorentzian, Frechet)` 逻辑
- SuPerator 当前 `task1_metrics.segmented_score` 的 `score3_frechet` 为 `None`
- scoring 是纯函数，不依赖训练、推理、数据加载，适配风险最低
- 可以立即用已知答案测试验证，不需要等待模型适配

## 来源

- 参考源码：`external_references/pdeagent_code_ref/code-ref/utils.py`
- 核心函数：`compute_segment_scores`, `compute_rel_mse`, `compute_rmse`, `compute_frechet_distance`

## 实现方式

SuPerator clean-room adapter（`src/adapters/pdeagent/scoring.py`）：
- 使用 numpy 重新实现全部评分公式 — 不依赖 torch，不依赖 pdeagent 运行环境
- 不 import external_references 中的模块
- 遵循与 pdeagent 参考实现相同的数学公式和分段约定

## 支持的函数

| 函数 | 说明 |
|---|---|
| `rel_mse_by_segment(pred, target, eps, cap)` | 逐样本、逐时间步 capped Relative MSE |
| `rmse(pred, target)` | 全局 Root Mean Square Error |
| `frechet_distance_1d(pred, target)` | 轻量 Frechet-like 距离（基于空间 mean/std 统计量） |
| `lorentzian_score(pred, target)` | `100 / (1 + 10 * RMSE)` |
| `frechet_score(pred, target)` | `50 * exp(-FD^2)` |
| `segment_scores(pred, target)` | 完整 3 段评分，输入 shape `(N, 190, X)` |
| `compare_with_supertor_proxy(pred, target)` | 对比 adapter 与 SuPerator proxy 的输出差异 |

## 分段约定

匹配 pdeagent 参考实现：
- Segment 1: indices `[0:48]` — 48 steps, weight 25%
- Segment 2: indices `[48:96]` — 48 steps, weight 25%
- Segment 3: indices `[96:190]` — 94 steps, weight 50%

输入应为未来 190 步预测（不含初始条件），即 `(N, 190, X)`。调用方负责从 `(N, 200, X)` 全序列中切出后 190 步。

## 与 src/eval/task1_metrics.py 的关系

- **不修改** `src/eval/task1_metrics.py`
- adapter 的 `segment_scores` 是完整实现（含 Frechet），SuPerator proxy 的 `score3_frechet` 仍为 `None`
- `compare_with_supertor_proxy` 可对比两者输出
- 建议：在 A9.4/A10 中让 experiment suite 可选使用 adapter score

## 当前风险

1. **Frechet 实现为 lightweight proxy**（基于 mean/std 统计量）而非完整的 2-Wasserstein 距离
   - 这与 pdeagent 参考实现一致
   - 如需完整的 Frechet 距离（含协方差矩阵 sqrt），需后续增强
2. **SuPerator proxy 与 adapter 可能存在微小数值差异**
   - 因为 SuPerator proxy 使用 `np.asarray(dtype=float64)` 而 adapter 保持输入精度
   - `rel_mse` 的实现细节（sum over axis 的时机）可能产生 1e-6 量级的差异
3. **分段索引差异**
   - pdeagent 参考实现使用 `[0:48, 48:96, 96:190]`
   - Adapter 与之对齐
   - 如果比赛官方更新了分段定义，需同步更新

## 下一步

- A9.4: 将 segment_scores 集成到 ChunkedFNO1d model adapter 的 smoke test
- A10: experiment suite 可选使用 `segment_scores` 进行实验评估
- 后续：根据官方或 pdeagent 真实运行结果验证 Frechet 数值精度
