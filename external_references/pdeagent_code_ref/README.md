# pdeagent Code Reference (Isolated Import)

本目录是从 [pdeagent](../../../pdeagent/) 项目隔离导入的 P0 参考资产。

## 导入范围

### code-ref/ — 模型与训练基线

| 文件 | 内容 | 行数 |
|---|---|---|
| model.py | ChunkedFNO1d, SpectralConv1d, FNOBlock1d, FiLM, nu_estimator | ~290 |
| dataset.py | WindowedBurgersDataset, Normalizer, get_dataloaders | ~350 |
| train.py | 完整训练循环 + validate + checkpoint + physics_loss | ~480 |
| infer.py | 推理 + HDF5 输出 + Task 2 nu 推断 | ~200 |
| utils.py | compute_segment_scores, spectral_gradient_loss, Timer | ~230 |
| eval_checkpoint.py | 独立 checkpoint 评估工具 | ~250 |

### agent/ — Agent 架构参考

| 文件 | 内容 | 行数 |
|---|---|---|
| llm_client.py | LLM API 客户端 + 自动合规 JSONL 日志 | ~250 |
| tools.py | 11 个工具注册表 (ToolRegistry) | ~460 |
| phases.py | 四阶段科研闭环 (Literature/Diagnosis/Design/Experiment) | ~700 |
| orchestrator.py | 主编排器 + code-ref fallback + submission 生成 | ~530 |
| config.py | 配置管理 (dataclass + YAML) | ~130 |
| memory.py | ResearchMemory 持久化 + 实验追踪 | ~130 |

## 未导入的内容

以下 pdeagent 文件/目录**刻意排除**：

- `config.yaml` — 含 API key，安全风险
- `pack_submission.py` — 使用合成日志，不适合直接使用
- `run_agent.py` / `run_baseline.py` — 入口脚本，需适配
- `scripts/` — 工具脚本，需逐个评估
- `task1/` / `task2/` — 运行产物
- `output/` — 生成输出
- `.venv/` — Python 虚拟环境
- `data_and_sample_submission/` — 竞赛数据
- `AGENTS.md` / `AGENT_CODE_GUIDE.md` — 含竞赛策略

## 迁移状态

`migration_status: isolated_reference_only`

本目录是只读参考。后续阶段将逐步适配到 SuPerator 正式模块。
