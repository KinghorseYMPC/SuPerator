# Risk Register

| # | Risk | Project | Severity | Evidence | Mitigation | Decision Impact |
|---|---|---|---|---|---|---|
| 1 | **Log provenance risk — pack_submission.py 生成合成日志** | pdeagent | **High** | `pack_submission.py` 第 6 行：`base = datetime(2026, 5, 17, 8, 0, 0, tzinfo=timezone.utc)`，所有日志条目使用硬编码时间戳和预写内容，不读取 Agent 真实日志 | 必须重写 pack_submission.py，从 `task{N}/task{N}_logs.log` 读取 Agent 真实运行日志 | 无论哪个项目为主，都必须修复 |
| 2 | **config.yaml hardcoded API key** | pdeagent | **High** | `config.yaml` 第 5 行包含硬编码 DeepSeek API key（已审计确认，key 已遮蔽） | 立即从 config.yaml 删除 key，改用环境变量 `OPENAI_API_KEY`；确保 `config.yaml` 已被 `.gitignore` 覆盖或 git rm --cached | 必须在任何代码合并前处理 |
| 3 | **Code-log consistency: trust but verify** | pdeagent | **High** | `validate_submission.py` 的 code-log 检查提取 log 中的 write_file content 与 zip 中文件对比，但 `pack_submission.py` 从 `code/` 读取文件并写入 log，形成循环论证：log content = 当前 code/ 中的文件内容，一致性总是通过 | 改为：Agent 生成代码时通过 `write_file` 工具产生 log，Infer/train 使用该代码后，code-log 验证应检查 log 中的内容而非读取当前 code/ | 必须重新设计 code-log 一致性流程 |
| 4 | **development_summary_log gap** | SuPerator | **High** | `task_definition.md` 和 `validate_task_logs.py` 明确标注 provenance 为 `development_summary_log`，无法证明是真实 Agent 运行产物 | 迁移 pdeagent 的 `llm_client.py` 实现自动 API 日志捕获 | 通过迁移 pdeagent 的 llm_client 解决 |
| 5 | **SuPerator 不支持 Task 2** | SuPerator | **High** | `src/models/fno1d.py` 仅为基本 FNO1D，无 FiLM，无 nu_estimator，无 Task 2 数据加载逻辑 | 迁移 pdeagent 的 code-ref (model/dataset/infer/utils) | 强烈影响决策（倾向迁移 pdeagent 模型代码） |
| 6 | **pdeagent AGENTS.md 含竞赛执行策略** | pdeagent | **Medium** | `pdeagent/AGENTS.md` (18862 bytes) 和 `AGENT_CODE_GUIDE.md` (34440 bytes) 包含大量竞赛特定策略指导 | 不迁移这些文件；SuPerator 的合规边界禁止此类内容 | 任何合并方案都不应包含这些文件 |
| 7 | **SuPerator engineering overcomplexity** | SuPerator | **Medium** | A7 全自动控制器依赖 Kaggle executor + SLURM executor + local executor + command_runner + config generation + backend selector + result comparison + summary + finalize — 链条长 | 评估是否可以简化；本阶段不删除代码 | 如果以 pdeagent 为主，暂不迁移复杂的多后端系统 |
| 8 | **pdeagent 缺乏协作治理** | pdeagent | **Medium** | 无 CONTRIBUTING.md、无分支策略、无 pre-push audit、无 tests、无 code review 流程 | 从 SuPerator 迁移治理文档和工具 | 如果以 pdeagent 为主，必须补充 |
| 9 | **Kaggle / SLURM queue/network risk** | SuPerator | **Medium** | A7 控制器支持远程后端但依赖网络和队列可用性；`engineering_execution_log.md` 记录 SLURM 端到端未完全测试 | 本地 GPU 作为首选 fallback；Kaggle/SLURM 作为可选增强 | 保持 local-first 策略 |
| 10 | **Task 2 不能使用 Task 1 checkpoint** | 两者 | **Medium** | 官方规则：Task 1 和 Task 2 必须独立训练，不能复用 checkpoint；pdeagent 的 code-ref 代码共享 `model.py` 但通过 `task` 参数区分 | 确保 train.py 和 infer.py 正确处理 task 参数，不跨任务加载 checkpoint | 需要在代码和文档中明确 |
| 11 | **Extra data / numeric solver prohibition** | 两者 | **Medium** | 官方规则禁止使用外部数据和数值求解器；pdeagent 的 physics_loss 使用 Burgers 残差可能被误解 | 确认 physics_loss 的计算不使用外部 solver，仅使用模型输出计算残差 | 文档中明确说明 physics_loss 的来源 |
| 12 | **Official submission format risk** | 两者 | **Medium** | 提交格式变更、HDF5 key 名称变更等可能与官方最新要求不一致 | 定期对照官方 sample submission 验证 | 持续关注官方公告 |
| 13 | **Time limit risk (Task 2 infer ≤120s)** | pdeagent | **Medium** | pdeagent README 记录 Task 2 推理 ~25s (quick mode, 20 epochs)，但更大模型或更多 epochs 可能接近限制 | 在 validate_submission 中已有检查；训练前评估推理时间 | 监控推理时间，必要时优化 |
| 14 | **Hidden large files risk** | pdeagent | **Medium** | 项目根目录有 `data_and_sample_submission.zip` (1.1GB)，`.venv/` 目录巨大 | 确保 `.gitignore` 覆盖这些路径；运行 pre_push_audit 验证 | 必须不提交这些文件 |
| 15 | **config.yaml 被 run_baseline.py 原地覆盖** | pdeagent | **Low** | `run_baseline.py` 第 91-96 行：`yaml.safe_load(cfg)` → 修改 → `yaml.dump(cfg, f)` 直接覆盖原文件 | 改为使用临时配置文件或 `--dry-run` 模式 | 低优先级，但影响可复现性 |
| 16 | **pdeagent .venv 在项目内** | pdeagent | **Low** | `.venv/` 目录在项目根下，可能被误提交 | 确保 `.gitignore` 包含 `.venv/` | 低风险，已有 .gitignore |
| 17 | **API key rotation risk** | pdeagent | **High** | hardcoded key 已在此审计中曝光，需立即使其失效 | 联系 DeepSeek 平台 revoke 该 key，生成新 key 并通过环境变量使用 | **必须在任何操作前处理** |

## Risk Summary

| Severity | Count | Items |
|---|---|---|
| **High** | 6 | Log provenance (#1), API key leak (#2), Code-log circularity (#3), development_summary_log (#4), Task 2 不支持 (#5), API key rotation (#17) |
| **Medium** | 8 | AGENTS 策略 (#6), Overcomplexity (#7), 缺乏治理 (#8), Kaggle/SLURM (#9), Task 2 checkpoint (#10), Extra data (#11), Format (#12), Time limit (#13), Large files (#14) |
| **Low** | 2 | config.yaml 覆盖 (#15), .venv (#16) |
