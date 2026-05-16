"""
四阶段科研闭环实现

1. Literature Phase   - 文献解析与逻辑解构
2. Diagnosis Phase    - 瓶颈诊断与假设提出
3. Design Phase       - 自主设计与代码演进
4. Experiment Phase   - 实验验证与科学迭代
"""
import json
import os
import re
from datetime import datetime
from typing import Any

from .llm_client import LLMClient
from .memory import ResearchMemory, ExperimentRecord
from .tools import registry


SYSTEM_PROMPT = """# PDE Neural Operator Research Agent

你是一个自主科研智能体，目标是零人工干预下完成 1D Burgers 方程神经算子的研究、改进与验证。

## 身份与原则

- **严谨**: 每个假设必须有理论依据或数据支撑
- **务实**: 优先选择工程可行、效果可验证的方案
- **记录**: 完整记录思考链路与实验轨迹
- **迭代**: 从失败中提取信息指导下一步尝试
- **工具优先**: 所有操作必须通过工具调用完成，不要只输出计划而不执行

## 输出规范

每次回复末尾必须包含以下标记之一（用于阶段流转决策）：

```
DECISION: CONTINUE|PIVOT|STOP
REASON: <一句话原因>
```

- CONTINUE: 当前方向有潜力，继续迭代
- PIVOT:   遇到瓶颈，切换假设或架构
- STOP:    结果已达预期或资源耗尽

## 技术参考

### 评分规则
- 190 个预测步分为 3 段: S1(0-47步,权重25%), S2(47-95步,权重25%), S3(95-190步,权重50%)
- S1 = 100*exp(-20*Rel-MSE),  S2 = 100*exp(-10*Rel-MSE)
- S3 = max(100/(1+10*RMSE), 50*exp(-FD^2))
- Rel-MSE 逐样本逐时间步计算，单样本上限 clamp(max=5.0)
- 总分 = 0.25*S1 + 0.25*S2 + 0.5*S3

### 数据约定
- Task1: task1_val.hdf5 (训练/验证), task1_test.hdf5 (测试) — key: "tensor", "x-coordinate", "t-coordinate"
- Task2: task2_part{0,1,2}_train.h5 (训练, 含 nu), task2_val.h5, task2_test.h5 (测试, 无 nu) — key: "tensor", "x_coordinate", "t_coordinate", "nu"
- 输入 [B,10,256] → 输出 [B,190,256] → 提交 [B,200,256] (前10步=GT)

### 模型架构
- ChunkedFNO1d: SpectralConv1d(modes=24) + FNOBlock1d(width=64,depth=4)
- Lift: Conv1d(t_in+1→width), Project: Conv1d(width→t_out)
- 残差: last_frame.expand + project(features)
- Task2: FiLM conditioning + nu_estimator (Conv1d→AvgPool→Linear)
- 验证: 完整 190 步自回归 rollout, 非 teacher forcing

### CLI 参数接口 (train.py / infer.py)
- `--task` (task1|task2), `--output_dir`, `--data_dir` 必须支持
- argparse 别名: `--output-dir/--output_dir`, `--data-dir/--data_dir`
- device 默认: `"cuda" if torch.cuda.is_available() else "cpu"`
- checkpoint 命名: `best_checkpoint.pt`
- 所有参数必须有合理默认值, 不可设 required=True

### 必须生成的 5 个文件
- code/model.py: SpectralConv1d, FNOBlock1d, FiLM, ChunkedFNO1d, build_model
- code/dataset.py: Normalizer, BurgersDataset, WindowedBurgersDataset, get_dataloaders
- code/train.py: 训练循环 + 验证 + 早停 + LR调度 + 保存 checkpoint
- code/infer.py: 加载 checkpoint → 预测 → HDF5 输出 (前10步=GT)
- code/utils.py: compute_segment_scores, spectral_gradient_loss, temporal_difference_loss
"""


class Phase:
    """阶段基类"""
    def __init__(self, client: LLMClient, memory: ResearchMemory, cfg=None):
        self.client = client
        self.memory = memory
        self.cfg = cfg
    
    def run(self) -> bool:
        """执行阶段，返回是否成功完成"""
        raise NotImplementedError
    
    def _chat(self, messages: list, tools: bool = True) -> dict[str, Any]:
        """调用LLM，可选择是否启用工具"""
        schemas = registry.get_schemas() if tools else None
        return self.client.chat(messages, tools=schemas)
    
    def _tool_call_loop(self, messages: list, max_rounds: int = 12) -> str:
        """工具调用循环：LLM 反复 思考→调用工具→观察结果，直到不再调用工具或达到上限。

        关键改进:
        - 截断大文件结果 (避免 API 400)
        - 格式化工具结果为可读摘要
        - 重试时正确携带 reasoning_content (DeepSeek 兼容)
        - 达到上限时返回部分结果而非空字符串
        """
        last_content = ""
        for round_idx in range(max_rounds):
            try:
                resp = self._chat(messages)
            except Exception as e:
                print(f"[tool_call_loop] API 调用失败 (round {round_idx+1}): {e}")
                if last_content:
                    return last_content
                return f"[Error] LLM API call failed: {e}"

            if resp.get("tool_calls"):
                # 构建助手消息（含工具调用和 reasoning_content）
                assistant_msg = {
                    "role": "assistant",
                    "content": resp["content"] or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{round_idx}_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for i, tc in enumerate(resp["tool_calls"])
                    ],
                }
                if resp.get("reasoning_content"):
                    assistant_msg["reasoning_content"] = resp["reasoning_content"]
                messages.append(assistant_msg)

                # 执行工具并格式化结果
                for i, tc in enumerate(resp["tool_calls"]):
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]
                    result = registry.call(tool_name, tool_args)

                    # 格式化工具结果摘要（方便 LLM 快速理解）
                    summary = self._format_tool_result(tool_name, tool_args, result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{round_idx}_{i}"),
                        "content": summary,
                    })

                last_content = resp["content"] or ""
            else:
                return resp["content"] or ""

        print(f"[tool_call_loop] 达到最大轮数 {max_rounds}，返回部分结果")
        return last_content or messages[-1].get("content", "") if messages else ""

    def _format_tool_result(self, name: str, args: dict, result: str) -> str:
        """格式化工具结果为紧凑摘要，避免超出 API 上下文限制。"""
        MAX_LEN = 6000  # 每个工具结果最大字符数

        if name == "read_file":
            path = args.get("path", "?")
            truncated = len(result) > MAX_LEN
            if truncated:
                # 保留头部和尾部
                head = result[:MAX_LEN * 2 // 3]
                tail = result[-MAX_LEN // 3:]
                return head + f"\n\n... [文件过长，已截断: 共 {len(result)} 字符，显示 {len(head)} 字符头部 + {len(tail)} 字符尾部] ...\n\n" + tail
            return result

        elif name == "run_shell":
            truncated = len(result) > MAX_LEN
            if truncated:
                # 只保留 stdout 的开头和 stderr
                try:
                    parsed = json.loads(result)
                    stdout = parsed.get("stdout", "")
                    stderr = parsed.get("stderr", "")
                    returncode = parsed.get("returncode", "?")
                    summary = f"[returncode={returncode}]\nSTDOUT:\n{stdout[-3000:]}\nSTDERR:\n{stderr[-1000:]}"
                    return summary[:MAX_LEN]
                except Exception:
                    return result[:MAX_LEN] + f"\n... [truncated, total {len(result)} chars]"
            return result

        elif name == "inspect_hdf5":
            # 只保留关键信息
            try:
                parsed = json.loads(result)
                datasets = parsed.get("datasets", {})
                lines = [f"{k}: shape={v.get('shape','?')} dtype={v.get('dtype','?')}"
                         for k, v in list(datasets.items())[:10]]
                return "HDF5 summary:\n" + "\n".join(lines)
            except Exception:
                return result[:MAX_LEN]

        elif name == "summarize_code":
            try:
                parsed = json.loads(result)
                classes = parsed.get("classes", [])
                functions = parsed.get("functions", [])
                lines = [f"Classes ({len(classes)}):"] + \
                        [f"  - {c['signature']} (line {c['line']})" for c in classes[:8]] + \
                        [f"Functions ({len(functions)}):"] + \
                        [f"  - {f['signature']} (line {f['line']})" for f in functions[:12]]
                return "\n".join(lines)[:MAX_LEN]
            except Exception:
                return result[:MAX_LEN]

        elif name == "analyze_log":
            try:
                parsed = json.loads(result)
                stats = parsed.get("stats", {})
                records = parsed.get("last_10_records", [])
                lines = [f"Analysis: {parsed.get('matches', 0)} matches, trend={stats.get('trend','?')}"]
                lines.append(f"Stats: first={stats.get('first','?')}, last={stats.get('last','?')}, "
                            f"min={stats.get('min','?')}, max={stats.get('max','?')}")
                return "\n".join(lines)[:MAX_LEN]
            except Exception:
                return result[:MAX_LEN]

        # 其他工具：简单截断
        if len(result) > MAX_LEN:
            return result[:MAX_LEN] + f"\n... [truncated, total {len(result)} chars]"
        return result


# =============================================================================
# Phase 1: 文献解析与逻辑解构
# =============================================================================

class LiteraturePhase(Phase):
    """文献解析阶段：阅读项目文档、理解数据、分析基线"""

    def run(self) -> bool:
        print("\n[Phase 1] 文献解析与逻辑解构...")

        task_data = {
            "task1": "固定 nu=0.001, 数据 task1_val.hdf5 (100,200,256) + task1_test.hdf5 (1000,10,256)",
            "task2": "变 nu (1e-4~1e-2), 训练 task2_part{0,1,2}_train.h5 (各1000,320,256) + task2_val.h5 (100,210,256) + task2_test.h5 (1000,10,256, 无 nu)",
        }
        task_desc = task_data.get(self.memory.task, "task1")

        # 预计算文件名，避免 f-string 中的反斜杠
        val_filename = "task1_val.hdf5" if self.memory.task == "task1" else "task2_part0_train.h5"
        test_filename = "task1_test.hdf5" if self.memory.task == "task1" else "task2_test.h5"
        prompt = f"""## 文献解析任务 (Phase 1: Literature)

当前任务: {self.memory.task} — {task_desc}

### 操作步骤

请按顺序执行以下工具调用，每步完成后根据结果进入下一步：

**步骤 1: 阅读核心文档**
- read_file Background.md — 理解比赛规则与评分标准
- read_file NEURAL_OPERATOR_PRINCIPLES.md — 理解 FNO/DeepONet 数学原理
- read_file AGENTS.md — 理解提交规范与合规要求

**步骤 2: 探查数据结构**
- inspect_hdf5 data_and_sample_submission/train_val_test_init/{val_filename}
- inspect_hdf5 data_and_sample_submission/train_val_test_init/{test_filename}
- list_files data_and_sample_submission/train_val_test_init/

**步骤 3: 分析参考代码** (如 code-ref/ 存在)
- summarize_code code-ref/model.py
- summarize_code code-ref/train.py
- read_file code-ref/utils.py — 重点关注 compute_segment_scores

**步骤 4: 综合输出**

基于以上信息，输出结构化的文献综述，包含：
1. **核心科学问题**: 本任务要解决什么 PDE 问题？有哪些困难？
2. **数据特征**: 训练/验证/测试数据的规模、维度、物理含义
3. **基线架构**: FNO 的核心数学原理和工程实现要点
4. **评分影响**: 3 段评分规则的权重要求（S3 占 50%）对模型设计意味着什么？
5. **已知优化方向**: 长时稳定性、物理一致性、泛化能力的改进策略

### 关键数据路径
- 数据目录: ./data_and_sample_submission/train_val_test_init/
- 参考代码: ./code-ref/
- 一旦确定设计方向，将在实验阶段使用 code/train.py 等进行训练
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        summary = self._tool_call_loop(messages)
        self.memory.literature_summary = summary
        self.memory.current_phase = "diagnosis"

        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_literature_summary.md",
            "content": f"# 文献与技术综述\n\n{summary}\n",
        })

        print("[Phase 1] 完成。文献综述已保存。")
        return True


# =============================================================================
# Phase 2: 瓶颈诊断与假设提出
# =============================================================================

class DiagnosisPhase(Phase):
    """瓶颈诊断阶段：分析基线性能，提出优化假设"""

    def run(self) -> bool:
        print("\n[Phase 2] 瓶颈诊断与假设提出...")

        context = self.memory.get_context()

        prompt = f"""## 瓶颈诊断任务 (Phase 2: Diagnosis)

基于文献综述阶段的分析结果，识别当前基线的主要瓶颈并提出可验证的优化假设。

### 研究上下文

{context}

### 诊断框架

请从以下四个维度逐一分析瓶颈：

**1. 长时稳定性 (最关键, S3 权重 50%)**
- 自回归预测的误差累积机制是什么？
- ChunkedFNO1d 的 chunk_size=10 是否最优？
- 是否需要 multistep rollout loss 或 scheduled sampling？

**2. 物理一致性**
- 模型输出是否满足 Burgers 方程约束？
- 激波位置的相移误差如何影响 Frechet distance？
- use_physics_loss 是否值得启用？（注意计算开销）

**3. 泛化能力 (Task2 专属)**
- FiLM 条件化是否充分？nu_estimator 的精度如何？
- 训练数据中 nu 的分布是否覆盖了测试集范围？

**4. 计算效率**
- 模型参数量（425k）是否合适？是否可以压缩/扩大？
- 推理时间是否有超出 120s 的风险？

### 输出格式

对每个瓶颈给出：

```
## 瓶颈 N: <名称>
- **现象**: <具体表现，引用数据或日志>
- **根因**: <理论分析>
- **影响**: <对评分的量化影响估算>

## 假设 N: <假设描述>
- **理论依据**: <为什么这个方向有希望>
- **具体方案**: <要改什么代码/参数>
- **预期效果**: <量化预期（如 S3 提升 3-5 分）>
- **验证方法**: <如何快速判断假设是否成立>
```

### 关键评分公式参考
- S1 = 100 × exp(-20 × Rel-MSE)，权重 25%
- S2 = 100 × exp(-10 × Rel-MSE)，权重 25%
- S3 = max(100/(1+10×RMSE), 50×exp(-FD²))，权重 50%
- 说明：S3 对 RMSE 和 Frechet distance 的敏感度不同，小 RMSE 时 Lorentzian 项主导，反之 Frechet 项更重要

请调用工具收集所需信息（如 analyze_log 分析训练动态、summarize_code 审查关键模块），然后输出诊断报告。
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        diagnosis = self._tool_call_loop(messages)
        self._parse_diagnosis(diagnosis)
        self.memory.current_phase = "design"

        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_diagnosis_report.md",
            "content": f"# 瓶颈诊断与假设报告\n\n{diagnosis}\n",
        })

        print(f"[Phase 2] 完成。识别瓶颈 {len(self.memory.bottlenecks)} 个，提出假设 {len(self.memory.hypotheses)} 个。")
        return True
    
    def _parse_diagnosis(self, text: str):
        """解析诊断文本，提取瓶颈和假设（正则+结构化匹配）"""
        # 按段落分割，寻找瓶颈/假设区块
        sections = re.split(r'\n(?=#{1,3}\s+|\d+[.\)]\s*(?:瓶颈|Bottleneck|假设|Hypothesis))', text)
        for section in sections:
            is_bn = bool(re.search(r'(?:瓶颈|Bottleneck)', section, re.IGNORECASE))
            is_hy = bool(re.search(r'(?:假设|Hypothesis)', section, re.IGNORECASE))
            # 提取列表项
            items = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+[.\)])\s*(.+?)(?=\n|$)', section)
            for item in items:
                item = item.strip()
                if is_bn and item:
                    self.memory.bottlenecks.append(item)
                elif is_hy and item:
                    self.memory.hypotheses.append(item)


# =============================================================================
# Phase 3: 自主设计与代码演进
# =============================================================================

class DesignPhase(Phase):
    """代码设计阶段：根据假设编写/修改代码"""

    def run(self) -> bool:
        print("\n[Phase 3] 自主设计与代码演进...")

        context = self.memory.get_context()

        # 读取当前代码状态
        code_files = []
        code_dir = "./code"
        if os.path.exists(code_dir):
            for fname in sorted(os.listdir(code_dir)):
                if fname.endswith(".py"):
                    content = registry.call("read_file", {
                        "path": os.path.join(code_dir, fname), "limit": 80
                    })
                    try:
                        snippet = json.loads(content).get('content', '')[:800]
                    except Exception:
                        snippet = content[:800]
                    code_files.append(f"### {fname}\n```python\n{snippet}\n```\n")

        code_status = "\n".join(code_files) if code_files else "code/ 目录为空，需要从零创建所有文件。"

        # 当前假设
        hypothesis_text = "\n".join(f"- {h}" for h in self.memory.hypotheses[:3]
                                   ) if self.memory.hypotheses else "尚未提出假设，请基于基线架构进行初始实现。"

        prompt = f"""## 代码设计任务 (Phase 3: Design)

### 研究上下文

{context}

### 当前代码状态

{code_status}

### 当前假设

{hypothesis_text}

### 操作指南

**如果 code/ 为空或缺少核心文件**: 请使用 write_file 工具**逐一**创建全部 5 个核心文件。
不要只在回复中输出代码——必须调用 write_file 工具将代码真正写入磁盘！

**如果 code/ 已有代码**: 请基于当前假设进行针对性修改。
先 read_file 读取需要修改的文件，再 write_file 覆盖写入完整代码。

### 必须创建的 5 个核心文件

| 文件 | 核心内容 | 关键约束 |
|------|---------|---------|
| code/model.py | SpectralConv1d, FNOBlock1d, FiLM, ChunkedFNO1d, build_model, burgers_residual | Task2 需 FiLM + nu_estimator；forward 返回 (pred, cond) |
| code/dataset.py | Normalizer, BurgersDataset, WindowedBurgersDataset, get_dataloaders, get_test_loader | 归一化在训练集上计算并共享；Task1/2 HDF5 key 不同 |
| code/train.py | parse_args, train_epoch, validate, main | argparse 支持 --task/--output_dir/--data_dir；checkpoint=best_checkpoint.pt；validate 必须完整 rollout |
| code/infer.py | parse_args, main | 加载 checkpoint → 预测 → HDF5；前 10 步复制 GT 并 assert；支持 --checkpoint/--output |
| code/utils.py | compute_segment_scores, spectral_gradient_loss, temporal_difference_loss, save_hdf5, Timer, Logger | compute_segment_scores 接受 [B,190,256]；Rel-MSE clamp(5.0)；S3=max(Lorentzian,Frechet) |

### CLI 参数模板 (train.py)

```python
parser.add_argument("--task", default="task1", choices=["task1", "task2"])
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output")
parser.add_argument("--data-dir", "--data_dir", dest="data_dir",
                    default="./data_and_sample_submission/train_val_test_init")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
```

### 设计约束

- **代码必须完整且可执行** — 不能有 TODO 或 pass 占位符
- **所有 imports 必须明确** — 使用 `import torch; from torch import nn`
- **每写完一个文件** → 立即调用 validate_code 检查语法
- **全部写完后** → 调用 quick_test_model 做 smoke test
- **最后** → 调用 list_files code/ 确认目录结构

### 参考学习

你可以阅读 code-ref/ 中的参考代码来学习设计思路：
- read_file code-ref/model.py (limit=200) — 学习 SpectralConv1d 和 ChunkedFNO1d 的写法
- read_file code-ref/train.py (limit=150) — 学习训练循环的结构

但这些是**学习参考**，你最终写入 `code/` 的代码必须在理解吸收后用自己的方式表达，而不是逐字复制。

请现在开始。如果你已理解以上要求，请直接调用工具开始工作。
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = self._tool_call_loop(messages, max_rounds=20)

        self.memory.current_phase = "experiment"
        self.memory.code_versions.append({
            "iteration": self.memory.iteration,
            "timestamp": datetime.now().isoformat(),
            "note": result[:500],
        })

        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_design_notes.md",
            "content": f"# 设计迭代 {self.memory.iteration}\n\n{result}\n",
        })

        print("[Phase 3] 完成。代码已更新。")
        return True


# =============================================================================
# Phase 4: 实验验证与科学迭代
# =============================================================================

class ExperimentPhase(Phase):
    """实验验证阶段：运行训练、评估、分析结果、决定下一步"""
    
    def run(self) -> bool:
        print("\n[Phase 4] 实验验证与科学迭代...")

        # 预检查：核心代码文件必须存在，否则无法执行实验
        required_files = ["code/train.py", "code/infer.py", "code/model.py", "code/dataset.py", "code/utils.py"]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(f"[Phase 4] 代码文件缺失，无法执行实验: {missing}")
            self.memory.current_phase = "design"
            return False

        self.memory.iteration += 1
        exp_id = len(self.memory.experiments) + 1
        
        context = self.memory.get_context()
        
        # 构建运行命令，传入配置参数
        data_dir = self.cfg.research.data_dir if self.cfg else "./data_and_sample_submission/train_val_test_init"
        output_iter = f"output/{self.memory.task}/iter_{self.memory.iteration}"
        epochs = getattr(self.cfg.model, "epochs", 220) if self.cfg else 220
        batch_size = getattr(self.cfg.model, "batch_size", 16) if self.cfg else 16

        import sys
        py_exe = sys.executable  # 使用当前 Agent 的 Python（确保 GPU 版本一致）
        import torch
        device_flag = "--device cuda" if torch.cuda.is_available() else "--device cpu"
        train_cmd = f'"{py_exe}" code/train.py --task {self.memory.task} --output_dir {output_iter} --data_dir {data_dir} --epochs {epochs} --batch_size {batch_size} {device_flag}'
        infer_cmd = f'"{py_exe}" code/infer.py --task {self.memory.task} --checkpoint {output_iter}/best_checkpoint.pt --output {output_iter}/pred.hdf5 --data_dir {data_dir} {device_flag}'
        
        print(f"[Experiment] Train command: {train_cmd}")
        print(f"[Experiment] Infer command: {infer_cmd}")
        
        # 运行训练
        train_result = registry.call("run_shell", {
            "command": train_cmd,
            "timeout": 1800,
        })
        
        # 运行推理
        infer_result = registry.call("run_shell", {
            "command": infer_cmd,
            "timeout": 300,
        })
        
        # 尝试读取验证指标
        metrics = {}
        metrics_path = f"output/{self.memory.task}/iter_{self.memory.iteration}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        
        # 记录实验
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            phase="experiment",
            hypothesis=self.memory.hypotheses[0] if self.memory.hypotheses else "baseline",
            code_changes=[],
            config={},
            metrics=metrics,
            conclusion="",
            status="success" if metrics else "failed",
        )
        
        # 让LLM分析实验结果并决定下一步
        prompt = f"""## 实验分析任务 (Phase 4: Experiment)

实验已执行完毕。请按决策树顺序分析，输出结论和下一步行动。

### 研究上下文

{context}

### 训练输出（最近 2000 字符）

{train_result[-2000:] if len(train_result) > 2000 else train_result}

### 推理输出

{infer_result[-2000:] if len(infer_result) > 2000 else infer_result}

### 验证指标

{json.dumps(metrics, ensure_ascii=False, indent=2)}

### 决策树（严格按顺序执行，匹配到即停止）

**Step A — CLI/参数错误检测（最高优先级）**

如果输出包含以下任一关键字，这是代码接口问题，不是模型问题：
- `error: unrecognized arguments`
- `error: the following arguments are required`
- `can't open file` (returncode=2)

→ **必须立即修复**: 检查 train.py/infer.py 的 argparse，确保 `--task`, `--output_dir/--output-dir`, `--data_dir/--data-dir` 全部支持，且 checkpoint 命名为 `best_checkpoint.pt`
→ 决策: `CONTINUE` (修复 CLI 后重跑)
→ **不要分析不存在的训练结果，不要提出模型架构修改**

**Step B — 数据/环境错误检测**

如果输出包含:
- `FileNotFoundError` 或 `HDF5 file not found`
- `ImportError` 或 `ModuleNotFoundError`

→ 检查 dataset.py 的默认 data_dir 和所有 import
→ 决策: `CONTINUE` (修复后重跑)

**Step C — 训练正常进行（有 loss 曲线和 val 指标）**

按以下框架分析:

1. **收敛状态**: loss 是否稳定下降？是否过拟合（train loss << val loss）？
2. **分段表现**: S1(短期), S2(中期), S3(长期:50%权重) 各自的健康度？
3. **与历史最优对比**: 本次 total={metrics.get('total','N/A')} vs 历史最优={self.memory.best_metrics.get('total','N/A')}
4. **决策依据**:
   - 如果 total 相比历史最优提升 > 5 分 → 强烈 CONTINUE
   - 如果 total 连续 3 次没有提升 → 考虑 PIVOT
   - 如果 total > 85 (预测精度接近满分) → 考虑 STOP
   - 如果训练时间接近限制 → 优先 CONTINUE 并减少 epochs

### 输出格式（严格按此格式，解析器依赖）

```
DECISION: CONTINUE|PIVOT|STOP
REASON: <基于上述分析框架的一句话原因>
NEXT_ACTION: <具体的下一步操作描述>
```
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        analysis = self.client.chat(messages)["content"]
        record.conclusion = analysis
        
        # 解析决策（正则匹配，兼容多种LLM输出格式）
        decision = "CONTINUE"
        m = re.search(r'DECISION\s*:\s*(\S+)', analysis, re.IGNORECASE)
        if m:
            decision = m.group(1).strip().upper()
        if decision not in ("CONTINUE", "PIVOT", "STOP"):
            decision = "CONTINUE"
        
        if decision == "STOP":
            record.status = "success"
            self.memory.stop_reason = "Agent decided to stop after analysis."
        elif decision == "PIVOT":
            self.memory.current_phase = "diagnosis"
        else:
            self.memory.current_phase = "design"
        
        self.memory.add_experiment(record)
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_experiment_{exp_id}_report.md",
            "content": f"# 实验 {exp_id} 报告\n\n{analysis}\n",
        })
        
        print(f"[Phase 4] 完成。实验 {exp_id} 决策: {decision}")
        return True
