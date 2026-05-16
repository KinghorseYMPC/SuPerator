# External References

本目录存放从外部项目导入的只读参考资产。

这些资产不参与 SuPerator 主运行流程，仅用于代码学习、对比分析和后续适配。

## 目录

| 目录 | 来源 | 用途 |
|---|---|---|
| [pdeagent_code_ref/](pdeagent_code_ref/) | pdeagent 项目 | 模型基线、评分函数、Agent 架构参考 |

## 使用规则

- 只读参考，不作为 SuPerator 主运行代码；
- 不包含数据、checkpoint、输出、日志、config.yaml 或 API key；
- 后续适配时应从本目录迁移到 src/adapters/ 或 src/models/ 等正式模块；
- 不要直接修改导入的文件，保留原始状态以保持参考完整性。
