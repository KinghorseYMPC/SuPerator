# debug-and-fix

## Purpose

用于处理报错、测试失败、训练中断、HDF5 读取错误、submission 校验失败、SLURM 作业失败等问题。

## Debug workflow

1. Reproduce：复现错误，保留完整错误信息；
2. Localize：定位到具体文件、函数、输入、shape 或配置；
3. Hypothesize：提出最小解释；
4. Patch：做最小修复；
5. Regression test：添加或更新测试；
6. Verify：重新运行失败命令；
7. Record：记录根因和修复方式。

## Common failure classes

- HDF5 key 错误；
- shape 不匹配；
- dtype 不一致；
- 前 10 步初始条件不一致；
- NaN / Inf；
- time.csv schema 错误；
- code/ 为空；
- zip 顶层结构错误；
- CUDA / PyTorch 环境错误；
- SLURM 路径或环境激活错误。

## Rules

- 不允许通过删除测试来“修复”失败；
- 不允许吞掉异常；
- 不允许只打印 warning 而跳过硬约束；
- 不允许随机改多个模块；
- 修复后必须说明根因。
