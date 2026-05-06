# safe-code-change

## Purpose

用于所有代码修改，保证 Codex 小步、可回滚、可测试地改动代码。

## Rules

- 修改前先运行 git status；
- 先读相关文件，再编辑；
- 一次只解决一个明确问题；
- 不进行大范围重写；
- 不移动或删除官方数据；
- 不硬编码用户本机绝对路径；
- 路径必须通过 config 或相对路径管理；
- 新功能必须尽量添加测试；
- 修改后运行相关最小测试；
- 不提交 hdf5、h5、pt、pth、ckpt、zip、log、大型输出文件；
- 如果修改影响 submission，需要运行 validate_submission。

## Change workflow

1. Identify target files
2. Inspect current implementation
3. Make minimal patch
4. Run focused test
5. Run broader test if needed
6. Update docs if behavior changes
7. Commit atomic change

## Required final report

- 修改了哪些文件；
- 为什么修改；
- 运行了哪些测试；
- 是否有风险；
- git commit hash。
