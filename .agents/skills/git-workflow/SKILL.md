# git-workflow

## Purpose

定义 Codex 在 SuPerator 项目中的 git 操作规范。

## Rules

- 每次修改前运行 git status；
- 工作区不干净时，先说明已有改动，不要覆盖用户修改；
- commit 必须小而清晰；
- commit message 使用英文祈使句或简短动词短语；
- 不提交数据、checkpoint、预测、日志、压缩包；
- 不提交 outputs/、experiments/ 下的大文件；
- 不提交 __pycache__、.pytest_cache、.venv；
- 修改 .gitignore 后检查 git status；
- 每次 commit 后输出 commit hash。

## Suggested commit messages

- add agent skills structure
- update project governance docs
- add submission validator tests
- add task1 baseline dataset loader
- add fno1d smoke test
- fix hdf5 key detection

## Pre-commit checklist

- git status
- pytest -q 或相关最小测试
- 确认没有大文件进入 staged changes
- git diff --cached --stat
- git commit
