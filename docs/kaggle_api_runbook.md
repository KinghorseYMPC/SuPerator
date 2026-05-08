# Kaggle API Runbook

本 runbook 只描述用户手动执行 Kaggle API 训练闭环前后的本地准备步骤。项目脚本默认不调用 Kaggle API，不读取 `kaggle.json`，不训练模型。

## 1. 安全放置 kaggle.json

- Windows 路径：`%USERPROFILE%\.kaggle\kaggle.json`
- 不要把 `kaggle.json` 放入项目目录。
- 不要复制、打印、提交或上传 `kaggle.json`。

## 2. 安装和验证

```bash
python -m pip install kaggle
kaggle --version
kaggle datasets list -s test
```

这些命令由用户在本机手动执行，用于确认 Kaggle CLI 已安装且凭据位置正确。

## 3. 创建私有 dataset

先在本地构建 dataset package：

```bash
python scripts/create_kaggle_dataset_package.py
```

然后编辑 `kaggle_dataset_package/superator-inputs/dataset-metadata.json` 中的 `id`，把 `<KAGGLE_USERNAME>` 替换成自己的 Kaggle 用户名。

创建私有 dataset：

```bash
kaggle datasets create -p kaggle_dataset_package/superator-inputs --dir-mode zip
```

后续更新使用：

```bash
kaggle datasets version -p kaggle_dataset_package/superator-inputs --dir-mode zip -m "update inputs"
```

## 4. 创建 / 推送私有 kernel

先在本地构建 kernel package：

```bash
python scripts/create_kaggle_kernel_package.py --username <KAGGLE_USERNAME>
```

再手动推送：

```bash
kaggle kernels push -p kaggle_kernel/package
```

## 5. 查看状态

```bash
kaggle kernels status <KAGGLE_USERNAME>/superator-task1-min-train
```

## 6. 下载输出

```bash
kaggle kernels output <KAGGLE_USERNAME>/superator-task1-min-train -p kaggle_outputs/task1_min_train
```

`kaggle_outputs/` 是本地忽略目录，不能提交到 git。

## 7. 回传本地后

- 解析 `outputs/checkpoints`。
- 解析 `experiments` 和 `experiments/experiment_registry.jsonl`。
- 本地 validation / submission 仍在笔记本执行。
- Kaggle 输出只作为返回 artifact，不能替代本地 source of truth。

## 8. 不允许

- 不上传 `kaggle.json`。
- 不公开私有数据。
- 不把 Kaggle 输出提交 git。
- 不在 Kaggle 上作为唯一源码修改项目。
- 不把 Kaggle notebook 当成唯一代码副本。
