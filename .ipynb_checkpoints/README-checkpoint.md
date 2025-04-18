# AES

自动评分系统（Automatic Essay Scoring）

---

## 1. 数据准备（prepare_data）

- 对输入文本进行长度截断（默认 `2048` 字符）。
- 将训练集划分为 5 个折叠（Fold），用于后续的交叉验证训练。
- 当前在 `fold1` 上进行测试。

---

## 2. 推理（infer）

用于根据 Ground Truth (`--label` 参数) 或直接推理生成结果和理由。

### 使用方式：

```bash
python infer.py --label --model_id <模型路径> --input_dir <输入文件路径>
