# AES

自动评分系统（Automatic Essay Scoring）

---

## 1. 数据准备（prepare_data）

- 对输入文本进行长度截断（默认 `2048` 字符）。
- 将训练集划分为 5 个折叠（Fold），用于后续的交叉验证训练。
- 当前在 `fold0` 上进行测试。

---

## 2. 推理（infer）
用于根据 label-标签生成推理过程 tips-根据标准生成推理过程 --inferj仅有场景和输出格式。

### 使用方式：

```bash
python infer.py --mode 'label|tips|infer' --model_id <模型路径> --input_dir <输入文件路径> --len 测试的文件长度，会在对应csv前面添加_len
