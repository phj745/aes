import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

def quadratic_weighted_kappa(y_true, y_pred, score_range=(1, 6)):
    """
    计算二次加权Kappa
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        score_range: 评分范围，默认(1, 6)
    返回:
        QWK分数
    """
    min_score, max_score = score_range
    n = max_score - min_score + 1  # 类别数
    y_true = np.clip(y_true, min_score, max_score) - min_score
    y_pred = np.clip(y_pred, min_score, max_score) - min_score

    # 初始化混淆矩阵
    confusion = np.zeros((n, n))
    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1  # 转换为0-based索引

    # 计算权重矩阵（二次权重）
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i, j] = (i - j) ** 2 / ((n - 1) ** 2)

    # 计算实际分布和期望分布
    hist_true = np.sum(confusion, axis=1)
    hist_pred = np.sum(confusion, axis=0)
    expected = np.outer(hist_true, hist_pred) / np.sum(confusion)

    # 计算QWK
    numerator = np.sum(weights * confusion)
    denominator = np.sum(weights * expected)
    kappa = 1 - numerator / denominator

    return kappa

def evaluate_predictions(csv_path):
    """
    评估预测结果
    参数:
        csv_path: CSV文件路径，需包含 `score` 和 `pred` 列
    返回:
        评估指标（QWK、Accuracy、MSE、RMSE）
    """
    # 读取数据并清洗
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=['score', 'pred'])
    
    # 检查异常值
    invalid_scores = df_clean[(df_clean['score'] < 1) | (df_clean['score'] > 6)]
    invalid_preds = df_clean[(df_clean['pred'] < 1) | (df_clean['pred'] > 6)]
    if len(invalid_scores) > 0 or len(invalid_preds) > 0:
        print(f"警告：发现 {len(invalid_scores)} 个无效的真实评分和 {len(invalid_preds)} 个无效的预测评分！")
        print("这些值已被裁剪到有效范围 [1, 6]。")
    
    # 强制限制分数在1-6范围内
    y_true = np.clip(df_clean['score'].astype(int), 1, 6)
    y_pred = np.clip(df_clean['pred'].astype(int), 1, 6)
    
    # 检查数据有效性
    if len(y_true) == 0:
        raise ValueError(f"清洗后无有效数据！请检查CSV文件内容。\n文件路径: {csv_path}")
    
    # 计算指标
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 打印结果
    print(f"样本数量: {len(y_true)}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return {"QWK": qwk, "Accuracy": accuracy, "MSE": mse, "RMSE": rmse}

# 使用示例
if __name__ == "__main__":
    csv_path = 'D:/VScode/code/llm/data/val_qwen2.5-7b_lora_sft_1epoch.csv'  # 替换为你的路径
    evaluate_predictions(csv_path)