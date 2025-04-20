from tqdm import tqdm
import re

def truncate_full_text(df, tokenizer, max_length=2048):
    truncated_texts = []
    for _, row in tqdm(df.iterrows(),total=len(df)):
        text = row['full_text']
        text = re.sub(r'[\ud800-\udfff]', '', text)
        token_ids = tokenizer.encode(str(text))

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        truncated_text = tokenizer.decode(token_ids)
        truncated_texts.append(truncated_text)

    df['truncated_text'] = truncated_texts
    return df

def extract_pred(predictions):
    preds = []
    for prediction in predictions:
        # 使用非贪婪匹配，提取 conclusion 标签之间的内容
        match = re.search(r'<conclusion>(.*?)</conclusion>', prediction, re.DOTALL)
        if match:
            preds.append(match.group(1).strip())
        else:
            preds.append("")  # 没有匹配上也保留空字符串（你也可以用 None）
    return preds


        