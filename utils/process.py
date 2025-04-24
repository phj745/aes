from tqdm import tqdm
import re
import os
import pandas as pd
def get_df(input_dir,model_name,df_len):
    output_dir = input_dir.replace('.csv', f'_{model_name}.csv') 
    output_dir = output_dir if not df_len else output_dir.replace(f'_{model_name}.csv',f'_{model_name}_{df_len}.csv')
# 读取数据
# 判断文件是否存在并读取
    if os.path.exists(output_dir):
        df = pd.read_csv(output_dir)
        print(f"读取已存在的文件: {output_dir}")
    else:
        df = pd.read_csv(input_dir)
        print(f"读取默认输入文件: {input_dir}")
    if df_len:
        df=df[:df_len]
    return df

def get_output_column(mode):
    if mode=='label':
        r_name = 'reason_gt'
        p_name = 'pred_gt'
    elif mode=='infer':
        r_name = 'reason'
        p_name = 'pred'
    else:
        r_name = 'reason_tips'
        p_name = 'pred_tips'
    return r_name,p_name
    
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


        