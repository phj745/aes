import pandas as pd
from utils.message import create_dpo_message, create_dpo_dataset  # 假设你已经实现了这两个函数
import json
import os
version=1
# 定义数据集类型和折数
type_names = ['train','val']
def remove_invisible_unicode(text):
    import re
    invisible_chars = [
        '\u200b', '\u200c', '\u200d', '\ufeff', '\u2060'
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    return text
    
for type_name in type_names:
    for i in range(1):  # 假设有 5 折交叉验证
        # 输入文件路径
        input_dir = f'../../data/AES/fold{i}/dpo_{type_name}_v{version}.csv'
        df = pd.read_csv(input_dir)
        
        # 获取 fold 名称
        fold_name = input_dir.split('/')[-2]
        
        # 数据预处理
        # 确保相关列存在并且为字符串类型
        df['preferred'] = df['preferred'].apply(lambda x: remove_invisible_unicode(str(x)))
        df['rejected'] = df['rejected'].apply(lambda x: remove_invisible_unicode(str(x)))
        df['truncated_text'] = df['truncated_text'].apply(lambda x: remove_invisible_unicode(str(x)))
        
        # 提取人类输入、更优回答和更差回答
        humans = df['truncated_text'].tolist()          # 人类输入
        preferred_responses = df['preferred'].tolist()  # 更优回答
        rejected_responses = df['rejected'].tolist()    # 更差回答
        
        # 构建 DPO 数据集
        dpo_messages = create_dpo_dataset(humans, preferred_responses, rejected_responses)  # 调用 create_dpo_dataset 函数
        
        # 输出路径
        output_path = f"/mnt/afs1/panhaojun/project/LLaMA-Factory/data/aes_{type_name}_{fold_name}_dpo_v{version}.json"
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 如果需要创建目录，取消注释此行
        
        # 保存 DPO 数据集
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_messages, f, ensure_ascii=False, indent=2)
        
        print(f"✅ DPO 数据集已保存到: {output_path}")
        
        # 加载 dataset_info 文件
        dataset_info_path = '/mnt/afs1/panhaojun/project/LLaMA-Factory/data/dataset_info.json'
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        # 添加新的字段
        dataset_name = output_path.split('/')[-1].replace('.json', '')
        dataset_info[dataset_name] = {
            "ranking":True,
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
            },
        }
        
        # 保存回原文件
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ dataset_info['{dataset_name}'] 已成功添加并保存到 {dataset_info_path}")