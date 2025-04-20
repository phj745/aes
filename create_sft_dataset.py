import pandas as pd
from utils.message import *
from utils.prompt import *
import json
import os
type_names=['train','val']
for type_name in type_names:
    for i in range(5):
        input_dir=f'../../data/AES/fold{i}/{type_name}_Qwen2.5-72B-Instruct.csv'
        df=pd.read_csv(input_dir)
        fold_name=input_dir.split('/')[-2]
        df['pred_gt']=df['pred_gt'].apply(lambda x:str(x))
        df['score']=df['score'].apply(lambda x:str(x))
        df=df[df['pred_gt']==df['score']]
        humans = df['truncated_text'].tolist()
        gpts = df['reason_gt'].tolist()
        messages = create_sft_dataset(system_cot_infer,humans,gpts)
        output_path = f"/mnt/afs1/panhaojun/project/LLaMA-Factory/data/aes_{type_name}_{fold_name}.json"
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"✅ message 已保存到: {output_path}")
        # 加载已有的 JSON 文件
        dataset_info_path='/mnt/afs1/panhaojun/project/LLaMA-Factory/data/dataset_info.json'
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        # 添加新的字段
        dataset_name=output_path.split('/')[-1].replace('.json','')
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            }
        }
        
        # 保存回原文件
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ dataset_info['{dataset_name}'] 已成功添加并保存到 {dataset_info_path}")

        
