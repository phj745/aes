import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from utils.process import truncate_full_text
from transformers import AutoTokenizer
model_id='/mnt/afs1/llm_gard/share/model/Qwen/Qwen2.5-72B-Instruct'
train=pd.read_csv('../../data/AES/train_ori.csv')
train = train.reset_index(drop=True)
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
train=truncate_full_text(train,tokenizer)
train['fold'] = 0
skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
for fold, (train_index, test_index) in enumerate(skf.split(train, train['score'].values)):
    train.loc[test_index, 'fold'] = fold
print(train['fold'].value_counts())
for fold in range(5):
    tra = train[train['fold'] != fold]
    dev = train[train['fold'] == fold]
    folder_path = f'../../data/AES/fold{fold}/'
    os.makedirs(folder_path, exist_ok=True) 
    tra.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
    dev.to_csv(os.path.join(folder_path, 'val.csv'), index=False)