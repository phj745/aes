import argparse
import pandas as pd
from utils.prompt import system_cot_label, system_cot_infer,system_cot_infer_tips
from utils.message import get_messages
from utils.process import extract_pred,get_output_column,get_df
from vllm import LLM, SamplingParams
import torch 
import os

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../../data/AES/fold0/train.csv', help='Path to the input CSV file')
parser.add_argument('--model_id', type=str, default='/mnt/afs1/llm_gard/share/model/Qwen/Qwen2.5-72B-Instruct', help='Path to the model')
parser.add_argument('--mode', type=str, help='Whether to use label mode',default='infer')
parser.add_argument('--len', type=int, help='Whether to use label mode')
args = parser.parse_args()

# 参数赋值
input_dir = args.input_dir
model_id = args.model_id
mode = args.mode
df_len=args.len
model_name = model_id.split('/')[-1]

path_name = ('/').join(input_dir.split('/')[:-1])
r_name,p_name=get_output_column(mode)
n_gpu = torch.cuda.device_count()
df=get_df(input_dir,model_name,df_len)
    
# 构造提示
if mode=='label':
    messages = get_messages(texts=df['truncated_text'], scores=df['score'], system_prompt=system_cot_label)
elif mode=='infer':
    messages = get_messages(texts=df['truncated_text'], system_prompt=system_cot_infer)
else:
    messages = get_messages(texts=df['truncated_text'], system_prompt=system_cot_infer_tips)
    

# 模型生成
sampling_params = SamplingParams(temperature=0.8, top_p=0.25, max_tokens=3000)
try:
    llm = LLM(model=model_id,tensor_parallel_size=n_gpu, gpu_memory_utilization=0.8)
except Exception as e:
    llm = LLM(model=model_id,tensor_parallel_size=n_gpu//2, gpu_memory_utilization=0.8)
outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=True)

# 提取预测
prediction = [output.outputs[0].text for output in outputs]
df[p_name] = extract_pred(prediction)
df[r_name] = prediction

# 保存结果
df.to_csv(output_dir, index=False)
