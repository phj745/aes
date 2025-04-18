import argparse
import pandas as pd
from utils.prompt import system_cot_label, system_cot_infer
from utils.message import get_messages
from utils.process import extract_pred
from vllm import LLM, SamplingParams

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../../data/AES/fold1/val.csv', help='Path to the input CSV file')
parser.add_argument('--model_id', type=str, default='/mnt/afs1/llm_gard/share/model/Qwen/Qwen2.5-72B-Instruct', help='Path to the model')
parser.add_argument('--label', action='store_true', help='Whether to use label mode')
parser.add_argument('--len', type=int, help='Whether to use label mode')
args = parser.parse_args()

# 参数赋值
input_dir = args.input_dir
model_id = args.model_id
label = args.label
df_len=args.len
model_name = model_id.split('/')[-1]
output_dir = input_dir.replace('.csv', f'_{model_name}.csv')

r_name = 'reason_gt' if label else 'reason'
p_name = 'pred_gt' if label else 'pred'

# 读取数据
df = pd.read_csv(input_dir)
if df_len:
    output_dir=output_dir.replace(f'_{model_name}.csv',f'_{model_name}_{df_len}.csv')
    df=df[:df_len]
    print(output_dir)
# 构造提示
if label:
    messages = get_messages(texts=df['full_text'], scores=df['score'], system_prompt=system_cot_label)
else:
    messages = get_messages(texts=df['full_text'], system_prompt=system_cot_infer)

# 模型生成
sampling_params = SamplingParams(temperature=0.15, top_p=0.95, max_tokens=3000)
llm = LLM(model=model_id, tensor_parallel_size=8, gpu_memory_utilization=0.8)
outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=True)

# 提取预测
prediction = [output.outputs[0].text for output in outputs]
df[p_name] = extract_pred(prediction)
df[r_name] = [p.replace('<think>', '').replace('</think>', '') for p in prediction]

# 保存结果
df.to_csv(output_dir, index=False)
