import pandas as pd
from utils.prompt import system_cot_label,system_cot_infer
from utils.message import get_messages
from utils.process import extract_pred
from vllm import LLM, SamplingParams
input_dir='../../data/AES/fold1/train.csv'
model_id='/mnt/afs1/llm_gard/share/model/Qwen/Qwen2.5-72B-Instruct'
model_name=model_id.split('/')[-1]
output_dir=input_dir.replace('.csv',f'_{model_name}.csv')
label=True
r_name='reason_gt' if label else 'reason'
p_name='pred_gt' if label else 'pred'
df=pd.read_csv(input_dir)
if label:
    messages=get_messages(texts=df['full_text'],scores=df['score'],system_prompt=system_cot_label)
else:
    messages=get_messages(texts=df['full_text'],scores=df['score'],system_prompt=system_cot_infer)

sampling_params = SamplingParams(temperature=0.15, top_p=0.95,max_tokens=3000)
llm = LLM(model=model_id,tensor_parallel_size=8,gpu_memory_utilization=0.88)
outputs = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=True)
prediction=[]
for output in outputs:
    generated_text = output.outputs[0].text
    prediction.append(generated_text)
df[p_name]=extract_pred(prediction)
df[r_name]=prediction.replace('<think>','').replace('</think>')
df.to_csv(output_dir,index=False)

        
    
    

    
    
    
    
    