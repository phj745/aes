def create_message(system,user):
    message = [dict(role="system", content=system)]
    user_content = dict(role="user", content=user)
    message.append(user_content)
    return message
    
def get_messages(system_prompt,texts,scores=[]):
    if len(scores):
        messages=[]
        for text,score in zip(texts,scores):
            messages.append(create_message(system_prompt.format(score=score),text))
    else:
        messages=[create_message(system_prompt,text) for text in texts]
    return messages

def create_sft_message(system, human, gpt):
    return {
        "conversations": [
            {
                "from": "human",
                "value": human
            },
            {
                "from": "gpt",
                "value": gpt
            }
        ],
        "system": system
    }
def create_sft_dataset(system,humans,gpts):
    return [create_sft_message(system, human, gpt) for human,gpt in zip(humans,gpts)]
        

def create_dpo_message(human, preffered, rejected):
    """
    构建单个 DPO 数据样本。
    
    参数:
        human_input (str): 人类输入的指令或问题。
        good_response (str): 更优的回答（chosen）。
        bad_response (str): 更差的回答（rejected）。
    
    返回:
        dict: 符合 ShareGPT 格式的单个数据样本。
    """
    # 构建对话历史
    conversations = [
        
        {"from": "human", "value": human},  # 人类输入
    ]
    
    # 构建 chosen 和 rejected 部分
    dpo_message = {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": preffered},
        "rejected": {"from": "gpt", "value": rejected},
        "system":""
    }
    
    return dpo_message


def create_dpo_dataset(humans,preffereds,rejecteds):
    """
    构建完整的 DPO 数据集。
    
    参数:
        data_tuples (list of tuple): 包含多个 (human_input, good_response, bad_response) 的元组列表。
    
    返回:
        list: 符合 ShareGPT 格式的 DPO 数据集。
    """
    dataset = []
    
    for human,preffered,rejected in zip(humans,preffereds,rejecteds):
        # 调用 create_dpo_message 构建单个样本
        dpo_message = create_dpo_message(human=human, preffered=preffered, rejected=rejected)
        dataset.append(dpo_message)
    
    return dataset


    
    
    
    