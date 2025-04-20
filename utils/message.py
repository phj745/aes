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
        
        
    
    
    
    