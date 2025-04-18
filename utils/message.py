def create_message(system,user):
    message = [dict(role="system", content=user)]
    user_content = dict(role="user", content=user)
    message.append(user_content)
    return message
    
def get_messages(texts,system_prompt,scores=[]):
    if len(scores):
        messages=[]
        for text,score in zip(texts,scores):
            messages.append(create_message(system_prompt.format(score=score),text))
    else:
        messages=[create_message(system_prompt,text) for text in texts]
    return messages