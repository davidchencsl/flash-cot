from ollama import chat

def get_response(model, system_prompt, user_prompt):
    response = chat(model=model, messages=[
    {
        'role': 'system',
        'content': f'{system_prompt}',
    },
    {
        'role': 'user',
        'content': f'{user_prompt}',
    },
    ])
    return response['message']['content']