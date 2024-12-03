from openai import OpenAI
from ollama import chat

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="EMPTY"
)

def call_openai(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages,
        model="/data/rzhou/renzhou/models/Qwen2.5-7B-Instruct/",
        temperature=0, 
        max_tokens=1000, 
        n=1,
    )
    return response.choices[0].message.content

def call_ollama(model, prompt):
    response = chat(model=model, messages=[
        {
            'role': 'user',
            'content': f'{prompt}',
        },
    ])
    return response['message']['content']
