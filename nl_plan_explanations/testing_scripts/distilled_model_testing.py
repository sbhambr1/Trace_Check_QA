import ollama

response = ollama.chat(
    model='deepseek-r1:8b',
    messages=[
        {
            'role': 'user',
            'content': 'Hello'
        }
    ]
)

print(response['message']['content'])