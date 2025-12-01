import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # or "https://api.deepseek.com/v1"
)

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a concise assistant for a CS student."},
    {"role": "user", "content": "Explain what a Python virtual environment is in 3 sentences."},
]

model: str = "deepseek-chat"

response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)

