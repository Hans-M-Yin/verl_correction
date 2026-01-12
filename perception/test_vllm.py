from openai import OpenAI
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://172.17.0.2:18903/v1"
)
response = client.chat.completions.create(
    model="qwen3-vl-30b-a3b",
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请问你是谁？"
                }
            ]
        }
    ],
)
response = response.choices[0].message.content
print(response)