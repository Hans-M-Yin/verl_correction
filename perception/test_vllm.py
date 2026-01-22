import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

# =========================
# 1. 模型与 processor 加载
# =========================

# model_name = "/root/autodl-tmp/qwen2.5-vl-7b"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer   # Qwen 的 tokenizer 内含 chat template


# =========================
# 2. 构造多模态对话（raw messages）
# =========================
# 注意：image 用占位符表示，真正的 image 在 processor 中传入

messages = [
    {
        "role": "system",
        "content": [
            {"type": 'text',
             "text": "A conversation between User and Qwen Assistant. The assistant will answer user's question helpfully. The assistant first reasoning the question step by step in <think></think> tags and finally provide the final answer with few words in <answer></answer>."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "从图中来看，一共有几个训练的epoch的正确率高于或等于98%？请你首先仔细观察图片内容，然后再分析和回答问题。"}
        ],
    },
#     {
#         "role": "assistant",
#         "content": [
#             {
#                 "type": "text",
#                 "text": """<think>图片展示了一张深度学习模型的正确率随着训练步数增长的变化情况，横轴表示训练的epoch（迭代次数），纵轴表示准确率（Accuracy %）。可以看到，模型在训练过程中逐步提高其准确率。
#
# 根据图表，我们可以看到以下几点：
#
# 1. 在第1个epoch时，准确率为96.0%。
# 2. 在第2个epoch时，准确率上升到97.0%。
# 3. 在第3个epoch时，准确率进一步提升至97.8%。
# 4. 在第4个epoch时，准确率达到最高点，为98.0%。
# 5. 从第5个epoch开始，准确率快速下降到40%。
# 等等，我之前的回答可能有错，"""
#             }
#         ]
#     }
]

# =========================
# 3. apply_chat_template
# =========================
# 这一步只做“文本层面”的模板拼接
# 不会真正处理 image tensor

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("===== Chat Prompt =====")
# prompt = prompt[:prompt.rfind("<|im_end|>")]
print(prompt)

# =========================
# 4. 准备图像
# =========================

image = Image.open("example.png").convert("RGB")

# =========================
# 5. Processor 处理多模态输入
# =========================
# processor 会：
# - tokenizer(prompt)
# - image -> vision encoder 所需 tensor
# - 自动对齐 <image> token

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

# 移动到模型所在 device
inputs = {k: v.to(model.device) for k, v in inputs.items()}


# =========================
# 6. 模型推理（generate）
# =========================
for i in range(8):
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.5,
        )

    # =========================
    # 7. 解码输出
    # =========================

    output_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print("===== Model Output =====")
    print(output_text)
    print("========================")
