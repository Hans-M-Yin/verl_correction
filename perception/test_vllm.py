from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# 方法1: 查看特殊 token
print("所有特殊 token:", tokenizer.special_tokens_map)
print("图像相关 token id:")
print(f"<image>: {tokenizer.convert_tokens_to_ids('<image>')}")
print(f"</image>: {tokenizer.convert_tokens_to_ids('</image>')}")
print(f"<|image_pad|>: {tokenizer.convert_tokens_to_ids('<|image_pad|>')}")
print(f"<|im_end|>: {tokenizer.convert_tokens_to_ids('<|im_end|>')}")