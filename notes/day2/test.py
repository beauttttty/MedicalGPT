from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    trust_remote_code=True,
)

text = "老人突然一侧肢体无力、说话不清"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print(tokens)
print(ids)
print(tokenizer.decode(ids))


messages = [
    {"role": "system", "content": "你是一个研究用医疗助手。"},
    {"role": "user", "content": text},
]
print(tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
))