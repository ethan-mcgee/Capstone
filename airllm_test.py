import torch
from airllm import AutoModel

MODEL_ID = "Qwen/Qwen3-14B"
LAYER_PATH = r"C:\AI\airllm-layers"

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_ID,
    layer_shards_saving_path=LAYER_PATH,
    max_seq_len=512,
)

messages = [
    {
        "role": "system",
        "content": "You are a concise and helpful assistant.",
    },
    {
        "role": "user",
        "content": "Explain what machine learning is in three sentences.",
    },
]

input_ids = model.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=False,
)

input_length = input_ids.shape[-1]

result = model.generate(
    input_ids.cuda(),
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
    return_dict_in_generate=True,
)

new_tokens = result.sequences[0][input_length:]

answer = model.tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)

print("\nModel response:\n")
print(answer)
