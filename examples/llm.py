from llama import Model

#
# first define and load/init model
#
# https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
# Apache license 2.0
model = Model( # 1.71B
    creator_hf_repo='HuggingFaceTB/SmolLM2-1.7B-Instruct',
    hf_repo='bartowski/SmolLM2-1.7B-Instruct-GGUF',
    hf_file='SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
)

model.init(ctx_size=8192, gpu_layers=99)

#
# messages
#
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '1 + 1 = ?'},
    {'role': 'assistant', 'content': '2'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
]

completions = model.completions(
    messages=messages,
    predict=1024,
    temp=0.7,
    top_p=0.8,
    top_k=100,
)

for chunk in completions:
    print(chunk, flush=True, end='')

print()

#
# prompt
#
prompt='Evaluate 1 + 2 in Python. Result in Python is'

completions = model.completions(
    prompt=prompt,
    predict=1024,
    temp=0.7,
    top_p=0.8,
    top_k=100,
)

for chunk in completions:
    print(chunk, flush=True, end='')

print()
