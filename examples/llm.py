from llama import Model


#
# first define and load/init model
#
model = Model(
    creator_hf_repo='HuggingFaceTB/SmolLM2-1.7B-Instruct',
    hf_repo='bartowski/SmolLM2-1.7B-Instruct-GGUF',
    hf_file='SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
)

model.init(ctx_size=8192, predict=1024, gpu_layers=99)

#
# messages
#
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '1 + 1 = ?'},
    {'role': 'assistant', 'content': '2'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
]

for chunk in model.completions(messages=messages, temp=0.7, top_p=0.8, top_k=100):
    print(chunk, flush=True, end='')
else:
    print()

#
# prompt
#
for chunk in model.completions(prompt='Evaluate 1 + 2 in Python. Result in Python is', temp=0.7, top_p=0.8, top_k=100):
    print(chunk, flush=True, end='')
else:
    print()
