from llama.llama_cli_cffi_cpu import llama_generate, Model, Options
from llama.formatter import get_config

model = Model(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
)

config = get_config(model.creator_hf_repo)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
]

options = Options(
    ctx_size=config.max_position_embeddings,
    predict=-2,
    model=model,
    prompt=messages,
)

for chunk in llama_generate(options):
    print(chunk, flush=True, end='')

# newline
print()
