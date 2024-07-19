from llama.llama_cli_cffi_cpu import llama_generate, Model, Options
# from llama.llama_cli_cffi_cuda_12_5 import llama_generate, Model, Options
# from llama.llama_cli_ctypes_cuda import llama_generate, Model, Options
# from llama.llama_cli_ctypes_cuda_12_5 import llama_generate, Model, Options

from llama.formatter import get_config

model = Model(
    creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hf_repo='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    hf_file='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
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
