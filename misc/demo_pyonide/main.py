from huggingface_hub import hf_hub_download
from llama.cffi import llama_generate, LlamaOptions


options = LlamaOptions(
    no_display_prompt=True,
    n_ctx=1024,
    predict=512,
    # flash_attn=True,
    # cont_batching=True,
    simple_io=True,
    # log_disable=True,
    # mlock=True,
    # no_mmap=True,
    # hf_repo='bartowski/Phi-3.1-mini-128k-instruct-GGUF',
    # hf_file='Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    # hf_file='Phi-3.1-mini-128k-instruct-IQ2_M.gguf',
    model="./models/bartowski/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-IQ2_M.gguf",
    # model="./models/IndexTeam/Index-1.9B-Chat-GGUF/ggml-model-Q4_K_M.gguf",
    chat_template='chatml',
    prompt='<|im_start|>user\nWho is Novak Djokovic?<|im_end|>\n<|im_start|>assistant\n',
)

print(f'{options = }')
# hf_hub_download(repo_id=options.hf_repo, filename=options.hf_file, local_dir='/models')

def print_chunk(chunk):
    print(chunk, flush=True, end='')

for chunk in llama_generate(options, callback=print_chunk):
    print('!!!', chunk, flush=True, end='')
