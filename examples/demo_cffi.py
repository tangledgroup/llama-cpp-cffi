import os
import sys
sys.path.append(os.path.abspath('.'))

from llama.cffi import llama_generate, LlamaOptions


options = LlamaOptions(
    no_display_prompt=True,
    ctx_size=8192,
    predict=-2,
    flash_attn=True,
    cont_batching=True,
    simple_io=True,
    log_disable=True,
    hf_repo='bartowski/Phi-3.1-mini-128k-instruct-GGUF',
    hf_file='Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    chat_template='chatml',
    prompt='<|im_start|>user\nWho is Novak Djokovic?<|im_end|>\n<|im_start|>assistant\n',
)

for chunk in llama_generate(options):
    print(chunk, flush=True, end='')
