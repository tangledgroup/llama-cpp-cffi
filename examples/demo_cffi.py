import psutil
from llama.cffi import llama_generate, LlamaOptions


options = LlamaOptions(
    no_display_prompt=True,
    threads=psutil.cpu_count(logical=False),
    # ctx_size=8192,
    ctx_size=4 * 4096,
    predict=512,
    flash_attn=True,
    cont_batching=True,
    simple_io=True,
    # log_disable=True,
    hf_repo='bartowski/Phi-3.1-mini-128k-instruct-GGUF',
    hf_file='Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    # hf_file='Phi-3.1-mini-128k-instruct-IQ2_M.gguf',
    chat_template='chatml',
    # prompt='<|im_start|>user\nEvaluate 1 + 2.<|im_end|>\n<|im_start|>assistant\n',
    prompt='<|system|>\nYou are a helpful assistant.<|end|><|user|>\nEvaluate 1 + 2.<|end|>\n<|assistant|>\n',
)

for chunk in llama_generate(options):
    print(chunk, flush=True, end='')
