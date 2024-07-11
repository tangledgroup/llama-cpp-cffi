from llama.model import Model

models = [
    Model(
        'microsoft/Phi-3-mini-128k-instruct',
        'bartowski/Phi-3.1-mini-128k-instruct-GGUF',
        'Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    ),
    Model(
        'Qwen/Qwen2-1.5B-Instruct',
        'Qwen/Qwen2-1.5B-Instruct-GGUF',
        'qwen2-1_5b-instruct-q4_k_m.gguf',
    ),
    Model(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    ),
]