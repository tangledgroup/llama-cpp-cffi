from llama.model import Model

models = [
    Model(
        '01-ai/Yi-1.5-9B-Chat-16K',
        'mradermacher/Yi-1.5-9B-Chat-16K-i1-GGUF',
        # 'Yi-1.5-9B-Chat-16K.i1-IQ2_M.gguf',
        # 'Yi-1.5-9B-Chat-16K.i1-IQ3_M.gguf',
        'Yi-1.5-9B-Chat-16K.i1-IQ4_XS.gguf',
        # 'Yi-1.5-9B-Chat-16K.i1-Q4_K_M.gguf',
    ),
    Model(
        'mistralai/Mistral-7B-Instruct-v0.3',
        'bartowski/Mistral-7B-Instruct-v0.3-GGUF',
        # 'Mistral-7B-Instruct-v0.3-IQ2_M.gguf',
        # 'Mistral-7B-Instruct-v0.3-IQ3_M.gguf',
        'Mistral-7B-Instruct-v0.3-IQ4_XS.gguf',
        # 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
    ),
    Model(
        'microsoft/Phi-3-mini-128k-instruct',
        'bartowski/Phi-3.1-mini-128k-instruct-GGUF',
        # 'Phi-3.1-mini-128k-instruct-IQ2_XS.gguf',
        # 'Phi-3.1-mini-128k-instruct-Q4_K_S.gguf',
        'Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    ),
    Model(
        'microsoft/Phi-3-mini-4k-instruct',
        'bartowski/Phi-3.1-mini-4k-instruct-GGUF',
        # 'Phi-3.1-mini-4k-instruct-Q4_K_S.gguf',
        'Phi-3.1-mini-4k-instruct-Q4_K_M.gguf',
    ),
    Model(
        'microsoft/phi-2',
        'andrijdavid/phi-2-GGUF',
        # 'ggml-model-Q4_K_S.gguf',
        'ggml-model-Q4_K_M.gguf',
    ),
    Model(
        'IndexTeam/Index-1.9B-Chat',
        'IndexTeam/Index-1.9B-Chat-GGUF',
        # 'ggml-model-Q4_0.gguf',
        'ggml-model-Q4_K_M.gguf',
    ),
    Model(
        'internlm/internlm2-chat-1_8b',
        'QuantFactory/internlm2-chat-1_8b-GGUF',
        # 'internlm2-chat-1_8b.Q4_K_S.gguf',
        'internlm2-chat-1_8b.Q4_K_M.gguf',
    ),
    Model(
        'Qwen/Qwen2-1.5B-Instruct',
        'Qwen/Qwen2-1.5B-Instruct-GGUF',
        # 'qwen2-1_5b-instruct-q4_0.gguf',
        'qwen2-1_5b-instruct-q4_k_m.gguf',
    ),
    Model(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        # 'tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf',
        'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    ),
]
