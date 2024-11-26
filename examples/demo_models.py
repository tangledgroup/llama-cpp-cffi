__all__ = ['models']

from llama.model import Model

models = {
    'TinyLlama/TinyLlama_v1.1': Model(
        creator_hf_repo='TinyLlama/TinyLlama_v1.1',
        hf_repo='QuantFactory/TinyLlama_v1.1-GGUF',
        hf_file='TinyLlama_v1.1.Q4_K_M.gguf',
    ),
    'TinyLlama/TinyLlama_v1.1_math_code': Model(
        creator_hf_repo='TinyLlama/TinyLlama_v1.1_math_code',
        hf_repo='mjschock/TinyLlama_v1.1_math_code-Q4_K_M-GGUF',
        hf_file='tinyllama_v1.1_math_code-q4_k_m.gguf',
    ),
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0': Model(
        creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        hf_repo='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        hf_file='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    ),
    'cognitivecomputations/TinyDolphin-2.8-1.1b': Model(
        creator_hf_repo='cognitivecomputations/TinyDolphin-2.8-1.1b',
        hf_repo='tsunemoto/TinyDolphin-2.8-1.1b-GGUF',
        hf_file='tinydolphin-2.8-1.1b.Q4_K_M.gguf',
    ),
    'HuggingFaceTB/SmolLM-1.7B-Instruct-v0.2': Model(
        creator_hf_repo='HuggingFaceTB/SmolLM-1.7B-Instruct-v0.2',
        hf_repo='bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF',
        hf_file='SmolLM-1.7B-Instruct-v0.2-Q4_K_M.gguf',
    ),
    'HuggingFaceTB/SmolLM2-1.7B-Instruct': Model(
        creator_hf_repo='HuggingFaceTB/SmolLM2-1.7B-Instruct',
        hf_repo='bartowski/SmolLM2-1.7B-Instruct-GGUF',
        hf_file='SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
    ),
    'microsoft/phi-2': Model(
        creator_hf_repo='microsoft/phi-2',
        hf_repo='andrijdavid/phi-2-GGUF',
        hf_file='ggml-model-Q4_K_M.gguf',
    ),
    'microsoft/Phi-3-mini-4k-instruct': Model(
        creator_hf_repo='microsoft/Phi-3-mini-4k-instruct',
        hf_repo='bartowski/Phi-3.1-mini-4k-instruct-GGUF',
        hf_file='Phi-3.1-mini-4k-instruct-Q4_K_M.gguf',
    ),
    'microsoft/Phi-3-mini-128k-instruct': Model(
        creator_hf_repo='microsoft/Phi-3-mini-128k-instruct',
        hf_repo='bartowski/Phi-3.1-mini-128k-instruct-GGUF',
        hf_file='Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    ),
    'Qwen/Qwen2-1.5B-Instruct': Model(
        creator_hf_repo='Qwen/Qwen2-1.5B-Instruct',
        hf_repo='Qwen/Qwen2-1.5B-Instruct-GGUF',
        hf_file='qwen2-1_5b-instruct-q4_k_m.gguf',
    ),
    'arcee-ai/arcee-lite': Model(
        creator_hf_repo='arcee-ai/arcee-lite',
        hf_repo='arcee-ai/arcee-lite-GGUF',
        hf_file='arcee-lite-Q4_K_M.gguf',
    ),
    'mistralai/Mistral-7B-Instruct-v0.3': Model(
        creator_hf_repo='mistralai/Mistral-7B-Instruct-v0.3',
        hf_repo='bartowski/Mistral-7B-Instruct-v0.3-GGUF',
        hf_file='Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
    ),
}
