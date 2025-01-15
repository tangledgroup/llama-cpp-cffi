__all__ = ['demo_models']

from llama.model import Model


demo_models = {
    #
    # llm
    #
    'Qwen/Qwen2.5-0.5B-Instruct': Model(
        creator_hf_repo='Qwen/Qwen2.5-0.5B-Instruct',
        hf_repo='Qwen/Qwen2.5-0.5B-Instruct-GGUF',
        hf_file='qwen2.5-0.5b-instruct-q4_k_m.gguf',
    ),
    'Qwen/Qwen2.5-1.5B-Instruct': Model(
        creator_hf_repo='Qwen/Qwen2.5-1.5B-Instruct',
        hf_repo='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
        hf_file='qwen2.5-1.5b-instruct-q4_k_m.gguf',
    ),
    'Qwen/Qwen2.5-3B-Instruct': Model(
        creator_hf_repo='Qwen/Qwen2.5-3B-Instruct',
        hf_repo='Qwen/Qwen2.5-3B-Instruct-GGUF',
        hf_file='qwen2.5-3b-instruct-q4_k_m.gguf',
    ),
    'Qwen/Qwen2.5-7B-Instruct': Model(
        creator_hf_repo='Qwen/Qwen2.5-7B-Instruct',
        hf_repo='bartowski/Qwen2.5-7B-Instruct-GGUF',
        hf_file='Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    ),
    'arcee-ai/arcee-lite': Model( # 1.5B Qwen2
        creator_hf_repo='arcee-ai/arcee-lite',
        hf_repo='arcee-ai/arcee-lite-GGUF',
        hf_file='arcee-lite-Q4_K_M.gguf',
    ),
    'arcee-ai/Llama-3.1-SuperNova-Lite': Model( # 8B
        creator_hf_repo='arcee-ai/Llama-3.1-SuperNova-Lite',
        hf_repo='arcee-ai/Llama-3.1-SuperNova-Lite-GGUF',
        hf_file='supernova-lite-v1.Q4_K_M.gguf',
    ),
    'HuggingFaceTB/SmolLM2-360M-Instruct': Model(
        creator_hf_repo='HuggingFaceTB/SmolLM2-360M-Instruct',
        hf_repo='bartowski/SmolLM2-360M-Instruct-GGUF',
        hf_file='SmolLM2-360M-Instruct-Q8_0.gguf',
    ),
    'HuggingFaceTB/SmolLM2-1.7B-Instruct': Model(
        creator_hf_repo='HuggingFaceTB/SmolLM2-1.7B-Instruct',
        hf_repo='bartowski/SmolLM2-1.7B-Instruct-GGUF',
        hf_file='SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
    ),
    'microsoft/Phi-3.5-mini-instruct': Model( # 3.82B
        creator_hf_repo='microsoft/Phi-3.5-mini-instruct',
        hf_repo='bartowski/Phi-3.5-mini-instruct-GGUF',
        hf_file='Phi-3.5-mini-instruct-Q4_K_M.gguf',
    ),
    'RWKV/v6-Finch-1B6-HF': Model(
        creator_hf_repo='RWKV/v6-Finch-1B6-HF',
        hf_repo='latestissue/rwkv-6-finch-1b6-gguf',
        hf_file='rwkv-6-finch-1b6-Q4_K.gguf',
    ),
    'RWKV/v6-Finch-3B-HF': Model(
        creator_hf_repo='RWKV/v6-Finch-3B-HF',
        hf_repo='bartowski/v6-Finch-3B-HF-GGUF',
        hf_file='v6-Finch-3B-HF-Q4_K_M.gguf',
    ),

    #
    # vlm
    #
    'Qwen/Qwen2-VL-2B-Instruct': Model( # 2.21B
        creator_hf_repo='Qwen/Qwen2-VL-2B-Instruct',
        hf_repo='bartowski/Qwen2-VL-2B-Instruct-GGUF',
        hf_file='Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
        mmproj_hf_file='mmproj-Qwen2-VL-2B-Instruct-f16.gguf',
    ),
    'Qwen/Qwen2-VL-7B-Instruct': Model( # 8.29B
        creator_hf_repo='Qwen/Qwen2-VL-7B-Instruct',
        hf_repo='bartowski/Qwen2-VL-7B-Instruct-GGUF',
        hf_file='Qwen2-VL-7B-Instruct-Q4_K_M.gguf',
        mmproj_hf_file='mmproj-Qwen2-VL-7B-Instruct-f16.gguf',
    ),
    'openbmb/MiniCPM-V-2_6': Model( # 8.1B
        creator_hf_repo='openbmb/MiniCPM-V-2_6',
        hf_repo='bartowski/MiniCPM-V-2_6-GGUF',
        hf_file='MiniCPM-V-2_6-Q4_K_M.gguf',
        mmproj_hf_file='mmproj-MiniCPM-V-2_6-f16.gguf',
    ),
    'liuhaotian/llava-v1.6-mistral-7b': Model( # 7.57B
        creator_hf_repo='liuhaotian/llava-v1.6-mistral-7b',
        hf_repo='cjpais/llava-1.6-mistral-7b-gguf',
        hf_file='llava-v1.6-mistral-7b.Q4_K_M.gguf',
        mmproj_hf_file='mmproj-model-f16.gguf',
        tokenizer_hf_repo='mistralai/Mistral-7B-Instruct-v0.2',
    ),
    'vikhyatk/moondream2': Model( # 1.87B
        creator_hf_repo='vikhyatk/moondream2',
        # hf_repo='vikhyatk/moondream2',
        hf_repo='moondream/moondream2-gguf',
        hf_file='moondream2-text-model-f16.gguf',
        mmproj_hf_file='moondream2-mmproj-f16.gguf',
    ),
    'qnguyen3/nanoLLaVA-1.5': Model( # 1.05B
        creator_hf_repo='qnguyen3/nanoLLaVA-1.5',
        hf_repo='bee-kake/nanollava-1.5-gguf',
        hf_file='nanollava-text-model-f16.gguf',
        mmproj_hf_file='nanollava-mmproj-f16.gguf',
    ),
}
