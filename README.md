# llama-cpp-cffi

Python binding for llama.cpp using cffi

## Build

```bash
python -m venv venv
source venv/bin/activate
pip install poetry
python -B examples/demo0.py

huggingface-cli download bartowski/Phi-3.1-mini-4k-instruct-GGUF Phi-3.1-mini-4k-instruct-Q4_K_M.gguf
huggingface-cli download bartowski/Phi-3.1-mini-128k-instruct-GGUF Phi-3.1-mini-128k-instruct-Q4_K_M.gguf
huggingface-cli download IndexTeam/Index-1.9B-Chat-GGUF  ggml-model-Q4_K_M.gguf
huggingface-cli download mradermacher/dolphin-2.9.3-qwen2-1.5b-GGUF dolphin-2.9.3-qwen2-1.5b.Q4_K_M.gguf
huggingface-cli download mradermacher/dolphin-2.9.3-qwen2-0.5b-GGUF dolphin-2.9.3-qwen2-0.5b.Q4_K_M.gguf
huggingface-cli download mradermacher/dolphin-2.9.3-llama-3-8b-GGUF dolphin-2.9.3-llama-3-8b.Q4_K_M.gguf
huggingface-cli download NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf

CXXFLAGS="-DSHARED_LIB" LDFLAGS="-shared -o libllama-cli.so" make -j llama-cli
```
