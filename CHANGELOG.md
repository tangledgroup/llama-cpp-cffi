# CHANGELOG

## v0.4.42

Changed:
  - `llama.cpp` revision `b3298fa47a2d56ae892127ea038942ab1cada190`

## v0.4.41

Changed:
  - Updated cibuildwheel build images
  - `llama.cpp` revision `3d652bfddfba09022525067e672c3c145c074649`

## v0.4.40

Changed:
  - `llama.cpp` revision `3d652bfddfba09022525067e672c3c145c074649`

## v0.4.39

Changed:
  - `llama.cpp` revision `1a24c4621f0280306b0d53a4fa474fc65d3f1b2e`

Fixed:
  - Server accepts parameter `grammar_ignore_until`

## v0.4.38

Added:
  - Conditional Structured Output for `DeepSeek R1`-like models

Changed:
  - `llama.cpp` revision `1a24c4621f0280306b0d53a4fa474fc65d3f1b2e`

## v0.4.37

Changed:
  - `llama.cpp` revision `cc473cac7cea1484c1f870231073b0bf0352c6f9`

## v0.4.36
Changed:
  - `llama.cpp` revision `06c2b1561d8b882bc018554591f8c35eb04ad30e`

## v0.4.35

Changed:
  - CUDA_ARCHITECTURES `50;61;70;75;80;86;89;90;100;101;120`
  - `llama.cpp` revision `08d5986290cc42d2c52739e046642b8252f97e4b`

## v0.4.34

Fixed:
  - Memory leak on stopping iteration/completion

## v0.4.33

Fixed:
  - Handle `MemoryError` in `model_init`
  - Handle `MemoryError` in `context_init`

## v0.4.32

Changed:
  - `llama.cpp` revision `9626d9351a6dfb665400d9fccbda876a0a96ef67`

## v0.4.31

Added:
  - `CompletionsOptions.force_model_reload` to force model reload on every server request

Fixed:
  - Stop condition in `Model.completions`
  - Fixed (Vulkan related) GPU memory leaks

## v0.4.30

Fixed:
  - Stop condition in `Model.completions`

## v0.4.29

Fixed:
  - Stop condition in `Model.completions`

## v0.4.28

Fixed:
  - In `server.py`, lock is now used correctly

## v0.4.27

Added:
  - `CompletionsOptions.stop` is optional strgin which triggers stop of generation

Changed:
  - In `llama_cpp.py`, use `threading.Lock` instead of `DummyLock`
  - In `server.py`, use `asyncio.Lock` instead of `threading.Lock`

## v0.4.26

Changed:
  - Updated all requirements

## v0.4.25

Changed:
  - Reimplemented `is_cuda_available` and `is_vulkan_available`

## v0.4.24

Changed:
  - `llama.server` now uses env vars `LLAMA_CPP_HOST` and `LLAMA_CPP_PORT`

## v0.4.23

Changed:
  - Use `DummyLock` instead of `threading.Lock`
  - `llama.cpp` revision `63ac12856303108ee46635e6c9e751f81415ee64`

## v0.4.22

Changed:
  - Async to sync lock in server
  - `llama.cpp` revision `5137da7b8c3eaa090476a632888ca178ba109f8a`

## v0.4.21

Changed:
  - CUDA 12.8.0 for x86_64
  - CUDA_ARCHITECTURES `61;70;75;80;86;89;90;100;101;120`
  - `llama.cpp` revision `73e2ed3ce3492d3ed70193dd09ae8aa44779651d`

## v0.4.20

Changed:
  - New repo at https://github.com/ggml-org/llama.cpp.git
  - `llama.cpp` revision `0f2bbe656473177538956d22b6842bcaa0449fab`

## v0.4.19

Changed:
  - `llama.cpp` revision `d774ab3acc4fee41fbed6dbfc192b57d5f79f34b`
  - Build process `manylinux_2_28` -> `manylinux_2_34`
  - Build for `aarch64` platform WIP

## v0.4.18

Changed:
  - `llama.cpp` revision `eb7cf15a808d4d7a71eef89cc6a9b96fe82989dc`

## v0.4.17

Changed:
  - `llama.cpp` revision `6152129d05870cb38162c422c6ba80434e021e9f`

Fixed:
  - Fixed build process, json patches.
  - Reverted server code to previous version due to bug.

## v0.4.16

Added:
  - Dynamically load/unload models while executing prompts in parallel.

Changed:
  - `llama.cpp` revision `adc5dd92e8aea98f5e7ac84f6e1bc15de35130b5`

## v0.4.15

Changed:
  - `llama.cpp` revision `0ccd7f3eb2debe477ffe3c44d5353cc388c9418d`

Fixed:
  - CUDA architectures: 61, 70, 75, 80, 86, 89, 90

## v0.4.14

Changed:
  - `llama.cpp` revision `0ccd7f3eb2debe477ffe3c44d5353cc388c9418d`

Fixed:
  - CUDA architectures: all (including: 61, 70, 75, 80, 86, 89, 90)

## v0.4.13

Changed:
  - `llama.cpp` revision `bbf3e55e352d309573bdafee01a014b0a2492155`

## v0.4.12

Changed:
  - `llama.cpp` revision `091592d758cb55af7bfadd6c397f61db387aa8f3`

Fixed:
  - `gguf_*` missing symbols from `_llama_cpp_*` shared libraries
  - CUDA default arch `-arch=sm_61`

## v0.4.11

Changed:
  - `llama.cpp` revision `44d1e796d08641e7083fcbf37b33c79842a2f01e`

## v0.4.10

Changed:
  - `llama.cpp` revision `44d1e796d08641e7083fcbf37b33c79842a2f01e`

## v0.4.9

Changed:
  - `llama.cpp` revision `39509fb082895d1eae2486f8ad2cbf0e905346c4`

## v0.4.8

Changed:
  - `llama.cpp` revision `39509fb082895d1eae2486f8ad2cbf0e905346c4`

## v0.4.7

Changed:
  - `llama.cpp` revision `a29f0870d4846f52eda14ae28cea612ab66d903c`
  - Migrate from `clang` to `gcc-13` compiler/linker

## v0.4.6

Changed:
  - `llama.cpp` revision `1244cdcf14900dd199907b13f25d9c91a507f578`

## v0.4.5

Changed:
  - `llama.cpp` revision `1244cdcf14900dd199907b13f25d9c91a507f578`

## v0.4.4

Changed:
  - `llama.cpp` revision `924518e2e5726e81f3aeb2518fb85963a500e93a`
  - Migrate from `gcc` to `clang` compiler/linker

## v0.4.3

Changed:
  - `llama.cpp` revision `924518e2e5726e81f3aeb2518fb85963a500e93a`

## v0.4.2

Changed:
  - `llama.cpp` revision `924518e2e5726e81f3aeb2518fb85963a500e93a`

## v0.4.1

Changed:
  - `llama.cpp` revision `924518e2e5726e81f3aeb2518fb85963a500e93a`

Fixed:
  - Migrate from `clang` to `gcc-13` compiler/linker

## v0.4.0

Changed:
  - `llama.cpp` revision `9a483999a6fda350772aaf7bc541f1cb246f8a29`
  - Migrate from `gcc` to `clang` compiler/linker
  - Migrate from `make` to `cmake` build system
  - Option `ctx_size` renamed to `n_ctx`
  - Option `batch_size` renamed to `n_batch`
  - Option `ubatch_size` renamed to `n_ubatch`
  - Option `threads` renamed to `n_threads`
  - Option `threads_batch` renamed to `n_threads_batch`
  - Option `cache_type_k` renamed to `type_k`
  - Option `cache_type_v` renamed to `type_v`
  - Option `mlock` renamed to `use_mlock`
  - Option `no_mmap` renamed to `use_mmap`

## v0.3.3

Changed:
  - `llama.cpp` revision `0827b2c1da299805288abbd556d869318f2b121e`

## v0.3.2

Added:
 - `server` support for OpenAI extra fields: `grammar`, `json_schema`, `chat_template`

Changed:
  - `llama.cpp` revision `0827b2c1da299805288abbd556d869318f2b121e`

## v0.3.1

Added:
  - llama-cpp-cffi server - support for dynamic load/unload of model - hot-swap of models on demand
  - llama-cpp-cffi server - compatible with llama.cpp cli options
  - llama-cpp-cffi server - limited compatibility for OpenAI API `/v1/chat/completions` for text and vision models
  - Support for `CompletionsOptions.messages` for VLM prompts with a single message containing just a pair of `text` and `image_url` in `content`.

Changed:
  - `llama.cpp` revision `0827b2c1da299805288abbd556d869318f2b121e`

## v0.3.0

Added:
  - Qwen 2 VL 2B / 7B vision models support
  - WIP llama-cpp-cffi server - compatible with llama.cpp cli options instead of OpenAI

Changed:
  - `llama.cpp` revision `5896c65232c7dc87d78426956b16f63fbf58dcf6`
  - Refactored `Options` class into two separate classes: `ModelOptions`, `CompletionsOptions`

Fixed:
  - Llava (moondream2, nanoLLaVA-1.5, llava-v1.6-mistral-7b) vision models support
  - MiniCPM-V 2.5 / 2.6 vision models support

Removed:
  - Removed ambiguous `Options` class

## v0.2.7

Changed:
  - In `format_messages`, optional `options` argument
  - `llama.cpp` revision `081b29bd2a3d91e7772e3910ce223dd63b8d7d26`

## v0.2.6

Changed:
  - `llama.cpp` revision `5437d4aaf5132c879acda0bb67f2f8f71da4c9fe`

## v0.2.5

Fixed:
  - Replaced `tokenizer.decode(new_token_id)` with custom `_common_token_to_piece(context, new_token_id, True)`

## v0.2.4

Fixed:
  - `sampler_init` because `llama_sampler_init_penalties` in `llama.cpp` changed its behaviour

## v0.2.3

Changed:
  - `llama.cpp` revision `4f51968aca049080dc77e26603aa0681ea77fe45`
  - Build process now has global variable `LLAMA_CPP_GIT_REF`

Fixed:
  - Issue with Phi 3.5 based models, `tokenizer.decode([new_token_id], clean_up_tokenization_spaces=False)`

## v0.2.2

Added:
  - `Model.free`

Changed:
  - Fixed revision of `llama.cpp` for all wheels
  - `llama.cpp` revision `c27ac678dd393af0da9b8acf10266e760c8a0912`
  - disabled `llama_kv_cache_seq_cp` in `_decode_tokens`

## v0.2.1

Fixed:
  - Batch "decode" process. NOTE: Encode part is missing for encoder-decoder models.
  - Thread-safe calls to the most important functions of llama, llava, clip, ggml API.

Removed:
  - `mllama_completions` for low-level function for Mllama-based VLMs

## v0.2.0

Added:
  - New high-level Python API
  - Low-level C API calls from llama.h, llava.h, clip.h, ggml.h
  - `completions` for high-level function for LLMs / VLMs
  - `text_completions` for low-level function for LLMs
  - `clip_completions` for low-level function for CLIP-based VLMs
  - WIP: `mllama_completions` for low-level function for Mllama-based VLMs

Changed:
  - All examples

Removed:
  - `llama_generate` function
  - `llama_cpp_cli`
  - `llava_cpp_cli`
  - `minicpmv_cpp_cli`

## v0.1.23

Added:
  - Support and examples for `llava` and `minicpmv` models.

## v0.1.22

Added:
  - `llava` high-level API calls
  - `minicpmv` high-level API support

Changed:
    - Updated `llama.cpp`.

## v0.1.21

Changed:
    - Updated `llama.cpp`.

## v0.1.20

Changed:
    - Updated `llama.cpp`.

## v0.1.19

Changed:
    - Updated `llama.cpp` with **RWKV 6** support.

## v0.1.18

Changed:
    - Updated `llama.cpp` with **RWKV** support.

## v0.1.17

Added:
    - `LLAMA_CPP_BACKEND` which can be `cuda`, `vulkan` or `cpu`.

Changed:
    - Updated `llama.cpp`.
    - Updated requirements.
    - CUDA backend imports only 12.6 library.

Fixed:
    - `Options.top_p` check using `isinstance`.

## v0.1.16

Changed:
    - Updated `llama.cpp`.

## v0.1.15

Added:
    - `SmolLM-1.7B-Instruct-v0.2` examples.

Changed:
    - Updated `llama.cpp`.

## v0.1.14

Fixed:
    - Vulkan detection.

## v0.1.13

Fixed:
    - CUDA and Vulkan detection.

## v0.1.12

Added:
    - Build `vulkan_1_x` for general GPU.
    - Build `cuda 12.4.1` as default.

Changed:
    - Renamed examples for TinyLlama (chat, tool calling) and OpenAI.
    - Updated demo models definitions.
    - Updated examples (chat, tool calling).
    - `get_special_tokens` not supports parameter `force_standard_special_tokens: bool=False` which bypasses tokenizer's special tokens with standard/common ones.
    - Build `cuda 12.5.1` as additional build target but packaged on PyPI.
    - Build `cuda 12.6` as additional build target but packaged on PyPI.
    - Build `openblas` as additional build target but packaged on PyPI.

Fixed:
    - Handle `Options.no_display_prompt` on Python side.

## v0.1.11

Changed:
    - OpenAI: allow import of `routes` and `v1_chat_completions` handler.
    - `examples/demo_0.py`, tool calling example.

## v0.1.10

Added:
    - In `openai`, support for `prompt` and `extra_body`. Reference: https://github.com/openai/openai-python/blob/195c05a64d39c87b2dfdf1eca2d339597f1fce03/src/openai/resources/completions.py#L41
    - Pass `llama-cli` options to `openai`.
    - `util` module with `is_cuda_available` function.
    - `openai` supports both `prompt` and `messages`. Reference: https://github.com/openai/openai-python/blob/195c05a64d39c87b2dfdf1eca2d339597f1fce03/src/openai/resources/completions.py#L45

## v0.1.9

Added:
    - Support for default CPU tinyBLAS (llamafile, sgemm) builds.
    - Support for CPU OpenBLAS (GGML_OPENBLAS) builds.

Changed:
    - Build scripts now have separate step/function `cuda_12_5_1_setup` which setups CUDA 12.5.1 env for build-time.

Fixed:
    - Stop thread in `llama_generate` on `GeneratorExit`.

Removed:
    - `callback` parameter in `llama_generate` and dependent functions.

## v0.1.8

Added:
    - `Model.tokenizer_hf_repo` as optional in case when `Model.creator_hf_repo` cannot be used to tokenize / format prompt/messages.

## v0.1.7

Added:
    - Support for `stop` tokens/words.

Changed:
    - `llama/llama_cli.py` unified CPU and CUDA 12.5 modules into single module.

Removed:
    - Removed separate examples for CPU and CUDA 12.5 modules.

## v0.1.6

Changed:
    - Updated `huggingface-hub`.

Fixed:
    - `llama.__init__` now correctly imports submodules and handles CPU and CUDA backends.
    - OpenAI: `ctx_size: int = config.max_position_embeddings if max_tokens is None else max_tokens`.

## v0.1.5

Fixed:
    - Build for linux, upx uses best compression option, 7z uses more aggressive compression.
    - Do not use UPX for shared/dynamic library compression.

## v0.1.4

Added:
    - README: supported GPU Compute Capability for CUDA.

Fixed:
    - Cleaned up `build.py`.
    - Type annotations in OpenAI related code.

## v0.1.3

Added:
    - Support for PyPy 3.10 versions.

Changed:
    - Disabled GitHub Actions.
    - Uses `upx -9` to compress shared/dynamic libraries.
    - Repacks `whl` with better compression rate.
    - Auto-detect CUDA support.

Removed:
    - ctypes version and demos.

## v0.1.2

Added:
    - Preparation for [Chat Completions API by OpenAI ©](https://platform.openai.com/docs/overview) compatible server.

Fixed:
    - Argument `options` is `deepcopy`-ed when passed to `llama_generate(options)`, so it can be reused.

Changed:
    - Build for `manylinux_2_28` and `musllinux_1_2`.
    - Build for [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 6.1.

## v0.1.1

Changed:
    - Updated: `huggingface-hub = "^0.24.0"`, `setuptools = "^71.0.3"`

## v0.1.0

Added:
    - Park first version.
