# CHANGELOG

## v0.1.2

Added:
    - OpenAI © compatible Chat Completions API server.

Fixed:
    - `options` is deepcopy-ed when passed to `llama_generate(options)`, so it can be reused.

Changed:
    - Build for manylinux_2_28 and musllinux_1_2

## v0.1.1

Changed:
    - Updated: huggingface-hub = "^0.24.0", setuptools = "^71.0.3"

## v0.1.0

Added:
    - Park first version.
