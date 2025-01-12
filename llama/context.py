__all__ = [
    'context_init',
    'context_free',
]

from .llama_cpp import lib, ffi, lock, llama_model_p, llama_context_p, llama_context_params
from .options import ModelOptions


def context_init(model: llama_model_p, model_options: ModelOptions) -> llama_context_p:
    ctx_params: llama_context_params = lib.llama_context_default_params()
    ctx_params.n_ctx = model_options.n_ctx
    ctx_params.n_batch = model_options.n_batch
    ctx_params.n_ubatch = model_options.n_ubatch
    ctx_params.n_threads = model_options.n_threads

    if model_options.n_threads_batch is None:
        ctx_params.n_threads_batch = model_options.n_threads
    else:
        ctx_params.n_threads_batch = model_options.n_threads_batch

    ctx_params.type_k = model_options.type_k.value
    ctx_params.type_v = model_options.type_v.value
    ctx_params.flash_attn = model_options.flash_attn

    with lock:
        context: llama_context_p = lib.llama_new_context_with_model(model, ctx_params)

    return context


def context_free(context: llama_context_p):
    with lock:
        lib.llama_free(context)
