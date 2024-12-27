
def context_init(model: llama_model_p, model_options: ModelOptions) -> llama_context_p:
    ctx_params: llama_context_params = lib.llama_context_default_params()
    ctx_params.n_ctx = model_options.ctx_size # TODO: use exact field names like in structs/API
    ctx_params.n_batch = model_options.batch_size
    ctx_params.n_ubatch = model_options.ubatch_size
    ctx_params.n_threads = model_options.threads

    # TODO: use exact field names like in structs/API
    if model_options.threads_batch is None:
        ctx_params.n_threads_batch = model_options.threads
    else:
        ctx_params.n_threads_batch = model_options.threads_batch

    with lock:
        context: llama_context_p = lib.llama_new_context_with_model(model, ctx_params)

    return context


def context_free(context: llama_context_p):
    with lock:
        lib.llama_free(context)
