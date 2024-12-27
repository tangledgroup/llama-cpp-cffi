__all__ = ['is_cuda_available', 'is_vulkan_available']

try:
    from numba import cuda
except Exception:
    pass

try:
    import vulkan as vk
except Exception:
    pass


def is_cuda_available():
    r: bool = False

    try:
        r = cuda.is_available()
    except Exception:
        r = False

    return r


def is_vulkan_available():
    vulkan_available: bool = False

    try:
        # Load the Vulkan library and create an instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Vulkan Check",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )

        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )

        # Try creating a Vulkan instance
        instance = vk.vkCreateInstance(instance_info, None)

        # If we reach this point, Vulkan is available
        vulkan_available = True

        # Clean up the Vulkan instance
        vk.vkDestroyInstance(instance, None)
    except Exception:
        pass

    return vulkan_available


def numa_init(numa: ggml_numa_strategy):
    with lock:
        lib.llama_numa_init(numa)


def _llama_decode(ctx: llama_context_p, batch: llama_batch) -> int:
    with lock:
        return lib.llama_decode(ctx, batch)


def _set_logits(ctx: llama_context_p, idx: int):
    logits: float_p = lib.llama_get_logits_ith(ctx, idx)
    n_vocab: int = lib.llama_n_vocab(lib.llama_get_model(ctx))

    cur: llama_token_data_p = ffi.new(
        'llama_token_data[]',
        [(token_id, logits[token_id], 0.0) for token_id in range(n_vocab)],
    )

    cur_p: llama_token_data_array_p = ffi.new('llama_token_data_array*', [cur, n_vocab, -1, False])
    global_weakkeydict[cur_p] = cur
    return cur, cur_p


def _llama_sampler_sample(smpl: llama_sampler_p, ctx: llama_context_p, idx: int):
    # reimplementation of C code
    cur, cur_p = _set_logits(ctx, idx)
    lib.llama_sampler_apply(smpl, cur_p)
    token: llama_token = cur_p.data[cur_p.selected].id
    lib.llama_sampler_accept(smpl, token)
    return token


def _common_sampler_sample(grmr: llama_sampler_p, chain: llama_sampler_p, ctx: llama_context_p, idx: int, grammar_first):
    cur, cur_p = _set_logits(ctx, idx)

    if grammar_first:
        lib.llama_sampler_apply(grmr, cur_p)

    lib.llama_sampler_apply(chain, cur_p)
    assert cur_p.selected != -1, "no selected token during sampling - check your sampling configuration"

    id: llama_token = cur_p.data[cur_p.selected].id
    # print(f'{id=}')

    if grammar_first:
        return id

    # check if it the sampled token fits the grammar
    single_token_data: llama_token_data_p = ffi.new(
        'llama_token_data*',
        [id, 1.0, 0.0],
    )

    single_token_data_array: llama_token_data_array_p = ffi.new(
        'llama_token_data_array*',
        [single_token_data, 1, -1, False]
    )

    global_weakkeydict[single_token_data_array] = single_token_data
    lib.llama_sampler_apply(grmr, single_token_data_array)

    # print(f'{single_token_data_array.data[0].logit=}')
    is_valid: bool = single_token_data_array.data[0].logit != float('-inf')

    if is_valid:
        # print(f'{id=} {is_valid=}')
        return id

    # resampling
    cur, cur_p = _set_logits(ctx, idx)
    lib.llama_sampler_apply(grmr,  cur_p)
    lib.llama_sampler_apply(chain, cur_p)
    assert cur_p.selected != -1, "no selected token during re-sampling - check your sampling configuration"

    id: llama_token = cur_p.data[cur_p.selected].id
    return id


def _common_sampler_accept(grmr: llama_sampler_p, chain: llama_sampler_p, token: llama_token, accept_grammar: bool):
    if accept_grammar:
        lib.llama_sampler_accept(grmr, token)

    lib.llama_sampler_accept(chain, token)


def _common_batch_clear(batch: llama_batch):
    batch.n_tokens = 0


def _common_batch_add(batch: llama_batch, id: llama_token, pos: llama_pos, seq_ids: list[llama_seq_id], logits: bool):
    assert batch.seq_id[batch.n_tokens], "llama_batch size exceeded"
    batch.token[batch.n_tokens] = id
    batch.pos[batch.n_tokens] = pos
    batch.n_seq_id[batch.n_tokens] = len(seq_ids)

    for i in range(len(seq_ids)):
        batch.seq_id[batch.n_tokens][i] = seq_ids[i]

    batch.logits[batch.n_tokens] = logits
    batch.n_tokens += 1


def _common_token_to_piece(ctx: llama_context_p, token: llama_token, special: bool) -> str:
    model: llama_model_p = lib.llama_get_model(ctx)
    _piece_size: int = 128
    _piece: char_p = ffi.new('char[]', _piece_size)
    n_chars: int = lib.llama_token_to_piece(model, token, _piece, _piece_size, 0, special)
    piece: bytes | str = ffi.string(_piece)
    assert isinstance(piece, bytes)

    try:
        piece = piece.decode()
    except Exception:
        piece = ''

    assert isinstance(piece, str)
    piece = piece[:n_chars]
    ffi.release(_piece)
    return piece


def _decode_tokens(context: llama_context_p, batch: llama_batch, prompt_tokens: list[int], seq_ids: list[llama_seq_id], n_begin: int, n_past: int) -> int:
    n_batch: int = lib.llama_n_batch(context)
    n_prompt_tokens: int = len(prompt_tokens)

    for i in range(n_begin, n_prompt_tokens, n_batch):
        _common_batch_clear(batch)
        j = 0

        while j < n_batch and i + j < n_prompt_tokens:
            _common_batch_add(batch, prompt_tokens[i + j], n_past, seq_ids, False)
            n_past += 1
            j += 1

        if i + n_batch >= n_prompt_tokens:
            batch.logits[batch.n_tokens - 1] = True

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

        if i + n_batch >= n_prompt_tokens:
            break

        # lib.llama_kv_cache_seq_cp(context, 0, i, 0, batch.n_tokens)

    return n_past
