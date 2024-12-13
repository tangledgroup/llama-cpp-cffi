__all__ = [
    # high-level API
    # 'completions',

    # low-level API
    'lib',
    'ffi',
    'backend_init',
    'backend_free',
    'numa_init',
    'model_init',
    'model_free',
    'context_init',
    'context_free',
    'sampler_init',
    'sampler_free',
    'clip_init_context',
    'clip_free_context',
    'text_completions',
    'clip_completions',
]

import os
import atexit
from typing import Iterator, NewType
from threading import Lock
from weakref import WeakKeyDictionary

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from .options import Options
from .util import is_cuda_available, is_vulkan_available
from .formatter import get_tokenizer, format_messages


os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'true')
os.environ['GGML_VK_DISABLE_COOPMAT'] = os.getenv('GGML_VK_DISABLE_COOPMAT', '1')

LLAMA_CPP_BACKEND = os.getenv('LLAMA_CPP_BACKEND', None)
LLAMA_SPLIT_MODE_NONE = 0 # single GPU
LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
LLAMA_SPLIT_MODE_ROW = 2 # split layers and KV across GPUs, use tensor parallelism if supported

try:
    if LLAMA_CPP_BACKEND:
        if LLAMA_CPP_BACKEND in ('cuda', 'CUDA'):
            from ._llama_cpp_cuda_12_6_3 import lib, ffi
        elif LLAMA_CPP_BACKEND in ('vulkan', 'VULKAN'):
            from ._llama_cpp_vulkan_1_x import lib, ffi
        elif LLAMA_CPP_BACKEND in ('cpu', 'CPU'):
            from ._llama_cpp_cpu import lib, ffi
        else:
            raise ValueError(f'{LLAMA_CPP_BACKEND = }')
    else:
        if is_cuda_available():
            from ._llama_cpp_cuda_12_6_3 import lib, ffi
        elif is_vulkan_available():
            from ._llama_cpp_vulkan_1_x import lib, ffi
        else:
            from ._llama_cpp_cpu import lib, ffi
except ImportError:
    from ._llama_cpp_cpu import lib, ffi


global_weakkeydict = WeakKeyDictionary()

#
# low-level API
#
void_p = NewType('void*', ffi.typeof('void*'))
char_p = NewType('char*', ffi.typeof('char*'))
int_p = NewType('int*', ffi.typeof('int*'))
float_p = NewType('float*', ffi.typeof('float*'))
ggml_log_level = NewType('ggml_log_level', ffi.typeof('enum ggml_log_level'))
ggml_numa_strategy = NewType('ggml_numa_strategy', ffi.typeof('enum ggml_numa_strategy'))
llama_model_params = NewType('llama_model_params', ffi.typeof('struct llama_model_params'))
llama_model = NewType('llama_model', ffi.typeof('struct llama_model'))
llama_model_p = NewType('llama_model*', ffi.typeof('struct llama_model*'))
llama_context = NewType('llama_context', ffi.typeof('struct llama_context'))
llama_context_p = NewType('llama_context*', ffi.typeof('struct llama_context*'))
llama_context_params = NewType('llama_context_params', ffi.typeof('struct llama_context_params'))
llama_sampler = NewType('llama_sampler', ffi.typeof('struct llama_sampler'))
llama_sampler_p = NewType('llama_sampler*', ffi.typeof('struct llama_sampler*'))
llama_sampler_chain_params = NewType('llama_sampler_chain_params', ffi.typeof('struct llama_sampler_chain_params'))
llama_batch = NewType('llama_batch', ffi.typeof('struct llama_batch'))
llama_pos = NewType('llama_pos', ffi.typeof('int32_t'))
llama_token = NewType('llama_token', ffi.typeof('int32_t'))
llama_seq_id = NewType('llama_seq_id', ffi.typeof('int32_t'))
llama_token_data = NewType('llama_token_data', ffi.typeof('struct llama_token_data'))
llama_token_data_p = NewType('llama_token_data*', ffi.typeof('struct llama_token_data*'))
llama_token_data_array = NewType('llama_token_data_array', ffi.typeof('struct llama_token_data_array'))
llama_token_data_array_p = NewType('llama_token_data_array*', ffi.typeof('struct llama_token_data_array*'))
clip_ctx = NewType('clip_ctx', ffi.typeof('struct clip_ctx'))
clip_ctx_p = NewType('clip_ctx*', ffi.typeof('struct clip_ctx*'))
llava_image_embed = NewType('llava_image_embed', ffi.typeof('struct llava_image_embed'))
llava_image_embed_p = NewType('llava_image_embed*', ffi.typeof('struct llava_image_embed*'))

lock = Lock()

# Set callback for all future logging events.
# If this is not called, or NULL is supplied, everything is output on stderr.
#
# LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);
#
# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
@ffi.def_extern()
def llama_cpp_cffi_ggml_log_callback(level: ggml_log_level, text: char_p, user_data: void_p):
    pass

# disable logs by default
lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)


def backend_init():
    with lock:
        lib.llama_backend_init()


def backend_free():
    with lock:
        lib.llama_backend_free()


def numa_init(numa: ggml_numa_strategy):
    with lock:
        lib.llama_numa_init(numa)


def model_init(options: Options) -> llama_model_p:
    model_path = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.hf_file)
    # print(f'{model_path=}')

    model_params: llama_model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = options.gpu_layers
    # model_params.split_mode = options.split_mode # FIXME: check Options
    model_params.main_gpu = options.main_gpu
    model_params.use_mmap = not options.no_mmap
    model_params.use_mlock = options.mlock
    model_params.check_tensors = options.check_tensors

    with lock:
        model: llama_model_p = lib.llama_load_model_from_file(model_path.encode(), model_params)

    return model


def model_free(model: llama_model_p):
    with lock:
        lib.llama_free_model(model)


def context_init(model: llama_model_p, options: Options) -> llama_context_p:
    ctx_params: llama_context_params = lib.llama_context_default_params()
    ctx_params.n_ctx = options.ctx_size
    ctx_params.n_batch = options.batch_size
    ctx_params.n_ubatch = options.ubatch_size
    ctx_params.n_threads = options.threads

    if options.threads_batch is None:
        ctx_params.n_threads_batch = options.threads
    else:
        ctx_params.n_threads_batch = options.threads_batch

    with lock:
        context: llama_context_p = lib.llama_new_context_with_model(model, ctx_params)

    return context


def context_free(context: llama_context_p):
    with lock:
        lib.llama_free(context)


def sampler_init(model: llama_model_p, options: Options) -> llama_sampler_p:
    sampler_params: llama_sampler_chain_params = lib.llama_sampler_chain_default_params()
    sampler: llama_sampler_p = lib.llama_sampler_chain_init(sampler_params)

    # dry
    seq_breakers: char_p = ffi.new('char*[]', options.dry_sequence_breaker)
    num_breakers: int = len(options.dry_sequence_breaker)

    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dry(
        model,
        options.dry_multiplier,
        options.dry_base,
        options.dry_allowed_length,
        options.dry_penalty_last_n,
        seq_breakers,
        num_breakers,
    ))

    # common
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp(options.temp))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(options.top_k))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_p(options.top_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_min_p(options.min_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(options.seed))

    # penalties
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_penalties(
        lib.llama_n_vocab(model),
        lib.llama_token_eos(model),
        lib.llama_token_nl(model),
        options.repeat_last_n,      # last n tokens to penalize (0 = disable penalty, -1 = context size)
        options.repeat_penalty,     # 1.0 = disabled
        options.frequency_penalty,  # 0.0 = disabled
        options.presence_penalty,   # 0.0 = disabled
        options.penalize_nl,        # consider newlines as a repeatable token
        options.ignore_eos,         # ignore the end-of-sequence token
    ))

    return sampler


def grammar_sampler_init(model: llama_model_p, options: Options) -> llama_sampler_p:
    # grammar
    if options.grammar:
        grammar_str: char_p = ffi.new('char[]', options.grammar.encode())
        grammar_root: char_p = ffi.new('char[]', b'root')
    elif options.json_schema:
        if options.json_schema == '{}':
            grammar = r'''
                array ::= "[" space ( value ("," space value)* )? "]" space
                boolean ::= ("true" | "false") space
                char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
                decimal-part ::= [0-9]{1,16}
                integral-part ::= [0] | [1-9] [0-9]{0,15}
                null ::= "null" space
                number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
                object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
                space ::= | " " | "\n" [ \t]{0,20}
                string ::= "\"" char* "\"" space
                value ::= object | array | string | number | boolean | null
                root ::= object
            '''
        else:
            _c_value: char_p = ffi.new('char[]', options.json_schema.encode())
            _grammar: char_p = lib.llama_json_schema_to_grammar(_c_value)
            grammar: str = ffi.string(_grammar).decode()
            lib.free(_grammar)
            ffi.release(_c_value)
            # print('grammar:')
            # print(grammar)

        grammar_str: char_p = ffi.new('char[]', grammar.encode())
        grammar_root: char_p = ffi.new('char[]', b'root')

    grmr: llama_sampler_p = lib.llama_sampler_init_grammar(
        model,
        grammar_str,
        grammar_root,
    )

    return grmr


def sampler_free(sampler: llama_sampler_p):
    lib.llama_sampler_free(sampler)


def clip_init_context(options: Options) -> clip_ctx_p:
    mmproj_path: str = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.mmproj_hf_file)

    with lock:
        clip_context: clip_ctx_p = lib.clip_model_load(mmproj_path.encode(), 0) # path, verbosity

    return clip_context


def clip_free_context(clip_context: clip_ctx_p):
    with lock:
        lib.clip_free(clip_context)


#
# util
#
def _llama_decode(ctx: llama_context_p, batch : llama_batch) -> int:
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


#
# llm
#
def text_completions(model: llama_model_p, options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str) or isinstance(options.messages, list)

    if options.verbose:
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(model, options)
    sampler = sampler_init(model, options)
    # print(f'{sampler=}')

    if options.grammar or options.json_schema:
        grammar_sampler = grammar_sampler_init(model, options)
    else:
        grammar_sampler = None

    # print(f'{grammar_sampler=}')

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # format messages if present into prompt
    if options.messages:
        options.prompt = format_messages(tokenizer, options.messages, options)

    # tokenize prompt
    prompt_tokens: list[int] = tokenizer.encode(options.prompt)
    n_prompt_tokens: int = len(prompt_tokens)
    # print(f'{n_prompt_tokens=}')

    # create a llama_batch, we use this object to submit token data for decoding
    n_batch: int = lib.llama_n_batch(context)
    batch: llama_batch = lib.llama_batch_init(n_batch, 0, 1)
    seq_ids: list[llama_seq_id] = [0]
    n_past: int = 0

    # evaluate the initial prompt
    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, n_past, n_past)

    # generate output
    n_cur: int = n_prompt_tokens
    n_ctx: int = lib.llama_n_ctx(context)
    n_decode: int = 0

    while n_cur < n_ctx and n_decode < options.predict:
        if grammar_sampler:
            new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
            _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)
        else:
            # new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)
            # new_token_id: llama_token = _llama_sampler_sample(sampler, context, -1)
            new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)

        n_decode += 1

        if lib.llama_token_is_eog(model, new_token_id):
            break

        piece: str = tokenizer.decode([new_token_id])
        yield piece

        _common_batch_clear(batch)
        _common_batch_add(batch, new_token_id, n_past, seq_ids, True)
        batch.logits[batch.n_tokens] = True
        n_past += 1
        n_cur += 1

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

    # lib.llama_perf_sampler_print(sampler)
    # lib.llama_perf_context_print(context)

    lib.llama_batch_free(batch)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    context_free(context)


#
# clip
#
def _llava_image_embed_make_with_filename(ctx_clip: clip_ctx_p, n_threads: int, image_path: bytes) -> llava_image_embed_p:
    with lock:
        return lib.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)


def _llava_image_embed_free(embed: llava_image_embed_p):
    with lock:
        lib.llava_image_embed_free(embed)


def _clip_process_eval_image_embed(context: llama_context_p,
                                   clip_context: clip_ctx_p,
                                   embeds: llava_image_embed_p,
                                   n_past: int,
                                   idx: int) -> int:
    n_past_p: int_p = ffi.new('int[]', [n_past])

    with lock:
        n_batch: int = lib.llama_n_batch(context)
        image_embed: void_p = lib.malloc(lib.clip_embd_nbytes(clip_context))
        image_embed: float_p = ffi.cast('float*', image_embed)

        lib.memcpy(
            image_embed,
            embeds.embed + idx * lib.clip_n_patches(clip_context) * lib.clip_n_mmproj_embd(clip_context),
            lib.clip_embd_nbytes(clip_context),
        )

        slice_embed: void_p = lib.malloc(ffi.sizeof('struct llava_image_embed'))
        slice_embed: llava_image_embed_p = ffi.cast('struct llava_image_embed*', slice_embed)
        slice_embed.embed = image_embed
        slice_embed.n_image_pos = lib.clip_n_patches(clip_context)

        lib.llava_eval_image_embed(context, slice_embed, n_batch, n_past_p)
        lib.llava_image_embed_free(slice_embed)

    n_past = n_past_p[0]
    ffi.release(n_past_p)
    return n_past


def _clip_uhd_num_image_embeds_col(ctx_clip: clip_ctx_p) -> int:
    with lock:
        return lib.clip_uhd_num_image_embeds_col(ctx_clip)


def clip_completions(model: llama_model_p, options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str)
    assert isinstance(options.image, str)
    assert options.messages is None

    if options.verbose:
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(model, options)
    clip_context = clip_init_context(options)
    sampler = sampler_init(model, options)
    print(f'{sampler=}')

    if options.grammar or options.json_schema:
        grammar_sampler = grammar_sampler_init(model, options)
    else:
        grammar_sampler = None

    print(f'{grammar_sampler=}')

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # create a llama_batch, we use this object to submit token data for decoding
    n_batch: int = lib.llama_n_batch(context)
    batch: llama_batch = lib.llama_batch_init(n_batch, 0, 1)
    seq_ids: list[llama_seq_id] = [0]
    n_past: int = 0
    idx: int = 0
    prompt: str
    prompt_tokens: list[int]

    # image embeddings
    embeds: llava_image_embed_p = _llava_image_embed_make_with_filename(clip_context, options.threads, options.image.encode())

    # pre-embeds prompt
    messages = [{'role': 'user', 'content': '<image>'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = prompt[:prompt.index('<image>') + len('<image>')]
    prompt_tokens = tokenizer.encode(prompt)
    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

    # embeds
    n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

    idx += 1

    # evaluate the post-embeds prompt
    prompt = '</image>'
    prompt_tokens = tokenizer.encode(prompt)
    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

    has_minicpmv_projector = lib.clip_is_minicpmv(clip_context)
    # print(f'{has_minicpmv_projector=}')

    if has_minicpmv_projector >= 2:
        num_image_embeds = embeds.n_image_pos / lib.clip_n_patches(clip_context)

        if num_image_embeds > 1:
            num_image_embeds_col = _clip_uhd_num_image_embeds_col(clip_context)

            prompt = '<slice>'
            prompt_tokens = tokenizer.encode(prompt)
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
            i = 0

            while i < (num_image_embeds - 1) / num_image_embeds_col:
                j = 0

                while j < num_image_embeds_col:
                    prompt = '<image>'
                    prompt_tokens = tokenizer.encode(prompt)
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

                    idx += 1

                    prompt = '</image>'
                    prompt_tokens = tokenizer.encode(prompt)
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    j += 1

                prompt = '\n'
                prompt_tokens = tokenizer.encode(prompt)
                n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                i += 1

            prompt = '</slice>'
            prompt_tokens = tokenizer.encode(prompt)
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

        _llava_image_embed_free(embeds)

        # user prompt after image ebeds
        messages = [{'role': 'user', 'content': f'\n{options.prompt}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt[prompt.index(f'\n{options.prompt}'):]
        prompt_tokens = tokenizer.encode(prompt)
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

    # generate output
    n_cur: int = n_past
    n_ctx: int = lib.llama_n_ctx(context)
    n_decode: int = 0
    # print(f'{n_cur=} {n_ctx=} {n_decode=} {options.predict=}')

    while n_cur < n_ctx and n_decode < options.predict:
        if grammar_sampler:
            new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
            _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)
        else:
            # new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)
            # new_token_id: llama_token = _llama_sampler_sample(sampler, context, -1)
            new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)

        n_decode += 1

        if lib.llama_token_is_eog(model, new_token_id):
            if n_decode > 1:
                break

        if not lib.llama_token_is_eog(model, new_token_id):
            piece: str = tokenizer.decode([new_token_id])
            yield piece

        # with lock:
        _common_batch_clear(batch)
        _common_batch_add(batch, new_token_id, n_past, seq_ids, True)
        batch.logits[batch.n_tokens] = True
        n_past += 1
        n_cur += 1

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

    # lib.llama_perf_sampler_print(sampler)
    # lib.llama_perf_context_print(context)

    lib.llama_batch_free(batch)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)


#
# backend
#
backend_init()
atexit.register(backend_free)
