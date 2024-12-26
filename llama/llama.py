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
import json
import atexit
from pprint import pprint
from typing import Any, Iterator, TypeAlias
from threading import Lock
from weakref import WeakKeyDictionary

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from .options import ModelOptions, CompletionsOptions
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
void_p: TypeAlias = ffi.typeof('void*') # type: ignore
char_p: TypeAlias = ffi.typeof('char*') # type: ignore
int_p: TypeAlias = ffi.typeof('int*') # type: ignore
float_p: TypeAlias = ffi.typeof('float*') # type: ignore
ggml_log_level: TypeAlias = ffi.typeof('enum ggml_log_level') # type: ignore
ggml_numa_strategy: TypeAlias = ffi.typeof('enum ggml_numa_strategy') # type: ignore
llama_model_params: TypeAlias = ffi.typeof('struct llama_model_params') # type: ignore
llama_model: TypeAlias = ffi.typeof('struct llama_model') # type: ignore
llama_model_p: TypeAlias = ffi.typeof('struct llama_model*') # type: ignore
llama_context: TypeAlias = ffi.typeof('struct llama_context') # type: ignore
llama_context_p: TypeAlias = ffi.typeof('struct llama_context*') # type: ignore
llama_context_params: TypeAlias = ffi.typeof('struct llama_context_params') # type: ignore
llama_sampler: TypeAlias = ffi.typeof('struct llama_sampler') # type: ignore
llama_sampler_p: TypeAlias = ffi.typeof('struct llama_sampler*') # type: ignore
llama_sampler_chain_params: TypeAlias = ffi.typeof('struct llama_sampler_chain_params') # type: ignore
llama_batch: TypeAlias = ffi.typeof('struct llama_batch') # type: ignore
llama_batch_p: TypeAlias = ffi.typeof('struct llama_batch*') # type: ignore
llama_pos: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_token: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_seq_id: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_token_data: TypeAlias = ffi.typeof('struct llama_token_data') # type: ignore
llama_token_data_p: TypeAlias = ffi.typeof('struct llama_token_data*') # type: ignore
llama_token_data_array: TypeAlias = ffi.typeof('struct llama_token_data_array') # type: ignore
llama_token_data_array_p: TypeAlias = ffi.typeof('struct llama_token_data_array*') # type: ignore
clip_ctx: TypeAlias = ffi.typeof('struct clip_ctx') # type: ignore
clip_ctx_p: TypeAlias = ffi.typeof('struct clip_ctx*') # type: ignore
clip_image_size: TypeAlias = ffi.typeof('struct clip_image_size') # type: ignore
clip_image_size_p: TypeAlias = ffi.typeof('struct clip_image_size*') # type: ignore
llava_image_embed: TypeAlias = ffi.typeof('struct llava_image_embed') # type: ignore
llava_image_embed_p: TypeAlias = ffi.typeof('struct llava_image_embed*') # type: ignore

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


def model_init(model_options: ModelOptions) -> llama_model_p:
    model_path = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.hf_file)
    print(f'{model_path=}')

    model_params: llama_model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = model_options.gpu_layers
    # model_params.split_mode = model_options.split_mode # FIXME: check Options
    model_params.main_gpu = model_options.main_gpu
    model_params.use_mmap = not model_options.no_mmap # TODO: use exact field names like in structs/API
    model_params.use_mlock = model_options.mlock
    model_params.check_tensors = model_options.check_tensors

    with lock:
        model: llama_model_p = lib.llama_load_model_from_file(model_path.encode(), model_params)

    return model


def model_free(model: llama_model_p):
    with lock:
        lib.llama_free_model(model)


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


def sampler_init(model: llama_model_p, completions_options: CompletionsOptions) -> llama_sampler_p:
    sampler_params: llama_sampler_chain_params = lib.llama_sampler_chain_default_params()
    sampler: llama_sampler_p = lib.llama_sampler_chain_init(sampler_params)

    # common
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_logit_bias(
        lib.llama_n_vocab(model),
        0,
        ffi.NULL,
    ))

    # dry
    seq_breakers: char_p = ffi.new('char*[]', completions_options.dry_sequence_breaker)
    num_breakers: int = len(completions_options.dry_sequence_breaker)

    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dry(
        model,
        completions_options.dry_multiplier,
        completions_options.dry_base,
        completions_options.dry_allowed_length,
        completions_options.dry_penalty_last_n,
        seq_breakers,
        num_breakers,
    ))

    # common
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(completions_options.top_k))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_p(completions_options.top_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_min_p(completions_options.min_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp(completions_options.temp))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(completions_options.seed))

    # penalties
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_penalties(
        completions_options.repeat_last_n,      # last n tokens to penalize (0 = disable penalty, -1 = context size)
        completions_options.repeat_penalty,     # 1.0 = disabled
        completions_options.frequency_penalty,  # 0.0 = disabled
        completions_options.presence_penalty,   # 0.0 = disabled
    ))

    return sampler


def grammar_sampler_init(model: llama_model_p, completions_options: CompletionsOptions) -> llama_sampler_p:
    # grammar
    grammar: bytes
    grammar_str: char_p
    grammar_root: char_p

    if completions_options.grammar:
        if isinstance(completions_options.grammar, str):
            grammar = completions_options.grammar.encode()
        elif isinstance(completions_options.grammar, str):
            grammar = completions_options.grammar
        else:
            raise ValueError(f'unsupported value grammar={completions_options.grammar}')
    elif completions_options.json_schema:
        if completions_options.json_schema in ('{}', b'{}', {}):
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
            '''.encode()
        else:
            json_schema: bytes

            if isinstance(completions_options.json_schema, str):
                json_schema = completions_options.json_schema.encode()
            elif isinstance(completions_options.json_schema, bytes):
                json_schema = completions_options.json_schema
            elif isinstance(completions_options.json_schema, dict):
                json_schema = json.dumps(completions_options.json_schema).encode()
            else:
                raise ValueError(f'unsupported value json_schema={completions_options.json_schema}')

            _c_value: char_p = ffi.new('char[]', json_schema)
            _grammar: char_p = lib.llama_json_schema_to_grammar(_c_value)
            grammar = ffi.string(_grammar)
            lib.free(_grammar)
            ffi.release(_c_value)
    else:
        raise ValueError('either grammar or json_schema is required')

    grammar_str = ffi.new('char[]', grammar)
    grammar_root = ffi.new('char[]', b'root')

    grmr: llama_sampler_p = lib.llama_sampler_init_grammar(
        model,
        grammar_str,
        grammar_root,
    )

    return grmr


def sampler_free(sampler: llama_sampler_p):
    lib.llama_sampler_free(sampler)


def clip_init_context(model_options: ModelOptions) -> clip_ctx_p:
    assert model_options.ctx_size >= 2048
    assert model_options.mmproj_hf_file
    mmproj_path: str | bytes = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.mmproj_hf_file)
    print(f'{mmproj_path=}')

    with lock:
        # clip_model_load(path, verbosity)
        clip_context: clip_ctx_p = lib.clip_model_load(mmproj_path.encode(), model_options.verbose)

    return clip_context


def clip_free_context(clip_context: clip_ctx_p):
    with lock:
        lib.clip_free(clip_context)


#
# util
#
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


def _qwen2vl_decode_tokens(context: llama_context_p, batch: llama_batch, prompt_tokens: list[int], seq_ids: list[llama_seq_id], n_begin: int, n_past: int, st_pos_id: int) -> tuple[int, int]:
    n_batch: int = lib.llama_n_batch(context)
    n_prompt_tokens: int = len(prompt_tokens)


    for i in range(n_begin, n_prompt_tokens, n_batch):
        _common_batch_clear(batch)

        j = 0

        while j < n_batch and i + j < n_prompt_tokens:
            _common_batch_add(batch, prompt_tokens[i + j], n_past, seq_ids, False)
            n_past += 1
            st_pos_id += 1
            j += 1

        if i + n_batch >= n_prompt_tokens:
            batch.logits[batch.n_tokens - 1] = True

        j = 0

        print(f'!!! {batch.n_tokens=}, {batch.n_tokens * 4=}')

        pos: Any = ffi.new('llama_pos[]', [
            int(st_pos_id + (j % batch.n_tokens))
            for j in range(batch.n_tokens * 4)
        ])

        print(f'{pos=}')
        batch.pos = pos

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

        if i + n_batch >= n_prompt_tokens:
            break

        ffi.release(pos)

    return n_past, st_pos_id


#
# llm
#
def text_completions(model: llama_model_p, model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    assert isinstance(completions_options.prompt, str) or isinstance(completions_options.messages, list)

    if completions_options.verbose:
        # default llama.cpp logger
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(model, model_options)
    sampler = sampler_init(model, completions_options)
    # print(f'{sampler=}')

    if completions_options.grammar or completions_options.json_schema:
        grammar_sampler = grammar_sampler_init(model, completions_options)
    else:
        grammar_sampler = None

    # print(f'{grammar_sampler=}')

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)

    # format messages if present into prompt
    if completions_options.messages:
        completions_options.prompt = format_messages(
            tokenizer,
            completions_options.messages,
            completions_options,
        )

    # tokenize prompt
    prompt_tokens: list[int] = tokenizer.encode(completions_options.prompt) # type: ignore
    # print('!', tokenizer.decode(prompt_tokens))
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
    output_tokens: list[int] = []

    while n_cur < n_ctx and n_decode < completions_options.predict:
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

        output_tokens.append(new_token_id)

        # piece: str = tokenizer.decode(new_token_id)
        piece = _common_token_to_piece(context, new_token_id, True)
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

    # print()
    # print('!!', tokenizer.decode(output_tokens))
    # print('!!', output_tokens)
    # print('!!', [(n, tokenizer.decode(n)) for n in output_tokens])

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
        # return lib.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)
        embed: llava_image_embed_p = lib.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)

    print(f'{embed=}')
    return embed


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


def _qwen2vl_eval_image_embed(ctx_llama: llama_context_p, ctx_clip: clip_ctx_p, image_embed: llava_image_embed_p, n_batch: int, n_past: int, st_pos_id: int) -> tuple[bool, int, int]:
    print('!!! [0]:', n_past, st_pos_id)
    image_size: Any = lib.clip_get_load_image_size(ctx_clip)
    n_embd: int = lib.llama_n_embd(lib.llama_get_model(ctx_llama))
    patch_size: int = 14 * 2
    ph: int = int(image_size.height / patch_size + (image_size.height % patch_size > 0))
    pw: int = int(image_size.width / patch_size + (image_size.width % patch_size > 0))
    img_tokens: Any = image_embed.n_image_pos
    mrope_pos: Any = ffi.new('llama_pos[]', img_tokens * 4)
    pprint(locals())

    for y in range(ph):
        for x in range(pw):
            i: int = y * pw + x
            mrope_pos[i + img_tokens * 0] = st_pos_id
            mrope_pos[i + img_tokens * 1] = st_pos_id + y
            mrope_pos[i + img_tokens * 2] = st_pos_id + x
            mrope_pos[i + img_tokens * 3] = 0

    st_pos_id += max(pw, ph)
    processed: int = 0
    batch_mrope_pos: Any = ffi.new('llama_pos[]', img_tokens * 4)
    pprint(locals())
    print('!!! [*]:', n_past, st_pos_id)

    for i in range(0, img_tokens, n_batch):
        n_eval: int = img_tokens - i

        if n_eval > n_batch:
            n_eval = n_batch

        # for i in range(img_tokens * 4):
        #     batch_mrope_pos[i] = 0

        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 0), ffi.addressof(mrope_pos, img_tokens * 0 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 1), ffi.addressof(mrope_pos, img_tokens * 1 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 2), ffi.addressof(mrope_pos, img_tokens * 2 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 3), ffi.addressof(mrope_pos, img_tokens * 3 + processed), n_eval * ffi.sizeof('llama_pos'))

        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 0), ffi.addressof(mrope_pos, img_tokens * 0 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 1), ffi.addressof(mrope_pos, img_tokens * 1 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 2), ffi.addressof(mrope_pos, img_tokens * 2 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 3), ffi.addressof(mrope_pos, img_tokens * 3 + processed), n_eval * ffi.sizeof('llama_pos'))

        embd: Any = image_embed.embed + i * n_embd
        # embd[0] = embd[0]
        # batch_mrope_pos[0] = batch_mrope_pos[0]
        # global_weakkeydict[image_embed] = embd
        print(f'{n_eval=}')
        print(f'{embd=}')
        print(f'{batch_mrope_pos=}')

        batch_p: llama_batch_p = ffi.new('llama_batch[]', [(
            n_eval, # n_tokens
            ffi.NULL, # token
            embd, # embd
            batch_mrope_pos, # pos
            ffi.NULL, # n_seq_id
            ffi.NULL, # seq_id
            ffi.NULL, # logits
        )])

        # global_weakkeydict[batch_p] = (embd, batch_mrope_pos)

        batch: llama_batch = batch_p[0]
        # global_weakkeydict[batch_p] = batch

        if _llama_decode(ctx_llama, batch):
            return False, n_past, st_pos_id

        n_past += n_eval
        processed += n_eval
        # ffi.release(batch_p)
        print(f'!!! {i=}')

    print('!!! [1]:', n_past, st_pos_id)
    return True, n_past, st_pos_id


def clip_completions(model: llama_model_p, model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    assert isinstance(completions_options.prompt, str)
    assert completions_options.messages is None, 'messages are not currently supported'
    assert isinstance(completions_options.image, str)

    if completions_options.verbose:
        # default llama.cpp logger
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(model, model_options)
    clip_context = clip_init_context(model_options)
    sampler = sampler_init(model, completions_options)
    print(f'{sampler=}')

    if completions_options.grammar or completions_options.json_schema:
        grammar_sampler = grammar_sampler_init(model, completions_options)
    else:
        grammar_sampler = None

    print(f'{grammar_sampler=}')

    has_minicpmv_projector: int = lib.clip_is_minicpmv(clip_context)
    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)

    begin_vision_token: str = '<image>'
    end_vision_token: str = '</image>'

    if is_qwen2vl:
        begin_vision_token = '<|vision_start|>'
        end_vision_token = '<|vision_end|>'

    print(f'{has_minicpmv_projector=}')
    print(f'{is_qwen2vl=}')

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)

    # image embeddings
    embeds: llava_image_embed_p = _llava_image_embed_make_with_filename(
        clip_context,
        model_options.threads,
        completions_options.image.encode(),
    )

    assert embeds != ffi.NULL
    # print(f'{embeds=}')

    # create a llama_batch, we use this object to submit token data for decoding
    n_batch: int = lib.llama_n_batch(context)
    batch: llama_batch = lib.llama_batch_init(n_batch, 0, 1)
    seq_ids: list[llama_seq_id] = [0]
    n_past: int = 0
    cur_pos_id: int = n_past # used by qwen2vl
    idx: int = 0
    prompt: str
    prompt_tokens: list[int]

    # pre-embeds prompt
    messages = [{'role': 'user', 'content': begin_vision_token}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
    prompt = prompt[:prompt.index(begin_vision_token) + len(begin_vision_token)]
    print(1, f'{prompt=}')
    prompt_tokens = tokenizer.encode(prompt) # type: ignore

    if is_qwen2vl:
        # n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        prev_n_past = n_past
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        cur_pos_id += n_past - prev_n_past
    else:
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

    # embeds
    if is_qwen2vl:
        s, n_past, cur_pos_id = _qwen2vl_eval_image_embed(context, clip_context, embeds, n_batch, n_past, cur_pos_id)
        assert s
    else:
        n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

    idx += 1

    # evaluate the post-embeds prompt
    if is_qwen2vl:
        # n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        # prev_n_past = n_past
        # n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        # cur_pos_id += n_past - prev_n_past
        pass
    else:
        prompt = end_vision_token
        print(2, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore
        print(2, f'{prompt_tokens=}')

        prev_n_past = n_past
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        cur_pos_id += n_past - prev_n_past

    print('!!! [2.0]:', n_past, cur_pos_id)

    # print(f'{has_minicpmv_projector=}')

    if has_minicpmv_projector >= 2:
        num_image_embeds = embeds.n_image_pos / lib.clip_n_patches(clip_context)

        if num_image_embeds > 1:
            num_image_embeds_col = _clip_uhd_num_image_embeds_col(clip_context)

            prompt = '<slice>'
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
            i = 0

            while i < (num_image_embeds - 1) / num_image_embeds_col:
                j = 0

                while j < num_image_embeds_col:
                    prompt = '<image>'
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

                    idx += 1

                    prompt = '</image>'
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    j += 1

                prompt = '\n'
                prompt_tokens = tokenizer.encode(prompt) # type: ignore
                n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                i += 1

            prompt = '</slice>'
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

        # _llava_image_embed_free(embeds)

        # user prompt after image ebeds
        messages = [{'role': 'user', 'content': f'\n{completions_options.prompt}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = prompt[prompt.index(f'\n{completions_options.prompt}'):]
        print(3, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
    elif is_qwen2vl:
        # user prompt after image ebeds
        messages = [{'role': 'user', 'content': f'{completions_options.prompt}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = end_vision_token + prompt[prompt.index(f'{completions_options.prompt}'):]
        print(3, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore

        n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        print('!!! [2.1]:', n_past, cur_pos_id)

    # generate output
    n_cur: int = n_past
    n_ctx: int = lib.llama_n_ctx(context)
    n_decode: int = 0
    # print(f'{n_cur=} {n_ctx=} {n_decode=} {options.predict=}')

    while n_cur < n_ctx and n_decode < completions_options.predict:
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
            # piece: str = tokenizer.decode(new_token_id)
            piece = _common_token_to_piece(context, new_token_id, True)
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

    _llava_image_embed_free(embeds)
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
