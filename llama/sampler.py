__all__ = [
    'sampler_init',
    'grammar_sampler_init',
    'sampler_free',
    '_llama_sampler_sample',
    '_common_sampler_sample',
    '_common_sampler_accept',
]

import json

from .llama_cpp import (
    lib,
    ffi,
    global_weakkeydict,
    char_p,
    llama_model_p,
    llama_sampler_p,
    llama_sampler_chain_params,
    llama_context_p,
    llama_token,
    llama_token_data_p,
    llama_token_data_array_p,
    llama_vocab_p,
)

from .options import CompletionsOptions
from .util import _set_logits


def sampler_init(model: llama_model_p, completions_options: CompletionsOptions) -> llama_sampler_p:
    vocab: llama_vocab_p = lib.llama_model_get_vocab(model)
    sampler_params: llama_sampler_chain_params = lib.llama_sampler_chain_default_params()
    sampler: llama_sampler_p = lib.llama_sampler_chain_init(sampler_params)

    # common
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_logit_bias(
        lib.llama_vocab_n_tokens(vocab),
        0,
        ffi.NULL,
    ))

    # dry
    seq_breakers: char_p = ffi.new('char*[]', completions_options.dry_sequence_breaker)
    num_breakers: int = len(completions_options.dry_sequence_breaker)

    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dry(
        vocab,
        lib.llama_model_n_ctx_train(model),
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
    vocab: llama_vocab_p = lib.llama_model_get_vocab(model)

    grmr: llama_sampler_p = lib.llama_sampler_init_grammar(
        vocab,
        grammar_str,
        grammar_root,
    )

    return grmr


def sampler_free(sampler: llama_sampler_p):
    lib.llama_sampler_free(sampler)


def _llama_sampler_sample(smpl: llama_sampler_p, ctx: llama_context_p, idx: int):
    # reimplementation of C code
    cur, cur_p = _set_logits(ctx, idx)
    lib.llama_sampler_apply(smpl, cur_p)
    token: llama_token = cur_p.data[cur_p.selected].id
    lib.llama_sampler_accept(smpl, token)
    return token


def _common_sampler_sample(grmr: llama_sampler_p, chain: llama_sampler_p, ctx: llama_context_p, idx: int, grammar_first: bool=False) -> llama_token:
    cur, cur_p = _set_logits(ctx, idx)

    if grammar_first and grmr:
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

    if grmr:
        lib.llama_sampler_apply(grmr, single_token_data_array)

    # print(f'{single_token_data_array.data[0].logit=}')
    is_valid: bool = single_token_data_array.data[0].logit != float('-inf')

    if is_valid:
        # print(f'{id=} {is_valid=}')
        return id

    # resampling
    cur, cur_p = _set_logits(ctx, idx)

    if grmr:
        lib.llama_sampler_apply(grmr,  cur_p)

    lib.llama_sampler_apply(chain, cur_p)
    assert cur_p.selected != -1, "no selected token during re-sampling - check your sampling configuration"

    id: llama_token = cur_p.data[cur_p.selected].id
    return id


def _common_sampler_accept(grmr: llama_sampler_p, chain: llama_sampler_p, token: llama_token, accept_grammar: bool):
    if accept_grammar and grmr:
        lib.llama_sampler_accept(grmr, token)

    lib.llama_sampler_accept(chain, token)
