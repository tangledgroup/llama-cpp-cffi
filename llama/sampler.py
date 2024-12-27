__all__ = [
    'sampler_init',
    'grammar_sampler_init',
    'sampler_free',
]

import json

from .llama_cpp import lib, ffi, llama_model_p, llama_sampler_p, llama_sampler_chain_params, char_p
from .options import CompletionsOptions


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
