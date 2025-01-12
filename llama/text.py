__all__ = ['text_completions']

from typing import Iterator

from transformers import AutoTokenizer

from .options import ModelOptions, CompletionsOptions
from .formatter import get_tokenizer, format_messages
from .llama_cpp import lib, ffi, llama_model_p, llama_batch, llama_seq_id, llama_token, llama_vocab_p
from .context import context_init, context_free
from .sampler import sampler_init, grammar_sampler_init, sampler_free, _common_sampler_sample, _common_sampler_accept

from .util import (
    _decode_tokens,
    _common_token_to_piece,
    _common_batch_clear,
    _common_batch_add,
    _llama_decode,
)


#
# llm
#
def text_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    assert isinstance(completions_options.prompt, str) or isinstance(completions_options.messages, list)

    _model: llama_model_p = model._model
    vocab: llama_vocab_p = lib.llama_model_get_vocab(_model)

    if completions_options.verbose:
        # default llama.cpp logger
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(_model, model_options)
    sampler = sampler_init(_model, completions_options)
    # print(f'{sampler=}')

    if completions_options.grammar or completions_options.json_schema:
        grammar_sampler = grammar_sampler_init(_model, completions_options)
    else:
        grammar_sampler = ffi.NULL

    # print(f'{grammar_sampler=}')

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    elif model_options.creator_hf_repo:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)
    else:
        raise ValueError()

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

        if lib.llama_token_is_eog(vocab, new_token_id):
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

    lib.llama_batch_free(batch)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    context_free(context)
