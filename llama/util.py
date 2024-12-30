__all__ = [
    '_numa_init',
    '_llama_decode',
    '_decode_tokens',
    '_set_logits',
    '_common_batch_clear',
    '_common_batch_add',
    '_common_token_to_piece',
    '_zero_array',
]

from typing import Any

from .llama_cpp import (
    lib,
    ffi,
    lock,
    global_weakkeydict,
    ggml_numa_strategy,
    llama_context_p,
    llama_batch,
    char_p,
    float_p,
    llama_token_data_p,
    llama_token_data_array_p,
    llama_token,
    llama_pos,
    llama_seq_id,
    llama_model_p,
)


def _numa_init(numa: ggml_numa_strategy):
    with lock:
        lib.llama_numa_init(numa)


def _llama_decode(ctx: llama_context_p, batch: llama_batch) -> int:
    with lock:
        return lib.llama_decode(ctx, batch)


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


def _zero_array(arr: Any):
    for i in range(len(arr)):
        arr[i] = 0
