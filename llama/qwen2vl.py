__all__ = ['qwen2vl_completions']

from typing import Iterator


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


def qwen2vl_completions(model: 'llama_model_p', model_options: 'ModelOptions', completions_options: 'CompletionsOptions') -> Iterator[str]:
    yield '[END]'
