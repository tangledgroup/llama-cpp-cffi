__all__ = [
    '_clip_process_eval_image_embed',
    '_clip_uhd_num_image_embeds_col',
    'clip_init_context',
    'clip_free_context',
]

from huggingface_hub import hf_hub_download

from .llama_cpp import lib, ffi, lock, llama_context_p, clip_ctx_p, llava_image_embed_p, void_p, float_p
from .options import ModelOptions


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


def clip_init_context(model_options: ModelOptions) -> clip_ctx_p:
    assert model_options.n_ctx >= 2048
    assert model_options.hf_repo
    assert model_options.mmproj_hf_file
    mmproj_path: str | bytes = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.mmproj_hf_file)
    # print(f'{mmproj_path=}')

    with lock:
        # clip_model_load(path, verbosity)
        clip_context: clip_ctx_p = lib.clip_model_load(mmproj_path.encode(), model_options.verbose)

    return clip_context


def clip_free_context(clip_context: clip_ctx_p):
    with lock:
        lib.clip_free(clip_context)
