__all__ = [
    '_llava_image_embed_make_with_filename',
    '_llava_image_embed_free',
]

from .llama_cpp import lib, ffi, lock, clip_ctx_p, llava_image_embed_p


def _llava_image_embed_make_with_filename(ctx_clip: clip_ctx_p, n_threads: int, image_path: bytes) -> llava_image_embed_p:
    embed: llava_image_embed_p

    with lock:
        embed = lib.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)

    return embed


def _llava_image_embed_free(embed: llava_image_embed_p):
    with lock:
        lib.llava_image_embed_free(embed)
