from llama import get_config, Model, Options

from llama import (
    backend_init,
    backend_free,
    model_init,
    model_free,
    context_init,
    context_free,
    sampler_init,
    sampler_free,
    clip_init_context,
    clip_free_context,
    minicpmv_generate,
)

from demo_models import models


model_id: str = 'openbmb/MiniCPM-V-2_6'
model: Model = models[model_id]
config = get_config(model.creator_hf_repo)

options = Options(
    model=model,
    ctx_size=config.max_position_embeddings,
    predict=-2,
    temp=0.7,
    top_p=0.8,
    top_k=100,
    prompt='What is in the image?',
    image='examples/llama-1.jpg',
    gpu_layers=99,
)

backend_init()

_model = model_init(options)
print(f'{_model=}')

_ctx = context_init(_model, options)
print(f'{_ctx=}')

_sampler = sampler_init(options)
print(f'{_sampler=}')

_clip_ctx = clip_init_context(options)
print(f'{_clip_ctx=}')

input('Press any key to generate')

for token in minicpmv_generate(_model, _ctx, _sampler, _clip_ctx, options):
    print(token, end='', flush=True)

print()

input('Press any key to exit')

clip_free_context(_clip_ctx)
sampler_free(_sampler)
context_free(_ctx)
model_free(_model)
backend_free()
