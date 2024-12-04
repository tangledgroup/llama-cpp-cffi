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
    text_completions,
)

from demo_models import models


# model_id: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
model_id: str = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
model: Model = models[model_id]
config = get_config(model.creator_hf_repo)

options = Options(
    model=model,
    ctx_size=config.max_position_embeddings,
    predict=-2,
    prompt='Meaning of life is',
    gpu_layers=99,
)

backend_init()

_model = model_init(options)
print(f'{_model=}')

_ctx = context_init(_model, options)
print(f'{_ctx=}')

_sampler = sampler_init(options)
print(f'{_sampler=}')

input('Press any key to generate')

for token in text_completions(_model, _ctx, _sampler, options):
    print(token, end='', flush=True)

print()

input('Press any key to exit')

sampler_free(_sampler)
context_free(_ctx)
model_free(_model)
backend_free()
