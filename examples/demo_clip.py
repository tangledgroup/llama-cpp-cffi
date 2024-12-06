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
    clip_completions,
)

from demo_models import demo_models


# model_id: str = 'liuhaotian/llava-v1.6-mistral-7b'
# model_id: str = 'openbmb/MiniCPM-Llama3-V-2_5'
model_id: str = 'openbmb/MiniCPM-V-2_6'
# model_id: str = 'vikhyatk/moondream2'
# model_id: str = 'BAAI/Bunny-v1_0-4B'

model: Model = demo_models[model_id]
config = get_config(model.creator_hf_repo)

options = Options(
    model=model,
    predict=-2,
    temp=0.7,
    top_p=0.8,
    top_k=100,
    # prompt='What is in the image?',
    prompt='What is in the image? Extract (OCR) all text from page as markdown.',
    # prompt='Extract (OCR) all text from page as markdown.',
    # image='examples/llama-1.png',
    image='examples/llama-3.png',
    # image='examples/llama-4.png',
    gpu_layers=99,
)

backend_init()

_model = model_init(options)
print(f'{_model=}')

_context = context_init(_model, options)
print(f'{_context=}')

_sampler = sampler_init(options)
print(f'{_sampler=}')

_clip_context = clip_init_context(options)
print(f'{_clip_context=}')

input('Press any key to generate')

for token in clip_completions(_model, _context, _sampler, _clip_context, options):
    print(token, end='', flush=True)

print()

input('Press any key to exit')

clip_free_context(_clip_context)
sampler_free(_sampler)
context_free(_context)
model_free(_model)
backend_free()
