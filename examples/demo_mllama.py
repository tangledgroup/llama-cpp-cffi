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
    mllama_init_context,
    mllama_free_context,
    mllama_completions,
)

from demo_models import demo_models


model_id: str = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
model: Model = demo_models[model_id]
config = get_config(model.creator_hf_repo)

options = Options(
    model=model,
    predict=-2,
    # temp=0.7,
    # top_p=0.8,
    # top_k=100,
    # prompt='What is in the image?',
    prompt='What is in the image? Extract (OCR) all text from page as markdown.',
    # prompt='Extract (OCR) all text from page as markdown.',
    # image='examples/llama-1.png',
    # image='examples/llama-3.png',
    image='examples/llama-4.png',
    gpu_layers=99,
)

backend_init()

_model = model_init(options)
print(f'{_model=}')

_context = context_init(_model, options)
print(f'{_context=}')

_sampler = sampler_init(options)
print(f'{_sampler=}')

_mllama_context = mllama_init_context(options)
print(f'{_mllama_context=}')

input('Press any key to generate')

for token in mllama_completions(_model, _context, _sampler, _mllama_context, options):
    print(token, end='', flush=True)

print()

input('Press any key to exit')

mllama_free_context(_mllama_context)
sampler_free(_sampler)
context_free(_context)
model_free(_model)
backend_free()
