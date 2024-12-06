from threading import Thread, Lock
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

from demo_models import demo_models


def demo_low_level():
    model_id: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    model: Model = demo_models[model_id]
    config = get_config(model.creator_hf_repo)

    options = Options(
        model=model,
        ctx_size=config.max_position_embeddings,
        predict=512,
        gpu_layers=99,
        prompt='Meaning of life is',
    )

    backend_init()

    _model = model_init(options)
    print(f'{_model=}')

    _context = context_init(_model, options)
    print(f'{_context=}')

    _sampler = sampler_init(options)
    print(f'{_sampler=}')

    # input('Press any key to generate')

    for token in text_completions(_model, _context, _sampler, options):
        print(token, end='', flush=True)

    print()
    # input('Press any key to exit')

    sampler_free(_sampler)
    context_free(_context)
    model_free(_model)
    backend_free()


def demo_high_level_gpt():
    models_ids = [
        'Qwen/Qwen2.5-0.5B-Instruct',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'HuggingFaceTB/SmolLM2-360M-Instruct',
        'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'arcee-ai/arcee-lite',
    ]

    models = [demo_models[models_id] for models_id in models_ids]

    for model in models:
        model.init(predict=512, gpu_layers=99)

    # input('Press any key to generate')

    def gen(idx, model, **options):
        text: list[str] | str = []

        for token in model.completions(**options):
            print(f'[{idx}]: {token}', end='', flush=True)
            text.append(token)

        text = ''.join(text)
        print(f'[{idx}]: [END]')
        print(f'[Final {idx} {model}]:', text)


    threads = []

    for i, model in enumerate(models):
        t = Thread(target=gen, args=[i, model], kwargs=dict(prompt='Meaning of life is'))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # input('Press any key to exit')


def demo_high_level_rwkv():
    models_ids = [
        'RWKV/v6-Finch-1B6-HF',
        'RWKV/v6-Finch-3B-HF',
    ]

    models = [demo_models[models_id] for models_id in models_ids]

    for model in models:
        model.init(predict=512, gpu_layers=99)

    # input('Press any key to generate')

    def gen(idx, model, **options):
        text: list[str] | str = []

        for token in model.completions(**options):
            print(f'[{idx}]: {token}', end='', flush=True)
            text.append(token)

        text = ''.join(text)
        print(f'[{idx}]: [END]')
        print(f'[Final {idx} {model}]:', text)

    threads = []

    for i, model in enumerate(models):
        t = Thread(target=gen, args=[i, model], kwargs=dict(prompt='Meaning of life is'))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # input('Press any key to exit')


if __name__ == '__main__':
    demo_low_level()
    demo_high_level_gpt()
    demo_high_level_rwkv()
