import gc
from threading import Thread
from llama import Model, ModelOptions, model_init, model_free, text_completions

from demo_models import demo_models
from llama.options import CompletionsOptions


def demo_high_level():
    # model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    # model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    # model_id = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    # model_id = 'arcee-ai/arcee-lite'
    # model_id = 'arcee-ai/Llama-3.1-SuperNova-Lite'
    # model_id = 'microsoft/Phi-3.5-mini-instruct'
    # model_id = 'numind/NuExtract-1.5'

    model = demo_models[model_id]
    model.init(ctx_size=4 * 1024, gpu_layers=99)

    # input('Press any key to generate')

    prompt = 'Explain the meaning of life. Meaning of life is'

    for token in model.completions(prompt=prompt, predict=1024):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')


def demo_high_level_chat():
    # model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    # model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    # model_id = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    # model_id = 'arcee-ai/arcee-lite'

    model = demo_models[model_id]
    model.init(ctx_size=4 * 1024, gpu_layers=99)

    # input('Press any key to generate')

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '1 + 1 = ?'},
        {'role': 'assistant', 'content': '2'},
        {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
    ]

    for token in model.completions(messages=messages, predict=512):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')


def demo_high_level_gpt():
    models_ids = [
        'Qwen/Qwen2.5-0.5B-Instruct',
        # 'Qwen/Qwen2.5-1.5B-Instruct',
        'HuggingFaceTB/SmolLM2-360M-Instruct',
        # 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'arcee-ai/arcee-lite',
    ]

    models = [demo_models[models_id] for models_id in models_ids]

    for model in models:
        model.init(ctx_size=4 * 1024, gpu_layers=99)

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
        t = Thread(target=gen, args=[i, model], kwargs=dict(prompt='Meaning of life is', predict=512))
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
        model.init(ctx_size=4 * 1024, gpu_layers=99)

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
        t = Thread(target=gen, args=[i, model], kwargs=dict(prompt='Meaning of life is', predict=512))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # input('Press any key to exit')


def demo_high_level_json():
    # model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    # model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    # model_id = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    # model_id = 'arcee-ai/arcee-lite'
    # model_id = 'numind/NuExtract-1.5'
    # model_id = 'RWKV/v6-Finch-1B6-HF'
    # model_id = 'RWKV/v6-Finch-3B-HF'

    model = demo_models[model_id]
    model.init(ctx_size=4 * 1024, gpu_layers=99)

    # input('Press any key to generate')

    prompt = 'Explain meaning of life in JSON format.\n'

    # json_schema = '{}'
    json_schema = '''{
      "type": "object",
      "properties": {
        "title": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "score": {
          "type": "number"
        }
      },
      "required": ["title", "description", "score"],
      "additionalProperties": false
    }'''

    for token in model.completions(prompt=prompt, predict=512, json_schema=json_schema):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')


if __name__ == '__main__':
    demo_high_level()
    gc.collect()

    demo_high_level_chat()
    gc.collect()

    demo_high_level_gpt()
    gc.collect()

    demo_high_level_rwkv()
    gc.collect()

    demo_high_level_json()
    gc.collect()
