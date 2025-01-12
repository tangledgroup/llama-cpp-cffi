import gc
from threading import Thread
from llama import Model

from demo_models import demo_models


def demo_high_level():
    model_id = 'openbmb/MiniCPM-V-2_6'
    model: Model = demo_models[model_id]
    model.init(n_ctx=4 * 1024, gpu_layers=99)

    # input('Press any key to generate')

    prompt = 'Describe this image.'
    image = 'examples/llama-1.png'
    # image = 'examples/llama-4.png'

    for token in model.completions(prompt=prompt, image=image, predict=512):
        print(token, end='', flush=True)

    print()
    # input('Press any key to exit')


def demo_high_level_gpt():
    models_ids = [
        'openbmb/MiniCPM-V-2_6',
        'openbmb/MiniCPM-V-2_6',
    ]

    models: list[Model] = [demo_models[models_id] for models_id in models_ids]

    for model in models:
        model.init(n_ctx=4 * 1024, gpu_layers=2)

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
        t = Thread(
            target=gen,
            args=[i, model],
            kwargs=dict(
                predict=512,
                prompt='Describe this image.',
                # prompt='What is in the image?',
                image='examples/llama-1.png',
                # image='examples/llama-3.png',
                # image='examples/llama-4.png',
            ),
        )

        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # input('Press any key to exit')


def demo_high_level_json():
    model_id = 'openbmb/MiniCPM-V-2_6'
    model: Model = demo_models[model_id]
    model.init(n_ctx=4 * 1024, gpu_layers=99)

    # input('Press any key to generate')

    prompt = 'What is in the image? Output in JSON format.\n'
    image = 'examples/llama-1.png'

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

    for token in model.completions(prompt=prompt, image=image, json_schema=json_schema, predict=1024):
        print(token, end='', flush=True)

    print()
    # input('Press any key to exit')


if __name__ == '__main__':
    demo_high_level()
    gc.collect()

    demo_high_level_gpt()
    gc.collect()

    demo_high_level_json()
    gc.collect()
