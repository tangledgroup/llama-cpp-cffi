from threading import Thread
from llama import Model, Options, model_init, model_free, clip_completions

from demo_models import demo_models


def demo_low_level():
    # model_id: str = 'liuhaotian/llava-v1.6-mistral-7b'
    # model_id: str = 'openbmb/MiniCPM-Llama3-V-2_5'
    model_id: str = 'openbmb/MiniCPM-V-2_6'
    # model_id: str = 'vikhyatk/moondream2'
    # model_id: str = 'BAAI/Bunny-v1_0-4B'

    model: Model = demo_models[model_id]

    options = Options(
        model=model,
        ctx_size=8 * 1024,
        predict=1024,
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

    _model = model_init(options)
    # print(f'{_model=}')

    # input('Press any key to generate')

    for token in clip_completions(_model, options):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')

    model_free(_model)


def demo_high_level_gpt():
    models_ids = [
        'vikhyatk/moondream2',
        'openbmb/MiniCPM-V-2_6',
    ]

    models = [demo_models[models_id] for models_id in models_ids]

    for model in models:
        model.init(ctx_size=1024, predict=1024, gpu_layers=8)

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
                temp=0.7,
                top_p=0.8,
                top_k=100,
                repeat_penalty=1.05,
                prompt='What is in the image? Extract (OCR) all text from page as markdown.',
                # image='examples/llama-1.png',
                image='examples/llama-3.png',
                # image='examples/llama-4.png',
            ),
        )

        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # input('Press any key to exit')


def demo_high_level():
    # model_id = 'vikhyatk/moondream2'
    model_id = 'openbmb/MiniCPM-V-2_6'

    model = demo_models[model_id]
    model.init(ctx_size=8 * 1024, predict=1024, gpu_layers=99)

    # input('Press any key to generate')

    prompt = 'What is in the image? Output in JSON format.\n'
    # image = 'examples/llama-1.png'
    image = 'examples/Capital-Call-Notice-pdf-0.jpg'

    for token in model.completions(prompt=prompt, image=image):
        print(token, end='', flush=True)

    for token in model.completions(prompt=prompt, image=image):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')


def demo_high_level_json():
    model_id = 'vikhyatk/moondream2'
    # model_id = 'openbmb/MiniCPM-V-2_6'

    model = demo_models[model_id]
    model.init(ctx_size=8 * 1024, predict=1024, gpu_layers=99)

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

    for token in model.completions(prompt=prompt, image=image, json_schema=json_schema):
        print(token, end='', flush=True)

    print()

    # input('Press any key to exit')


if __name__ == '__main__':
    # demo_low_level()
    # demo_high_level_gpt()
    demo_high_level()
    # demo_high_level_json()
