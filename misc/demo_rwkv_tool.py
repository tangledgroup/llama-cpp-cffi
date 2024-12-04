from llama import completions, Model, Options

from demo_models import models
from demo_messages import tools_messages as messages


def demo(model: Model):
    print(model)

    options = Options(
        ctx_size=0,
        predict=-1,
        no_context_shift=True,
        model=model,
        prompt=messages,
        temp=0.0,
        # json_schema='{}',
        no_display_prompt=True,
        gpu_layers=99,
    )

    for chunk in completions(options):
        print(chunk, flush=True, end='')

    # newline
    print('\n' + '-' * 20)


if __name__ == '__main__':
    models_ids: list[str] = [
        'RWKV/v6-Finch-1B6-HF',
        'RWKV/v6-Finch-3B-HF',
    ]

    for model_id in models_ids:
        model: Model = models[model_id]
        demo(model)
