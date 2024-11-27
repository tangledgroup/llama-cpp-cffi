from llama import llama_generate, Model, Options
from llama import SYSTEM_USER_ASSISTANT_TEMPLATE, create_alternate_messages

from demo_models import models
from demo_messages import few_shot_messages as messages


def demo_prompt(model: Model):
    print(model)

    options = Options(
        ctx_size=0,
        predict=-1,
        no_context_shift=True,
        model=model,
        prompt='''User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: Explain the meaning of life.

Assistant:''',
        reverse_prompt='User:',
        stop='User:',
        no_display_prompt=True,
        gpu_layers=99,
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print('\n' + '-' * 20)


def demo_messages(model: Model):
    print(model)

    options = Options(
        ctx_size=0,
        predict=-1,
        no_context_shift=True,
        model=model,
        prompt=create_alternate_messages(messages, convert_system_to_user=True),
        chat_template=SYSTEM_USER_ASSISTANT_TEMPLATE,
        # reverse_prompt='User:',
        stop='User:',
        no_display_prompt=True,
        gpu_layers=99,
    )

    for chunk in llama_generate(options):
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
        demo_prompt(model)
        demo_messages(model)
