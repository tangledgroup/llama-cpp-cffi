from llama import completions, get_config, Model, Options

from demo_models import models


def demo_prompt(model: Model):
    print(model)
    config = get_config(model.creator_hf_repo)

    options = Options(
        engine='minicpmv',
        n_ctx=config.max_position_embeddings,
        predict=-2,
        temp=0.7,
        top_p=0.8,
        top_k=100,
        model=model,
        prompt='What is in the image?',
        image='examples/llama-1.jpg',
        log_disable=False,
        gpu_layers=99,
    )

    for chunk in completions(options):
        print(chunk, flush=True, end='')

    # newline
    print('\n' + '-' * 20)


if __name__ == '__main__':
    models_ids: list[str] = [
        'openbmb/MiniCPM-V-2_6',
    ]

    for model_id in models_ids:
        model: Model = models[model_id]
        demo_prompt(model)
