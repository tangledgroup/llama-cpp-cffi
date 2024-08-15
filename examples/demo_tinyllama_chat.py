from llama import llama_generate, get_config, Model, Options

from demo_models import models
from demo_messages import selfaware_consciousness_messages as messages


def demo(model: Model):
    print(model)
    config = get_config(model.creator_hf_repo)

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        no_display_prompt=True,
        gpu_layers=99,
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print('\n' + '-' * 20)


if __name__ == '__main__':
    models_ids: list[str] = [
        'TinyLlama/TinyLlama_v1.1',
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'cognitivecomputations/TinyDolphin-2.8-1.1b',
    ]

    for model_id in models_ids:
        model: Model = models[model_id]
        demo(model)
