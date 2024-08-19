from llama import llama_generate, get_config, Model, Options

from demo_models import models
from demo_messages import tools_messages as messages


def demo(model: Model):
    print(model)
    config = get_config(model.creator_hf_repo)
    
    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        temp=0,
        no_display_prompt=True,
        gpu_layers=99,
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print('\n' + '-' * 20)


if __name__ == '__main__':
    models_ids: list[str] = [
        'arcee-ai/arcee-lite',
    ]
    
    for model_id in models_ids:
        model: Model = models[model_id]
        demo(model)