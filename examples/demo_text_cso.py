import gc
from threading import Thread

from llama import Model

from demo_models import demo_models


def demo_high_level_json():
    model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    model: list[Model] = demo_models[model_id]
    model.init(n_ctx=4 * 1024, gpu_layers=99)
    # input('Press any key to generate')

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

    prompt = f'Think step by step.\nExplain meaning of life in JSON format.\nUse JSON Schema:\n```\n{json_schema}\n```'

    for token in model.completions(prompt=prompt, predict=4 * 1024, json_schema=json_schema, grammar_ignore_until='</think>'):
        print(token, end='', flush=True)

    print()
    # input('Press any key to exit')


if __name__ == '__main__':
    demo_high_level_json()
    gc.collect()
