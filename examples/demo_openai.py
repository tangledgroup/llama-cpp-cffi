from openai import OpenAI
from llama import Model, file_to_data_uri

from demo_models import demo_models


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='llama-cpp-cffi',
)


def demo_text_chat_completions():
    print('demo_text_chat_completions:')
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    model: Model = demo_models[model_id]

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

    response = client.chat.completions.create(
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.0,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
        ),
    )

    print(response.choices[0].message.content)


def demo_text_chat_completions_stream():
    print('demo_text_chat_completions_stream:')
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    model: Model = demo_models[model_id]

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

    response = client.chat.completions.create( # type: ignore
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.0,
        stream=True,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
        ),
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


def demo_text_chat_completions_json_stream():
    print('demo_text_chat_completions_json_stream:')
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    model: Model = demo_models[model_id]

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

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

    response = client.chat.completions.create( # type: ignore
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.0,
        stream=True,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
            json_schema=json_schema,
        ),
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


def demo_image_chat_completions():
    print('demo_image_chat_completions:')
    model_id = 'Qwen/Qwen2-VL-2B-Instruct'
    model: Model = demo_models[model_id]

    prompt = 'Describe this image.'
    image_path = 'examples/llama-1.jpg'

    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': file_to_data_uri(image_path)}
            }
        ]}
    ]

    response = client.chat.completions.create( # type: ignore
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.0,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
        ),
    )

    print(response.choices[0].message.content)


def demo_image_chat_completions_stream():
    print('demo_image_chat_completions_stream:')
    model_id = 'Qwen/Qwen2-VL-2B-Instruct'
    model: Model = demo_models[model_id]

    prompt = 'Describe this image.'
    image_path = 'examples/llama-1.jpg'

    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': file_to_data_uri(image_path)}
            }
        ]}
    ]

    response = client.chat.completions.create( # type: ignore
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.0,
        stream=True,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
        ),
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


def demo_image_chat_completions_json_stream():
    print('demo_image_chat_completions_json_stream:')
    model_id = 'Qwen/Qwen2-VL-7B-Instruct'
    model: Model = demo_models[model_id]

    prompt = 'Describe this image.'
    image_path = 'examples/llama-1.jpg'

    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': file_to_data_uri(image_path)}
            }
        ]}
    ]

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

    response = client.chat.completions.create( # type: ignore
        model=str(model),
        messages=messages, # type: ignore
        temperature=0.7,
        stream=True,

        # llama-cpp-cffi
        extra_body=dict( # type: ignore
            n_ctx=4 * 1024,
            gpu_layers=99,
            predict=512,
            top_k=100,
            json_schema=json_schema,
        ),
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


if __name__ == '__main__':
    demo_text_chat_completions()
    demo_text_chat_completions_stream()
    demo_text_chat_completions_json_stream()
    demo_image_chat_completions()
    demo_image_chat_completions_stream()
    demo_image_chat_completions_json_stream()
