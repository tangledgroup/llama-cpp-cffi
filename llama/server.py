__all__ = ['routes', 'build_app']

import os
import json
import asyncio
from pprint import pprint
from typing import Any, Optional, AsyncIterator

from attrs import asdict
from aiohttp import web

from .model import Model
from .options import ModelOptions, CompletionsOptions
from .util import base64_image_to_tempfile


routes = web.RouteTableDef()

current_model = Model(
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    'qwen2.5-1.5b-instruct-q4_k_m.gguf',
)

current_model.init(
    n_ctx=8 * 1024,
    gpu_layers=99,
)

lock = asyncio.Lock()


async def process_completions(data: dict) -> AsyncIterator[str]:
    global current_model

    # pprint(data)
    # print('=' * 80)

    image: Optional[str] = data.pop('image', None)
    image_file: Optional[Any] = None

    if image:
        image_file = base64_image_to_tempfile(image)

    model_kwargs = {k: data[k] for k in ModelOptions.__annotations__.keys() if k in data}
    model_options = ModelOptions(**model_kwargs)
    # pprint(model_options)
    # print('=' * 80)

    completions_kwargs = {k: data[k] for k in CompletionsOptions.__annotations__ if k in data}
    completions_options = CompletionsOptions(**completions_kwargs)

    if image and image_file:
        completions_options.image = image_file.name

    # pprint(completions_options)
    # print('=' * 80)

    model = Model(options=model_options)
    # pprint(model)
    # print('=' * 80)

    # print(repr(current_model))
    # print('=' * 80)

    if current_model == model or (model.options and model.options.creator_hf_repo is None):
        # print('1')
        model = current_model
    else:
        # print('2')
        current_model.free()
        model.init(**asdict(model_options))
        current_model = model

    for token in model.completions(**asdict(completions_options)):
        yield token

    # remove temp image file
    if image and image_file:
        image_file.close()
        os.unlink(image_file.name)


#
# llama-cpp-cffi API
#
@routes.post('/api/1.0/completions')
async def api_1_0_completions(request: web.Request) -> web.Response | web.StreamResponse:
    global current_model

    data: dict = await request.json()
    # print('api_1_0_completions data:')
    stream: bool = data.pop('stream', False)

    async with lock:
        if stream:
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            await response.prepare(request)
            chunk_bytes: bytes

            async for token in process_completions(data):
                event_data = {
                    'choices': [{
                        'delta': {'content': token},
                        'finish_reason': None,
                        'index': 0
                    }]
                }

                chunk_bytes = f'data: {json.dumps(event_data)}\n\n'.encode('utf-8')
                await response.write(chunk_bytes)

            # Send the final message
            chunk_bytes = b'data: [DONE]\n\n'
            await response.write(chunk_bytes)
            return response
        else:
            full_response: list[str] | str = []

            async for token in process_completions(data):
                full_response.append(token)

            full_response = ''.join(full_response)

            return web.json_response({
                'choices': [{
                    'message': {'content': full_response},
                    'finish_reason': 'stop',
                    'index': 0
                }]
            })


#
# openai API
#
@routes.post('/v1/chat/completions')
async def v1_chat_completions(request: web.Request) -> web.Response | web.StreamResponse:
    global current_model

    data: dict = await request.json()
    # print('api_1_0_completions data:')
    prompt: Optional[str] = data.get('prompt')
    image: Optional[str] = data.get('image')
    messages: Optional[list[dict]] = data.get('messages')
    model: str = data.pop('model')
    frequency_penalty: float = data.get('frequency_penalty', 0.0)
    # logit_bias: bool = data.get('logit_bias')
    # logprobs: bool = data.get('logprobs', False)
    # top_logprobs: bool = data.get('top_logprobs')
    max_tokens: int = data.get('max_tokens', 512) # https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_tokens
    n: int = data.get('n', 1)
    presence_penalty: float = data.get('presence_penalty', 0.0)
    # response_format = data.get('response_format')
    seed = data.get('seed', 23)
    # service_tier = data.get('service_tier')
    # stop = data.get('stop')
    stream = data.get('stream', False)
    # stream_options = data.get('stream_options')
    temperature: float = data.get('temperature', 0.0) # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
    top_p: int = data.get('top_p', 0.9) # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    # tools = data.get('tools') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    # tool_choice = data.get('tool_choice') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # parallel_tool_calls = data.get('parallel_tool_calls', True) # NOTE: https://platform.openai.com/docs/api-reference/chat/create?lang=curl#chat-create-parallel_tool_calls
    # user = data.get('user')

    # repack data for llama-cpp-cffi
    llama_cpp_cffi_data: dict = {}
    llama_cpp_cffi_data['n_ctx'] = data.get('n_ctx', 8 * 1024) # extra_body
    llama_cpp_cffi_data['gpu_layers'] = data.get('gpu_layers', 99)  # extra_body

    llama_cpp_cffi_data['prompt'] = prompt
    llama_cpp_cffi_data['image'] = image
    llama_cpp_cffi_data['messages'] = messages

    model_items: list[str] = model.split(':')
    creator_hf_repo, hf_repo, hf_file, mmproj_hf_file, tokenizer_hf_repo = model_items + [None] * (5 - len(model_items))
    llama_cpp_cffi_data['creator_hf_repo'] = creator_hf_repo
    llama_cpp_cffi_data['hf_repo'] = hf_repo
    llama_cpp_cffi_data['hf_file'] = hf_file
    llama_cpp_cffi_data['mmproj_hf_file'] = mmproj_hf_file
    llama_cpp_cffi_data['tokenizer_hf_repo'] = tokenizer_hf_repo

    llama_cpp_cffi_data['frequency_penalty'] = frequency_penalty
    llama_cpp_cffi_data['predict'] = max_tokens
    assert n == 1
    llama_cpp_cffi_data['presence_penalty'] = presence_penalty
    llama_cpp_cffi_data['seed'] = seed
    llama_cpp_cffi_data['temp'] = temperature
    llama_cpp_cffi_data['top_p'] = top_p

    llama_cpp_cffi_data['grammar'] = data.get('grammar', None)
    llama_cpp_cffi_data['json_schema'] = data.get('json_schema', None)
    llama_cpp_cffi_data['chat_template'] = data.get('chat_template', None)

    async with lock:
        if stream:
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            await response.prepare(request)
            chunk_bytes: bytes

            async for token in process_completions(llama_cpp_cffi_data):
                event_data = {
                    'choices': [{
                        'delta': {'content': token},
                        'finish_reason': None,
                        'index': 0
                    }]
                }

                chunk_bytes = f'data: {json.dumps(event_data)}\n\n'.encode('utf-8')
                await response.write(chunk_bytes)

            # Send the final message
            chunk_bytes = b'data: [DONE]\n\n'
            await response.write(chunk_bytes)
            return response
        else:
            full_response: list[str] | str = []

            async for token in process_completions(llama_cpp_cffi_data):
                full_response.append(token)

            full_response = ''.join(full_response)

            return web.json_response({
                'choices': [{
                    'message': {'content': full_response},
                    'finish_reason': 'stop',
                    'index': 0
                }]
            })


def build_app():
    app = web.Application(client_max_size=1024 * 1024 * 1024)
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    app = build_app()
    web.run_app(app, host='0.0.0.0', port=11434)
