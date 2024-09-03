__all__ = ['openai_routes', 'v1_chat_completions', 'build_app']

import json
import asyncio
from pprint import pprint
from typing import AsyncIterator

from aiohttp import web

from .formatter import get_config, AutoConfig
from .llama_cli import llama_generate
from .model import Model
from .options import Options
from .util import is_cuda_available


openai_routes = web.RouteTableDef()


async def generate_response(options: Options) -> AsyncIterator[str]:
    for chunk in llama_generate(options):
        yield chunk
        await asyncio.sleep(0.0)


async def v1_chat_completions(request):
    data = await request.json()
    print('v1_chat_completions data:')
    pprint(data)
    prompt = data.get('prompt')
    messages = data.get('messages')
    model = data['model']
    frequency_penalty = data.get('frequency_penalty')
    logit_bias = data.get('logit_bias')
    logprobs = data.get('logprobs', False)
    top_logprobs = data.get('top_logprobs')
    max_tokens = data.get('max_tokens') # https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_tokens
    n = data.get('n', 1)
    presence_penalty = data.get('presence_penalty')
    response_format = data.get('response_format')
    seed = data.get('seed')
    service_tier = data.get('service_tier')
    stop = data.get('stop')
    stream = data.get('stream', False)
    stream_options = data.get('stream_options')
    temperature = data.get('temperature', 0.0) # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
    top_p = data.get('top_p') # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    tools = data.get('tools') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    tool_choice = data.get('tool_choice') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    parallel_tool_calls = data.get('parallel_tool_calls', True) # NOTE: https://platform.openai.com/docs/api-reference/chat/create?lang=curl#chat-create-parallel_tool_calls
    user = data.get('user')

    # llama-cpp-cffi
    ctx_size: int = data.get('ctx_size', 0)
    predict: int = data.get('predict', max_tokens)
    batch_size: int = data.get('batch_size')
    flash_attn: Optional[bool] = data.get('flash_attn')
    cont_batching: Optional[bool] = data.get('cont_batching')
    gpu_layers: Optional[int] = data.get('gpu_layers')
    gpu_layers_draft: Optional[int] = data.get('gpu_layers_draft')
    split_mode: Optional[str] = data.get('split_mode')
    tensor_split: Optional[str] = data.get('tensor_split')
    main_gpu: Optional[int] = data.get('main_gpu')

    assert frequency_penalty is None
    assert logit_bias is None
    assert logprobs == False
    assert top_logprobs is None
    assert max_tokens is None or isinstance(max_tokens, int)
    assert n == 1
    assert presence_penalty is None
    assert response_format is None or isinstance(response_format, dict)
    assert seed is None or isinstance(seed, int)
    assert service_tier is None
    assert stream_options is None
    assert top_p is None or isinstance(top_p, (int, float))

    model = Model(*model.split(':'))
    # config: AutoConfig = get_config(model.creator_hf_repo)
    # ctx_size: int = config.max_position_embeddings if max_tokens is None else max_tokens

    options = Options(
        seed=seed,
        ctx_size=ctx_size,
        batch_size=batch_size,
        predict=predict,
        prompt=prompt or messages,
        top_p=top_p,
        model=model,
        stop=stop,
    )

    if response_format:
        options.response_format = json.dumps(response_format)

    if is_cuda_available():
        options.flash_attn = flash_attn
        options.cont_batching = cont_batching
        options.gpu_layers = gpu_layers
        options.gpu_layers_draft = gpu_layers_draft
        options.split_mode = split_mode
        options.tensor_split = tensor_split
        options.main_gpu = main_gpu

    if stream:
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        await response.prepare(request)
        chunk_bytes: bytes

        async for chunk in generate_response(options):
            event_data = {
                'choices': [{
                    'delta': {'content': chunk},
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
        full_response = ''.join([chunk async for chunk in generate_response(options)])
        
        return web.json_response({
            'choices': [{
                'message': {'content': full_response},
                'finish_reason': 'stop',
                'index': 0
            }]
        })


@openai_routes.post('/v1/chat/completions')
async def _v1_chat_completions(request):
    res: web.Response = await v1_chat_completions(request)
    return res


def build_app():
    app = web.Application()
    app.add_routes(openai_routes)
    return app


if __name__ == '__main__':
    app = build_app()
    web.run_app(app, host='0.0.0.0', port=11434)
