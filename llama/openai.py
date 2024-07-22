import json
import asyncio

from aiohttp import web

from llama import llama_generate, get_config, Model, Options


async def generate_response(options: Options):
    for chunk in llama_generate(options):
        yield chunk


async def chat_completions(request):
    data = await request.json()
    messages = data['messages']
    model = data['model']
    frequency_penalty = data.get('frequency_penalty')
    logit_bias = data.get('logit_bias')
    logprobs = data.get('logprobs', False)
    top_logprobs = data.get('top_logprobs')
    max_tokens = data.get('max_tokens')
    n = data.get('n', 1)
    presence_penalty = data.get('presence_penalty')
    response_format = data.get('response_format') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
    seed = data.get('seed')
    service_tier = data.get('service_tier')
    stop = data.get('stop') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop
    stream = data.get('stream', False)
    stream_options = data.get('stream_options')
    temperature = data.get('temperature', 0.0) # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
    top_p = data.get('top_p') # NOTE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    tools = data.get('tools') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    tool_choice = data.get('tool_choice') # TODO: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    parallel_tool_calls = data.get('parallel_tool_calls', True)
    user = data.get('user')

    assert frequency_penalty is None
    assert logit_bias is None
    assert logprobs == False
    assert top_logprobs is None
    assert max_tokens is None or isinstance(max_tokens, int)
    assert n == 1
    assert presence_penalty is None
    assert response_format is None
    assert seed is None or isinstance(seed, int)
    assert service_tier is None
    assert stream_options is None
    assert top_p is None or ininstance(top_p, (int, float))

    model = Model(*model.split(':'))
    config = get_config(model.creator_hf_repo)
    
    if max_tokens:
        ctx_size = max_tokens
    else:
        ctx_size = config.max_position_embeddings
    
    options = Options(
        ctx_size=ctx_size,
        predict=-2,
        model=model,
        prompt=messages,
    )
    
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


app = web.Application()
app.router.add_post('/v1/chat/completions', chat_completions)


if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=11434)
