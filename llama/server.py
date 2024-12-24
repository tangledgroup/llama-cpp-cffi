__all__ = ['routes', 'build_app']

import json
from pprint import pprint
from typing import Optional

from attrs import asdict
from aiohttp import web

from .model import Model
from .options import Options


DEFAULT_MODEL_DEF = ':'.join([
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    '',
    '',
])


routes = web.RouteTableDef()
current_model: Optional[Model] = None
current_model_def: Optional[str] = None
current_model_options: Optional[Options] = None


@routes.post('/api/1.0/completions')
async def api_1_0_completions(request: web.Request) -> web.Response:
    global current_model
    global current_model_def
    global current_model_options

    data = await request.json()
    print('api_1_0_completions data:')
    pprint(data)

    model_def: str = data.pop('model', DEFAULT_MODEL_DEF)
    stream: bool = data.pop('stream', False)
    print(data)

    options = Options(**data)
    model_options: Options = options.get_model_options()
    pprint(model_options)

    if current_model is None:
        model = Model(*model_def.split(':'))
        model.init(**asdict(model_options))

        current_model = model
        current_model_def = model_def
        current_model_options = model_options
    else:
        if current_model_def == model_def and current_model_options == model_options:
            model = current_model
        else:
            current_model_def.free()

            model = Model(*model_def.split(':'))
            model.init(**asdict(model_options))

            current_model = model
            current_model_def = model_def
            current_model_options = model_options

    if stream:
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        await response.prepare(request)
        chunk_bytes: bytes

        for token in model.completions(**asdict(options)):
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
        full_response = ''.join([token for token in model.completions(**asdict(options))])

        return web.json_response({
            'choices': [{
                'message': {'content': full_response},
                'finish_reason': 'stop',
                'index': 0
            }]
        })


def build_app():
    app = web.Application()
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    app = build_app()
    web.run_app(app, host='0.0.0.0', port=11434)
