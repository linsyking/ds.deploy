import asyncio
import copy
import json
import unittest

from web_search import WebSearchMiddleware


def request(tool_type='web_search_20250305'):
    return {'model': 'test', 'messages': [{'role': 'user', 'content':
            'Perform a web search for the query: SGLang Exa integration'}],
            'tools': [{'type': tool_type, 'name': 'web_search'}],
            'tool_choice': {'type': 'tool', 'name': 'web_search'}}


class Searcher:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.failure = False

    async def __call__(self, query, include, exclude):
        self.calls.append((query, include, exclude))
        await asyncio.sleep(0)
        if self.failure:
            raise RuntimeError('Provider error text must not be returned')
        return {'results': [{'title': 'SGLang', 'url': 'https://docs.sglang.ai/',
                             'highlights': ['Native search integration.']}]}

    async def close(self):
        self.closed = True


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.searcher = Searcher()
        self.forwarded = []
        async def app(scope, receive, send):
            if scope['type'] == 'lifespan':
                return
            first = await receive()
            self.forwarded.append((scope, first['body']))
            # A streaming disconnect listener must get the original receive
            # after the one replayed body, rather than replaying forever.
            self.assertEqual((await receive())['type'], 'http.disconnect')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'data: first\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: done\n\n'})
        self.middleware = WebSearchMiddleware(app, searcher=self.searcher, api_key='test-key')

    async def invoke(self, body, auth=b'Bearer test-key', path='/v1/messages', middleware=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        pieces = [{'type': 'http.request', 'body': raw[:12], 'more_body': True},
                  {'type': 'http.request', 'body': raw[12:], 'more_body': False},
                  {'type': 'http.disconnect'}]
        async def receive():
            return pieces.pop(0)
        # The passthrough paths let the native app read the original stream.
        async def collect(scope, receive, send):
            if scope['type'] != 'http':
                return
            chunks = []
            while True:
                message = await receive()
                chunks.append(message.get('body', b''))
                if not message.get('more_body', False):
                    break
            self.forwarded.append((scope, b''.join(chunks)))
            self.assertEqual((await receive())['type'], 'http.disconnect')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'data: first\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: done\n\n'})
        selected = middleware or self.middleware
        selected.app = collect
        sent = []
        async def send(message):
            sent.append(message)
        scope = {'type': 'http', 'method': 'POST', 'path': path,
                 'headers': [(b'authorization', auth), (b'content-length', str(len(raw)).encode())]}
        await selected(scope, receive, send)
        return sent

    async def test_search_rewrites_and_preserves_streaming(self):
        sent = await self.invoke(request())
        self.assertEqual(self.searcher.calls, [('SGLang Exa integration', (), ())])
        scope, raw = self.forwarded[-1]
        result = json.loads(raw)
        self.assertNotIn('tools', result)
        self.assertNotIn('tool_choice', result)
        self.assertIn('https://docs.sglang.ai/', result['messages'][-1]['content'][-1]['text'])
        self.assertEqual(int(dict(scope['headers'])[b'content-length']), len(raw))
        self.assertEqual([m['body'] for m in sent[1:]], [b'data: first\n\n', b'data: done\n\n'])

    async def test_preserves_image_and_custom_tool(self):
        body = request()
        image = {'type': 'image', 'source': {'type': 'base64', 'data': 'abc', 'media_type': 'image/png'}}
        body['messages'][-1]['content'] = [image, {'type': 'text', 'text': 'SGLang search'}]
        custom = {'name': 'Read', 'input_schema': {'type': 'object'}}
        body['tools'].append(custom)
        await self.invoke(body)
        result = json.loads(self.forwarded[-1][1])
        self.assertEqual(result['messages'][-1]['content'][0], image)
        self.assertEqual(result['tools'], [custom])
        self.assertEqual(result['tool_choice'], {'type': 'auto'})

    async def test_non_search_is_byte_identical(self):
        raw = b'{ "messages": [{"role":"user","content":"hello"}], "tools": [] }'
        await self.invoke(raw)
        self.assertEqual(self.forwarded[-1][1], raw)
        self.assertFalse(self.searcher.calls)

    async def test_custom_websearch_is_not_a_server_tool(self):
        body = request()
        body['tools'] = [{'name': 'WebSearch', 'input_schema': {'type': 'object'}}]
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_unauthorized_never_searches(self):
        for auth in (b'', b'Bearer wrong', b'\xff'):
            await self.invoke(request(), auth=auth)
        self.assertFalse(self.searcher.calls)

    async def test_other_endpoint_never_searches(self):
        await self.invoke(request(), path='/v1/chat/completions')
        self.assertFalse(self.searcher.calls)

    async def test_filters_and_cache(self):
        body = request()
        body['tools'][0]['allowed_domains'] = ['docs.sglang.ai']
        await self.invoke(body)
        await self.invoke(body)
        self.assertEqual(self.searcher.calls, [('SGLang Exa integration', ('docs.sglang.ai',), ())])
        body['tools'][0].pop('allowed_domains')
        body['tools'][0]['blocked_domains'] = ['example.com']
        await self.invoke(body)
        self.assertEqual(len(self.searcher.calls), 2)
        self.assertEqual(self.searcher.calls[-1][2], ('example.com',))

    async def test_conflicting_filters_do_not_spend_credits(self):
        body = request()
        body['tools'][0].update(allowed_domains=['a.com'], blocked_domains=['b.com'])
        sent = await self.invoke(body)
        self.assertEqual(sent[0]['status'], 400)
        self.assertFalse(self.searcher.calls)
        self.assertFalse(self.forwarded)

    async def test_provider_failure_is_explicit_and_not_cached(self):
        self.searcher.failure = True
        sent = await self.invoke(request())
        self.assertEqual(sent[0]['status'], 502)
        self.assertNotIn(b'Provider error text', sent[-1]['body'])
        self.assertFalse(self.forwarded)
        self.searcher.failure = False
        await self.invoke(request())
        self.assertEqual(len(self.searcher.calls), 2)

    async def test_concurrent_identical_search_is_coalesced(self):
        await asyncio.gather(*(self.invoke(copy.deepcopy(request())) for _ in range(4)))
        self.assertEqual(len(self.searcher.calls), 1)

    async def test_search_disabled_by_request(self):
        body = request()
        body['tools'][0]['max_uses'] = 0
        await self.invoke(body)
        body = request()
        body['tool_choice'] = {'type': 'none'}
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_versioned_and_unversioned_tools(self):
        for tool_type in ('web_search', 'web_search_20260209', 'web_search_20260318'):
            body = request(tool_type)
            await self.invoke(body)
            self.assertNotIn('tools', json.loads(self.forwarded[-1][1]))

    async def test_malformed_input_passes_to_native_validation(self):
        for raw in (b'not JSON', b'[]'):
            await self.invoke(raw)
            self.assertEqual(self.forwarded[-1][1], raw)
        body = request()
        body['tool_choice'] = 'invalid'
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_lifespan_closes_exa_connection(self):
        async def ignore(*args):
            pass
        await self.middleware({'type': 'lifespan'}, ignore, ignore)
        self.assertTrue(self.searcher.closed)


if __name__ == '__main__':
    unittest.main()
