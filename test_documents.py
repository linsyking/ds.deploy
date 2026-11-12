"""CPU-only PDF/ASGI regressions. Native checks: test_documents_native.py.

No network, model weights or GPU allocation needed. The fixture has a real text
page and an image-only page, so text-only extraction cannot pass these checks.
"""
import asyncio
import base64
import copy
from dataclasses import replace
import io
import json
import threading
import unittest
from unittest.mock import patch
import zlib

from PIL import Image, ImageDraw, ImageFont

from documents import DocumentError, DocumentMiddleware, DocumentProcessor, Limits, render_pdf


def fixture_pdf(text='TEXT_CODE_3816', visual='VISUAL_CODE_9274'):
    image = Image.new('RGB', (1000, 700), 'white')
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 44)
    draw.text((40, 30), visual, fill='black', font=font)
    draw.text((40, 120), 'Blue bar: 3    Red bar: 7', fill='black', font=font)
    draw.rectangle((150, 390, 350, 600), fill='blue')
    draw.rectangle((550, 180, 750, 600), fill='red')
    objects = []
    def obj(value):
        objects.append(value.encode() if isinstance(value, str) else value)
    def stream(header, data):
        return (f'<< {header} /Length {len(data)} >>\nstream\n'.encode()
                + data + b'\nendstream')
    obj('<< /Type /Catalog /Pages 2 0 R >>')
    obj('<< /Type /Pages /Count 2 /Kids [3 0 R 4 0 R] >>')
    obj('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
        '/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>')
    obj('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 720 504] '
        '/Resources << /XObject << /Im1 8 0 R >> >> /Contents 7 0 R >>')
    obj('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    obj(stream('', f'BT /F1 24 Tf 40 700 Td ({text}) Tj ET'.encode()))
    obj(stream('', b'q 720 0 0 504 0 0 cm /Im1 Do Q'))
    obj(stream('/Type /XObject /Subtype /Image /Width 1000 /Height 700 '
               '/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode',
               zlib.compress(image.tobytes())))
    result, offsets = bytearray(b'%PDF-1.4\n'), [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f'{number} 0 obj\n'.encode() + value + b'\nendobj\n')
    start = len(result)
    result.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        result.extend(f'{offset:010d} 00000 n \n'.encode())
    result.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF'.encode())
    return bytes(result)


def document(data=None, **kwargs):
    return dict(type='document', source=dict(type='base64', media_type='application/pdf',
                data=base64.b64encode(data if data is not None else fixture_pdf()).decode()), **kwargs)


def payload(block=None, nested=False):
    doc = document() if block is None else block
    result = dict(model='test', max_tokens=100, stream=True,
                  messages=[dict(role='user', content=[dict(type='text', text='Read this PDF'), doc])])
    if nested:
        result['messages'] = [dict(role='user', content='Read this PDF'),
            dict(role='assistant', content=[dict(type='tool_use', id='toolu_01', name='Read', input={})]),
            dict(role='user', content=[dict(type='text', text='before'),
                dict(type='tool_result', tool_use_id='toolu_01', is_error=False,
                     content=[dict(type='text', text='read result'), doc]),
                dict(type='text', text='after')])]
    return result


class RenderTests(unittest.TestCase):
    def test_text_and_scan_both_rendered(self):
        pdf = render_pdf(fixture_pdf(), Limits())
        self.assertEqual(len(pdf.pages), 2)
        self.assertIn('TEXT_CODE_3816', pdf.pages[0].text)
        self.assertEqual(pdf.pages[1].text, '')
        self.assertNotIn('VISUAL_CODE', ''.join(p.text for p in pdf.pages))
        for page in pdf.pages:
            with Image.open(io.BytesIO(base64.b64decode(page.image))) as image:
                self.assertLessEqual(max(image.size), 2048)
                self.assertLessEqual(image.width * image.height, 4_000_000)
        with Image.open(io.BytesIO(base64.b64decode(pdf.pages[1].image))) as scan:
            # Verify the rendered red bar actually survives, not a blank image.
            self.assertGreater(scan.getpixel((900, 500))[0], 200)
            self.assertLess(scan.getpixel((900, 500))[1], 50)

    def test_page_limit_rejects_instead_of_truncating(self):
        with self.assertRaisesRegex(DocumentError, 'pages'):
            render_pdf(fixture_pdf(), replace(Limits(), pages=1))

    def test_corrupt_pdf_has_clear_error(self):
        with self.assertRaisesRegex(DocumentError, 'valid, unencrypted'):
            render_pdf(b'not a PDF', Limits())

    def test_output_limit(self):
        with self.assertRaisesRegex(DocumentError, 'too large'):
            render_pdf(fixture_pdf(), replace(Limits(), expanded_bytes=100))


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processor = DocumentProcessor()

    async def asyncTearDown(self):
        await self.processor.close()

    async def test_top_level_keeps_text_images_title_context_and_order(self):
        original = payload(document(title='Report', context='Read both pages'))
        before = copy.deepcopy(original)
        result, changed = await self.processor.expand(original)
        self.assertTrue(changed)
        self.assertEqual(original, before)
        blocks = result['messages'][0]['content']
        self.assertEqual([b['type'] for b in blocks], ['text', 'text', 'image', 'text', 'image'])
        self.assertIn('Report', blocks[1]['text'])
        self.assertIn('Read both pages', blocks[1]['text'])
        self.assertIn('Page 1/2', blocks[1]['text'])
        self.assertIn('Page 2/2', blocks[3]['text'])
        self.assertEqual(result['stream'], original['stream'])

    async def test_nested_tool_result_envelope_and_order(self):
        original = payload(nested=True)
        result, changed = await self.processor.expand(original)
        self.assertTrue(changed)
        outer = result['messages'][-1]['content']
        self.assertEqual([b['type'] for b in outer], ['text', 'tool_result', 'text'])
        tool = outer[1]
        self.assertEqual(tool['tool_use_id'], 'toolu_01')
        self.assertIs(tool['is_error'], False)
        self.assertEqual([b['type'] for b in tool['content']], ['text', 'text', 'image', 'text', 'image'])

    async def test_plain_text_sources(self):
        for source in [dict(type='text', data='precise → 17'), dict(type='base64', media_type='text/plain',
                        data=base64.b64encode('precise → 17'.encode()).decode())]:
            result, changed = await self.processor.expand(payload(dict(type='document', source=source)))
            self.assertTrue(changed)
            self.assertIn('precise → 17', result['messages'][0]['content'][-1]['text'])

    async def test_invalid_sources(self):
        cases = [dict(type='base64', media_type='application/pdf', data='?!'),
                 dict(type='base64', media_type='application/pdf'),
                 dict(type='base64', media_type='application/msword', data='YQ=='),
                 dict(type='base64', media_type='text/plain', data='/w=='),
                 dict(type='file', file_id='file_01'), dict(type='text', data=12),
                 dict(type='url', url='file:///etc/passwd'), None]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(DocumentError):
                    await self.processor.expand(payload(dict(type='document', source=source)))

    async def test_citations_not_silently_ignored(self):
        with self.assertRaisesRegex(DocumentError, 'citations'):
            await self.processor.expand(payload(document(citations={'enabled': True})))

    async def test_total_page_limit_includes_duplicates_and_nested_documents(self):
        self.processor.limits = replace(Limits(), pages=3)
        body = payload(nested=True)
        body['messages'][0]['content'] = [document()]
        with self.assertRaisesRegex(DocumentError, 'per request'):
            await self.processor.expand(body)

    async def test_cache_coalesces_concurrent_requests(self):
        calls = []
        def render(data, limits):
            calls.append(1)
            return render_pdf(data, limits)
        self.processor.renderer = render
        results = await asyncio.gather(*(self.processor.expand(payload()) for _ in range(5)))
        await self.processor.expand(payload())
        self.assertEqual(len(calls), 1)
        self.assertEqual(results[0], results[-1])

    async def test_cache_byte_bound_and_expiry(self):
        self.processor.limits = replace(Limits(), cache_bytes=1)
        await self.processor.expand(payload())
        self.assertEqual(self.processor.cache_size, 0)
        self.processor.limits = replace(Limits(), cache_seconds=0)
        await self.processor.expand(payload())
        self.processor._expire()
        self.assertEqual(self.processor.cache_size, 0)

    async def test_cancelled_waiter_does_not_cancel_shared_render(self):
        started, finish = threading.Event(), threading.Event()
        def render(data, limits):
            started.set()
            finish.wait(5)
            return render_pdf(data, limits)
        self.processor.renderer = render
        first = asyncio.create_task(self.processor.expand(payload()))
        await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(self.processor.expand(payload()))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        finish.set()
        result, changed = await second
        self.assertTrue(changed)
        self.assertEqual(len(self.processor.cache), 1)

    async def test_url_pdf_uses_same_conversion(self):
        source = dict(type='url', url='https://example.test/document.pdf')
        async def fetch(url):
            self.assertEqual(url, source['url'])
            return fixture_pdf()
        with patch.object(self.processor, 'fetch', fetch):
            result, _ = await self.processor.expand(payload(dict(type='document', source=source)))
        self.assertEqual(sum(b['type'] == 'image' for b in result['messages'][0]['content']), 2)

    async def test_url_fetch_redirect_limit_and_failures(self):
        import httpx
        original_client = httpx.AsyncClient
        data = fixture_pdf()
        def transport(request):
            if request.url.path == '/redirect':
                return httpx.Response(302, headers={'location': '/pdf'})
            if request.url.path == '/missing':
                return httpx.Response(404, text='private provider details')
            return httpx.Response(200, content=data)
        def client(**kwargs):
            return original_client(transport=httpx.MockTransport(transport), **kwargs)
        with patch('httpx.AsyncClient', client):
            self.assertEqual(await self.processor.fetch('https://example.test/redirect'), data)
            with self.assertRaisesRegex(DocumentError, '^Unable to fetch PDF document URL.$'):
                await self.processor.fetch('https://example.test/missing')
            self.processor.limits = replace(Limits(), pdf_bytes=10)
            with self.assertRaisesRegex(DocumentError, 'size limit'):
                await self.processor.fetch('https://example.test/pdf')

    async def test_failed_render_is_not_cached(self):
        for _ in range(2):
            with self.assertRaises(DocumentError):
                await self.processor.expand(payload(document(b'bad PDF')))
        self.assertFalse(self.processor.cache)
        self.assertFalse(self.processor.pending)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, body, path='/v1/messages', auth=b'Bearer test-key', limits=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        received, sent = [], []
        chunks = [dict(type='http.request', body=raw[:15], more_body=True),
                  dict(type='http.request', body=raw[15:]), dict(type='http.disconnect')]
        async def receive():
            return chunks.pop(0)
        async def app(scope, receive, send):
            data = bytearray()
            while True:
                item = await receive()
                data.extend(item.get('body', b''))
                if not item.get('more_body', False):
                    break
            received.append((scope, bytes(data)))
            self.assertEqual((await receive())['type'], 'http.disconnect')
            await send(dict(type='http.response.start', status=200, headers=[]))
            await send(dict(type='http.response.body', body=b'data: first\n\n', more_body=True))
            await send(dict(type='http.response.body', body=b'data: done\n\n'))
        async def send(item):
            sent.append(item)
        processor = DocumentProcessor(limits)
        middleware = DocumentMiddleware(app, 'test-key', processor)
        scope = dict(type='http', method='POST', path=path, headers=[
            (b'authorization', auth), (b'content-length', str(len(raw)).encode())])
        try:
            await middleware(scope, receive, send)
        finally:
            await processor.close()
        return received, sent

    async def test_conversion_length_and_streaming(self):
        for path in ('/v1/messages', '/v1/messages/count_tokens'):
            received, sent = await self.invoke(payload(nested=True), path=path)
            scope, raw = received[0]
            self.assertEqual(int(dict(scope['headers'])[b'content-length']), len(raw))
            tool = json.loads(raw)['messages'][-1]['content'][1]
            self.assertEqual(sum(b['type'] == 'image' for b in tool['content']), 2)
            self.assertEqual([s['body'] for s in sent[1:]], [b'data: first\n\n', b'data: done\n\n'])

    async def test_passthrough_is_byte_identical(self):
        for raw in (b'{ "messages": [{"role":"user", "content":"hello"}] }', b'{ malformed'):
            received, _ = await self.invoke(raw)
            self.assertEqual(received[0][1], raw)

    async def test_unauthorized_never_parses_documents(self):
        body = payload(document(b'not a pdf'))
        for auth in (b'', b'Bearer wrong', b'\xff'):
            received, sent = await self.invoke(body, auth=auth)
            self.assertEqual(json.loads(received[0][1]), body)
            self.assertEqual(sent[0]['status'], 200)  # native app owns authentication

    async def test_other_routes_untouched(self):
        body = payload(document(b'not a pdf'))
        received, _ = await self.invoke(body, path='/v1/chat/completions')
        self.assertEqual(json.loads(received[0][1]), body)

    async def test_invalid_pdf_returns_anthropic_error_without_forwarding(self):
        received, sent = await self.invoke(payload(document(b'private invalid PDF data')))
        self.assertFalse(received)
        self.assertEqual(sent[0]['status'], 400)
        body = json.loads(sent[1]['body'])
        self.assertEqual(body['type'], 'error')
        self.assertNotIn('private', body['error']['message'])

    async def test_body_limit(self):
        received, sent = await self.invoke(payload(), limits=replace(Limits(), request_bytes=10))
        self.assertFalse(received)
        self.assertEqual(sent[0]['status'], 413)

    async def test_search_sees_question_and_model_sees_full_document(self):
        from web_search import WebSearchMiddleware
        queries, received = [], []
        async def search(query, include, exclude):
            queries.append(query)
            return {'results': []}
        async def app(scope, receive, send):
            raw = (await receive())['body']
            received.append(json.loads(raw))
            await send(dict(type='http.response.start', status=200, headers=[]))
            await send(dict(type='http.response.body', body=b'{}'))
        processor = DocumentProcessor()
        # Match ds41ctl registration: search wraps document expansion.
        middleware = WebSearchMiddleware(DocumentMiddleware(app, 'test-key', processor),
                                         api_key='test-key', searcher=search)
        body = payload()
        question = 'Perform a web search for the query: PDF processing'
        body['messages'][0]['content'][0]['text'] = question
        body['tools'] = [{'type': 'web_search_20250305', 'name': 'web_search'}]
        raw = json.dumps(body).encode()
        chunks = [dict(type='http.request', body=raw), dict(type='http.disconnect')]
        async def receive():
            return chunks.pop(0)
        async def send(item):
            pass
        try:
            await middleware(dict(type='http', method='POST', path='/v1/messages',
                headers=[(b'authorization', b'Bearer test-key')]), receive, send)
        finally:
            await processor.close()
        self.assertEqual(queries, ['PDF processing'])
        blocks = received[0]['messages'][0]['content']
        self.assertEqual(sum(b['type'] == 'image' for b in blocks), 2)
        self.assertIn('TEXT_CODE_3816', '\n'.join(b.get('text', '') for b in blocks))
        self.assertIn('Web search evidence', blocks[-1]['text'])


if __name__ == '__main__':
    unittest.main()
