"""Anthropic document input adapter: PDF text AND page images, before validation.

SGLang 0.5.23 accepts images (also inside tool results) but its discriminated
content union rejects documents. Expand at the ASGI boundary so both its
request validator and its native vision converter receive supported blocks.
No model/package files or response semantics are patched.
"""
import asyncio
import base64
import binascii
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import logging
import math
import threading
import time

LOGGER = logging.getLogger(__name__)
MIB = 1024 * 1024
# PDFium is not thread-safe, even across independent documents. All native
# handles are created, used and explicitly closed while holding this mutex.
PDFIUM_LOCK = threading.Lock()


class DocumentError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Limits:
    request_bytes: int = 128 * MIB
    pdf_bytes: int = 96 * MIB
    pages: int = 600
    expanded_bytes: int = 256 * MIB
    page_text_chars: int = 200_000
    dpi: int = 144
    image_edge: int = 2048
    image_pixels: int = 4_000_000
    cache_bytes: int = 128 * MIB
    cache_seconds: int = 600


@dataclass(frozen=True)
class Page:
    text: str
    image: str  # base64 JPEG, never written to disk


@dataclass(frozen=True)
class PDF:
    pages: tuple[Page, ...]
    size: int


def render_pdf(data, limits):
    """Return page-aligned text and images; never silently truncate a PDF."""
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        try:
            with pdfium.PdfDocument(data) as pdf:
                if pdfium.raw.FPDF_GetSecurityHandlerRevision(pdf.raw) != -1:
                    raise DocumentError('Encrypted PDFs are unsupported; provide an unencrypted PDF.')
                pdf.init_forms()  # Include ordinary form fields in page images.
                count = len(pdf)
                if not 0 < count <= limits.pages:
                    raise DocumentError(f'PDF must have 1–{limits.pages} pages.')
                pages, size = [], 0
                for index in range(count):
                    with closing(pdf[index]) as page:
                        with closing(page.get_textpage()) as textpage:
                            if textpage.count_chars() > limits.page_text_chars:
                                raise DocumentError('PDF page contains too much text; split the document.')
                            text = textpage.get_text_bounded(errors='replace')
                        width, height = page.get_size()
                        if not all(math.isfinite(n) and n > 0 for n in (width, height)):
                            raise DocumentError('PDF has invalid page dimensions.')
                        scale = min(limits.dpi / 72, limits.image_edge / max(width, height),
                                    math.sqrt(limits.image_pixels / (width * height)))
                        with closing(page.render(scale=scale)) as bitmap:
                            # Copy before closing the PDFium-owned bitmap.
                            with bitmap.to_pil() as view, view.convert('RGB') as image:
                                buffer = io.BytesIO()
                                image.save(buffer, format='JPEG', quality=92)
                        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                        size += len(encoded) + len(text.encode('utf-8'))
                        if size > limits.expanded_bytes:
                            raise DocumentError('Rendered PDF is too large; split the document.', 413)
                        pages.append(Page(text, encoded))
                return PDF(tuple(pages), size)
        except DocumentError:
            raise
        except Exception:
            # Parser messages can contain content; do not log or expose them.
            raise DocumentError('Cannot read PDF; provide a valid, unencrypted PDF.') from None


class DocumentProcessor:
    def __init__(self, limits=None, renderer=render_pdf):
        self.limits = limits or Limits()
        self.renderer = renderer
        self.cache = OrderedDict()
        self.cache_size = 0
        self.pending = {}
        self.workers = asyncio.Semaphore(2)

    def _expire(self):
        now = time.monotonic()
        for key, (expires, pdf) in list(self.cache.items()):
            if expires <= now:
                self.cache.pop(key)
                self.cache_size -= pdf.size

    async def _produce(self, key, data):
        async with self.workers:
            pdf = await asyncio.to_thread(self.renderer, data, self.limits)
        self._expire()
        if pdf.size <= self.limits.cache_bytes:
            while self.cache and self.cache_size + pdf.size > self.limits.cache_bytes:
                _, (_, old) = self.cache.popitem(last=False)
                self.cache_size -= old.size
            self.cache[key] = (time.monotonic() + self.limits.cache_seconds, pdf)
            self.cache_size += pdf.size
        return pdf

    async def pdf(self, data):
        key = hashlib.sha256(data).digest()
        self._expire()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][1]
        task = self.pending.get(key)
        if task is None:
            task = asyncio.create_task(self._produce(key, data))
            self.pending[key] = task
            def finished(done):
                self.pending.pop(key, None)
                if not done.cancelled():
                    done.exception()  # retrieve exceptions if the client disconnected
            task.add_done_callback(finished)
        # A disconnected client must not release a running native worker slot
        # or cancel work shared by another request.
        return await asyncio.shield(task)

    async def close(self):
        if self.pending:
            await asyncio.gather(*list(self.pending.values()), return_exceptions=True)
        self.cache.clear()
        self.cache_size = 0

    def decode(self, source):
        encoded = source.get('data')
        if not isinstance(encoded, str) or len(encoded) > ((self.limits.pdf_bytes + 2) // 3) * 4:
            raise DocumentError('Document base64 data is missing or exceeds the size limit.', 400)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise DocumentError('Document data must be valid base64.') from None
        if len(data) > self.limits.pdf_bytes:
            raise DocumentError('Decoded document exceeds the size limit.', 413)
        return data

    async def fetch(self, url):
        import httpx
        if not isinstance(url, str) or not url.startswith(('https://', 'http://')):
            raise DocumentError('Document URL must use HTTP or HTTPS.')
        try:
            async with asyncio.timeout(60), httpx.AsyncClient(timeout=30, follow_redirects=True, trust_env=False) as client:
                async with client.stream('GET', url) as response:
                    response.raise_for_status()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > self.limits.pdf_bytes:
                            raise DocumentError('Downloaded PDF exceeds the size limit.', 413)
                    return bytes(data)
        except DocumentError:
            raise
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TimeoutError):
            raise DocumentError('Unable to fetch PDF document URL.') from None

    async def document(self, block, budget):
        source = block.get('source')
        if not isinstance(source, dict):
            raise DocumentError('Document source must be an object.')
        citations = block.get('citations')
        if citations is not None and (not isinstance(citations, dict) or citations.get('enabled')):
            raise DocumentError('Native document citations are unsupported; omit citations.enabled.')
        budget['documents'] += 1
        title = block.get('title') or f'Document {budget["documents"]}'
        context = block.get('context')
        if not isinstance(title, str) or (context is not None and not isinstance(context, str)):
            raise DocumentError('Document title and context must be strings.')
        heading = f'[Document {budget["documents"]}: {title}]'
        if context:
            heading += '\nContext: ' + context
        kind = source.get('type')
        media = source.get('media_type')
        text = None
        if kind == 'text':
            text = source.get('data')
            if not isinstance(text, str):
                raise DocumentError('Text document source.data must be a string.')
        elif kind == 'base64':
            data = self.decode(source)
            budget['source_bytes'] += len(data)
            if media == 'text/plain':
                try:
                    text = data.decode('utf-8')
                except UnicodeDecodeError:
                    raise DocumentError('text/plain documents must contain UTF-8 text.') from None
            elif media != 'application/pdf':
                raise DocumentError('Document media_type must be application/pdf or text/plain.')
        elif kind == 'url':
            data = await self.fetch(source.get('url'))
            budget['source_bytes'] += len(data)
        else:
            raise DocumentError('Document source.type must be base64, url (PDF), or text; Files API IDs are unsupported.')
        if budget['source_bytes'] > self.limits.pdf_bytes:
            raise DocumentError('Combined decoded documents exceed the request size limit.', 413)
        if text is not None:
            result = [{'type': 'text', 'text': heading + '\n' + text}]
            budget['expanded_bytes'] += len(result[0]['text'].encode('utf-8'))
        else:
            pdf = await self.pdf(data)
            budget['pages'] += len(pdf.pages)
            if budget['pages'] > self.limits.pages:
                raise DocumentError(f'PDFs exceed the {self.limits.pages}-page limit per request.')
            budget['expanded_bytes'] += pdf.size
            result = []
            for number, page in enumerate(pdf.pages, 1):
                label = f'{heading}\nPage {number}/{len(pdf.pages)}\nExtracted text:\n'
                label += page.text or '[No extractable text; read the page image.]'
                result.extend([{'type': 'text', 'text': label}, {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/jpeg', 'data': page.image}}])
        if budget['expanded_bytes'] > self.limits.expanded_bytes:
            raise DocumentError('Expanded documents are too large; split the request.', 413)
        return result

    async def expand(self, body):
        """Copy only changed branches, preserving tool envelopes and block order."""
        if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
            return body, False
        budget = dict(pages=0, documents=0, source_bytes=0, expanded_bytes=0)
        async def blocks(content, depth=0):
            if not isinstance(content, list):
                return content, False
            if depth > 16:
                raise DocumentError('Document content is nested too deeply.')
            result, changed = [], False
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'document':
                    result.extend(await self.document(block, budget))
                    changed = True
                elif isinstance(block, dict) and block.get('type') == 'tool_result':
                    inner, updated = await blocks(block.get('content'), depth + 1)
                    result.append(dict(block, content=inner) if updated else block)
                    changed |= updated
                else:
                    result.append(block)
            return result, changed
        messages, changed = [], False
        for message in body['messages']:
            if isinstance(message, dict):
                content, updated = await blocks(message.get('content'))
                messages.append(dict(message, content=content) if updated else message)
                changed |= updated
            else:
                messages.append(message)
        return (dict(body, messages=messages) if changed else body), changed


class DocumentMiddleware:
    def __init__(self, app, api_key, processor=None):
        self.app = app
        self.api_key = 'Bearer ' + api_key
        self.processor = processor or DocumentProcessor()

    async def error(self, send, error):
        raw = json.dumps({'type': 'error', 'error': {
            'type': 'request_too_large' if error.status == 413 else 'invalid_request_error',
            'message': str(error)}}).encode()
        await send({'type': 'http.response.start', 'status': error.status, 'headers': [
            (b'content-type', b'application/json'), (b'content-length', str(len(raw)).encode())]})
        await send({'type': 'http.response.body', 'body': raw})

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            try:
                return await self.app(scope, receive, send)
            finally:
                await self.processor.close()
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or scope.get('path') not in ('/v1/messages', '/v1/messages/count_tokens')):
            return await self.app(scope, receive, send)
        auth = dict(scope.get('headers', [])).get(b'authorization', b'')
        if not hmac.compare_digest(auth, self.api_key.encode()):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            chunk = await receive()
            if chunk['type'] == 'http.disconnect':
                return
            data = chunk.get('body', b'')
            size += len(data)
            if size > self.processor.limits.request_bytes:
                limit = self.processor.limits.request_bytes // MIB
                return await self.error(send, DocumentError(f'Request exceeds the {limit} MiB document adapter limit.', 413))
            chunks.append(data)
            if not chunk.get('more_body', False):
                break
        raw = b''.join(chunks)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            body = None  # Native validation owns malformed JSON.
        try:
            body, changed = await self.processor.expand(body)
            if changed:
                raw = json.dumps(body, ensure_ascii=False).encode()
                if len(raw) > self.processor.limits.expanded_bytes:
                    raise DocumentError('Expanded request is too large; split the request.', 413)
                scope = dict(scope, headers=[(n, v) for n, v in scope.get('headers', [])
                    if n.lower() not in (b'content-length', b'transfer-encoding')]
                    + [(b'content-length', str(len(raw)).encode())])
        except DocumentError as error:
            return await self.error(send, error)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': raw, 'more_body': False}
            return await receive()
        return await self.app(scope, replay, send)


def install(api_key, context_length):
    from sglang.srt.entrypoints.http_server import app
    limits = Limits(pages=600 if context_length >= 1_000_000 else 100)
    app.add_middleware(DocumentMiddleware, api_key=api_key, processor=DocumentProcessor(limits))
    LOGGER.info('Anthropic documents enabled: PDF text + page images, %d-page request limit', limits.pages)
