#!/usr/bin/env python3
"""Live PDF regression against this deployment (JSON, SSE, nested Read results)."""
import datetime
import json
from pathlib import Path
import runpy
import secrets
import time
import urllib.error
import urllib.request

from test_documents import document, fixture_pdf, payload


def main():
    root = Path(__file__).resolve().parent
    ctl = runpy.run_path(str(root / 'ds41ctl'), run_name='document_smoke')
    cfg = ctl['config']()
    destination = root / 'audit' / ('documents-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    destination.mkdir(parents=True, exist_ok=False)
    code, visual = 'TEXT_' + secrets.token_hex(3), 'VISUAL_' + secrets.token_hex(3)
    pdf = fixture_pdf(code, visual)
    (destination / 'fixture.pdf').write_bytes(pdf)
    results = []
    question = ('Read both pages of the PDF. Return the code on page 1, the code on page 2, '
                'and which bar is taller in the chart. Answer briefly.')
    for nested in (False, True):
        for stream in (False, True):
            name = f'{"tool_result" if nested else "attachment"}_{"stream" if stream else "json"}'
            body = payload(document(pdf, title='PDF vision regression'), nested=nested)
            body.update(model=cfg['served_model_name'], stream=stream, max_tokens=512,
                        temperature=0, thinking={'type': 'disabled'})
            if nested:
                # Anthropic requires tool results first in the responding user
                # turn; text before a pending result violates tool-call order.
                body['messages'][-1]['content'].pop(0)
                body['messages'][0]['content'] = question
                body['messages'][-1]['content'][-1]['text'] = question
            else:
                body['messages'][0]['content'][0]['text'] = question
            started = time.monotonic()
            try:
                response = (ctl['api_stream'](cfg, body) if stream else
                            ctl['api_json'](cfg, '/v1/messages', body))
                answer = ''.join(b.get('text', '') for b in response.get('content', []))
                ended = (any(e['type'] == 'message_stop' for e in response['events']) if stream
                         else response.get('stop_reason') == 'end_turn')
                passed = code in answer and visual in answer and 'red' in answer.lower() and ended
                result = dict(name=name, passed=passed, elapsed_seconds=round(time.monotonic()-started, 3), response=response)
            except Exception as error:
                result = dict(name=name, passed=False, error=str(error))
            results.append(result)
            print(f'{name}: {"PASS" if result["passed"] else "FAIL"}', flush=True)
    # Counting must accept exactly the same document shape, not just generation.
    body = payload(document(pdf), nested=True)
    body['messages'][-1]['content'].pop(0)
    for key in ('stream', 'max_tokens'):
        body.pop(key)
    body['model'] = cfg['served_model_name']
    try:
        response = ctl['api_json'](cfg, '/v1/messages/count_tokens', body)
        results.append(dict(name='count_tokens', passed=response.get('input_tokens', 0) > 0, response=response))
    except Exception as error:
        results.append(dict(name='count_tokens', passed=False, error=str(error)))
    print(f'count_tokens: {"PASS" if results[-1]["passed"] else "FAIL"}', flush=True)
    # Error responses must be explicit 400s rather than Pydantic union errors.
    invalid = payload(document(b'not a PDF'))
    invalid['model'] = cfg['served_model_name']
    req = urllib.request.Request(f'http://{cfg["host"]}:{cfg["port"]}/v1/messages',
        data=json.dumps(invalid).encode(), headers=dict(ctl['auth_headers'](), **{'Content-Type': 'application/json'}))
    try:
        ctl['OPENER'].open(req, timeout=30).close()
        results.append(dict(name='invalid_pdf', passed=False))
    except urllib.error.HTTPError as error:
        response = json.load(error)
        results.append(dict(name='invalid_pdf', passed=error.code == 400 and
                            response.get('error', {}).get('type') == 'invalid_request_error', response=response))
    print(f'invalid_pdf: {"PASS" if results[-1]["passed"] else "FAIL"}', flush=True)
    (destination / 'summary.json').write_text(json.dumps(dict(expected_text=code, expected_visual=visual,
        results=results), indent=2) + '\n')
    print('Evidence: ' + str(destination), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
