"""Native SGLang protocol/converter regression; use deployment CUDA environment.

No model is loaded. Confirms that ASGI normalization fixes both the request
union error and the downstream tool-result multimodal conversion.
"""
import copy
from types import SimpleNamespace
import unittest

from pydantic import ValidationError
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest, AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

from documents import DocumentProcessor
from test_documents import payload


class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_validator_and_converter_in_both_document_locations(self):
        processor = DocumentProcessor()
        serving = AnthropicServing(SimpleNamespace(supports_inline_system=True))
        try:
            for nested in (False, True):
                with self.subTest(nested=nested):
                    original = payload(nested=nested)
                    # Reproduce the reported failure before expansion.
                    with self.assertRaisesRegex(ValidationError, "Input tag 'document'"):
                        AnthropicMessagesRequest.model_validate(original)
                    expanded, changed = await processor.expand(original)
                    self.assertTrue(changed)
                    parsed = AnthropicMessagesRequest.model_validate(expanded)
                    converted = serving._convert_to_chat_completion_request(parsed).model_dump(exclude_none=True)
                    images, texts = [], []
                    for message in converted['messages']:
                        content = message.get('content')
                        if isinstance(content, list):
                            images.extend(part for part in content if part.get('type') == 'image_url')
                            texts.extend(part['text'] for part in content if part.get('type') == 'text')
                        elif isinstance(content, str):
                            texts.append(content)
                    self.assertEqual(len(images), 2)
                    self.assertTrue(all(p['image_url']['url'].startswith('data:image/jpeg;base64,') for p in images))
                    self.assertIn('TEXT_CODE_3816', '\n'.join(texts))
                    self.assertIn('Page 2/2', '\n'.join(texts))
                    if nested:
                        tools = [m for m in converted['messages'] if m['role'] == 'tool']
                        self.assertTrue(tools)
                        self.assertTrue(all(m['tool_call_id'] == 'toolu_01' for m in tools))
                        self.assertEqual(converted['messages'][-1]['content'], 'after')
                    # Count-token input shares the same recursive block union.
                    counted = copy.deepcopy(expanded)
                    counted.pop('max_tokens')
                    counted.pop('stream')
                    AnthropicCountTokensRequest.model_validate(counted)
        finally:
            await processor.close()


if __name__ == '__main__':
    unittest.main()
