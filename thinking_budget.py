"""Connect Anthropic thinking budgets to SGLang 0.5.23's native grammar masks.

Loaded by ds41ctl in the API process and spawned scheduler processes. Package
files are untouched. No client-supplied executable logit processor is enabled.
"""
from functools import wraps


def install():
    from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
    from sglang.srt.constrained.grammar_manager import GrammarManager

    original_convert = AnthropicServing._convert_to_chat_completion_request
    if getattr(original_convert, '_ds41_budget', False):
        return

    @wraps(original_convert)
    def convert(self, request):
        thinking = request.thinking
        budget = thinking.budget_tokens if thinking is not None else None
        if budget is None:
            return original_convert(self, request)
        # The validated enabled mode and adaptive mode have the same native
        # toggle semantics. Normalize a copy so upstream doesn't emit its
        # now-inapplicable "budget is not enforced" warning. Preserve effort,
        # tools, images, history and the original request's response settings.
        normalized = request.model_copy(update={'thinking': thinking.model_copy(
            update={'type': 'adaptive', 'budget_tokens': None})})
        chat = original_convert(self, normalized)
        chat.custom_params = dict(chat.custom_params or {}, thinking_budget=budget)
        return chat

    original_process = GrammarManager.process_req_with_grammar

    @wraps(original_process)
    def process(self, req):
        params = req.sampling_params
        custom = params.custom_params or {}
        budget = custom.get('thinking_budget')
        constrained = any(getattr(params, name) is not None for name in
                          ('json_schema', 'regex', 'ebnf', 'structural_tag'))
        if not constrained and not (type(budget) is int and budget >= 0):
            # --enable-strict-thinking normally attaches a grammar to every
            # request. Keep ordinary unbudgeted requests on the fast path.
            return False
        return original_process(self, req)

    convert._ds41_budget = True
    AnthropicServing._convert_to_chat_completion_request = convert
    GrammarManager.process_req_with_grammar = process
