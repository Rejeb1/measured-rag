"""LLM client tests.

The `num_ctx` test is the load-bearing one. Ollama defaults that option to 4096
regardless of the model's advertised context window, and long prompts do not degrade
gracefully there - a measured sweep showed the 1264-token aggregation prompt failing
5/5 at 4096 and passing 5/5 at 8192, with failures arriving as empty responses, HTTP
500s, and degenerate repeated-token output. It cost two golden-set pilots to pin down,
so it gets a test.

The retry tests cover the two shapes that slipped through earlier versions: an empty
body (caught in complete()) and a *truncated* one (non-empty, so only the parse step
sees it). Both are transient; a persistent failure must still raise, because silently
returning "" would be indistinguishable from a refusal and would inflate abstention
accuracy.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragmed.eval.build_golden import (
    MULTI_HOP_RULE_NAMES,
    RULE_NAMES,
    UNANSWERABLE_RULE_NAMES,
    _screen,
)
from ragmed.llm import LLMUnavailable, NullLLM, OllamaLLM, extract_json


class FakeTransport(httpx.BaseTransport):
    """Records outbound payloads and replays a scripted sequence of responses."""

    def __init__(self, responses: list[tuple[int, dict | str]]):
        self.responses = responses
        self.requests: list[dict] = []
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status, body = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)


def make_llm(responses, **kw) -> tuple[OllamaLLM, FakeTransport]:
    transport = FakeTransport(responses)
    llm = OllamaLLM(model="test:3b", **kw)
    llm._client = httpx.Client(transport=transport)
    llm._available = True
    return llm, transport


def ok(text: str) -> tuple[int, dict]:
    return 200, {"message": {"content": text}}


# --- num_ctx -------------------------------------------------------------------


def test_num_ctx_is_always_sent():
    """Omitting it means Ollama silently truncates to 4096."""
    llm, t = make_llm([ok("hi")], num_ctx=8192)
    llm.complete("prompt")
    assert t.requests[0]["options"]["num_ctx"] == 8192


def test_num_ctx_is_configurable_per_call():
    llm, t = make_llm([ok("hi")], num_ctx=8192)
    llm.complete("prompt", num_ctx=32768)
    assert t.requests[0]["options"]["num_ctx"] == 32768


def test_temperature_and_num_predict_are_sent():
    llm, t = make_llm([ok("hi")], temperature=0.0, max_tokens=500)
    llm.complete("prompt")
    opts = t.requests[0]["options"]
    assert opts["temperature"] == 0.0
    assert opts["num_predict"] == 500


def test_json_mode_sets_the_format_flag():
    llm, t = make_llm([ok('{"a": 1}')])
    assert llm.complete_json("prompt") == {"a": 1}
    assert t.requests[0]["format"] == "json"


def test_plain_completion_does_not_request_json():
    llm, t = make_llm([ok("hi")])
    llm.complete("prompt")
    assert "format" not in t.requests[0]


# --- transient failure handling --------------------------------------------------


def test_empty_response_is_retried():
    """A 3B model on a 4GB card intermittently returns nothing; one retry fixes it."""
    llm, t = make_llm([ok(""), ok("recovered")], max_retries=3)
    assert llm.complete("prompt") == "recovered"
    assert t.calls == 2


def test_server_error_is_retried():
    llm, t = make_llm([(500, "boom"), ok("recovered")], max_retries=3)
    assert llm.complete("prompt") == "recovered"
    assert t.calls == 2


def test_persistent_failure_raises_rather_than_returning_empty():
    """Silently returning "" would look like a refusal and corrupt abstention metrics."""
    llm, t = make_llm([ok("")], max_retries=2)
    with pytest.raises(LLMUnavailable, match="after 2 attempts"):
        llm.complete("prompt")
    assert t.calls == 2


def test_whitespace_only_response_counts_as_empty():
    llm, _ = make_llm([ok("   \n  ")], max_retries=1)
    with pytest.raises(LLMUnavailable):
        llm.complete("prompt")


def test_truncated_json_is_retried():
    """'{"question": "What' is non-empty, so only the parse step catches it."""
    llm, t = make_llm([ok('{"question": "What'), ok('{"question": "ok"}')], max_retries=3)
    assert llm.complete_json("prompt") == {"question": "ok"}
    assert t.calls == 2


def test_persistently_unparseable_json_raises():
    from ragmed.llm import LLMParseError

    llm, t = make_llm([ok("not json at all")], max_retries=2)
    with pytest.raises(LLMParseError):
        llm.complete_json("prompt")
    assert t.calls == 2


def test_valid_json_costs_exactly_one_call():
    llm, t = make_llm([ok('{"a": 1}')], max_retries=3)
    assert llm.complete_json("prompt") == {"a": 1}
    assert t.calls == 1


def test_system_prompt_is_passed_through():
    llm, t = make_llm([ok("hi")])
    llm.complete("question", system="you are a judge")
    msgs = t.requests[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "you are a judge"}
    assert msgs[1]["content"] == "question"


# --- NullLLM --------------------------------------------------------------------


def test_null_llm_refuses_clearly():
    llm = NullLLM()
    assert llm.available() is False
    with pytest.raises(Exception, match="no LLM configured"):
        llm.complete("anything")


# --- screener rule-number decoding -------------------------------------------------


class ScriptedLLM:
    name = "scripted"

    def __init__(self, payload):
        self.payload = payload

    def available(self):
        return True

    def complete(self, prompt, system=None, **kw):
        return json.dumps(self.payload)

    def complete_json(self, prompt, system=None, **kw):
        return extract_json(self.complete(prompt, system, **kw))

    def stream(self, prompt, system=None, **kw):
        yield self.complete(prompt, system, **kw)


def test_rule_numbers_are_decoded_to_readable_names():
    """The first pilot reported its top rejection reason as "27  1" — useless."""
    llm = ScriptedLLM({"keep": False, "violated_rules": [1, 4]})
    keep, reasons = _screen(llm, "q?", "a", [], rule_names=RULE_NAMES)
    assert keep is False
    assert reasons == [RULE_NAMES[1], RULE_NAMES[4]]


def test_numeric_strings_are_decoded_too():
    llm = ScriptedLLM({"keep": False, "violated_rules": ["1", "2."]})
    _, reasons = _screen(llm, "q?", "a", [])
    assert reasons == [RULE_NAMES[1], RULE_NAMES[2]]


def test_legacy_reasons_key_is_still_accepted():
    llm = ScriptedLLM({"keep": False, "reasons": [3]})
    _, reasons = _screen(llm, "q?", "a", [])
    assert reasons == [RULE_NAMES[3]]


def test_type_specific_rules_resolve_to_their_own_names():
    llm = ScriptedLLM({"keep": False, "violated_rules": [6]})
    _, mh = _screen(llm, "q?", "a", [], rule_names=MULTI_HOP_RULE_NAMES)
    _, un = _screen(llm, "q?", "a", [], rule_names=UNANSWERABLE_RULE_NAMES)
    assert "multi-hop" in mh[0]
    assert "answerable" in un[0]
    assert mh != un


def test_rejection_without_a_stated_rule_is_still_a_rejection():
    llm = ScriptedLLM({"keep": False})
    keep, reasons = _screen(llm, "q?", "a", [])
    assert keep is False and reasons == ["rejected without a stated rule"]


def test_a_clean_question_is_kept():
    llm = ScriptedLLM({"keep": True, "violated_rules": []})
    keep, reasons = _screen(llm, "q?", "a", [])
    assert keep is True and reasons == []


def test_unparseable_screener_output_rejects_rather_than_admits():
    """Screening must fail closed: an unscreened question must never reach the set."""

    class Broken(ScriptedLLM):
        def complete(self, prompt, system=None, **kw):
            return "I cannot comply."

    keep, reasons = _screen(Broken(None), "q?", "a", [])
    assert keep is False
    assert reasons and "screening_error" in reasons[0]
