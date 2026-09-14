import json

import pytest

from transform.llm import (
    LLMCallError,
    MockLLMClient,
    OllamaLLMClient,
    get_default_client,
)


def test_mock_client_returns_default_response_with_no_config():
    client = MockLLMClient()
    result = client.call("some prompt")
    assert result.mock is True
    assert result.model == "mock-llm"
    assert "error" in result.text


def test_mock_client_records_calls_for_assertions():
    client = MockLLMClient(default_response="ok")
    client.call("prompt-1", system="sys-1", temperature=0.2, max_tokens=100)
    client.call("prompt-2")

    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == "prompt-1"
    assert client.calls[0]["system"] == "sys-1"
    assert client.calls[0]["temperature"] == 0.2
    assert client.calls[0]["max_tokens"] == 100
    assert client.calls[1]["prompt"] == "prompt-2"


def test_mock_client_consumes_scripted_responses_in_order():
    client = MockLLMClient(responses=["first", "second", "third"])
    assert client.call("p").text == "first"
    assert client.call("p").text == "second"
    assert client.call("p").text == "third"


def test_mock_client_repeats_last_scripted_response_once_exhausted():
    # Simulates a re-prompt loop that retries more times than there are
    # canned responses — should not raise IndexError.
    client = MockLLMClient(responses=['{"bad": "json"', '{"ok": true}'])
    assert client.call("p").text == '{"bad": "json"'
    assert client.call("p").text == '{"ok": true}'
    assert client.call("p").text == '{"ok": true}'  # repeats, doesn't crash


def test_mock_client_response_fn_can_react_to_prompt_content():
    def responder(prompt: str, system):
        if "two sum" in prompt.lower():
            return json.dumps({"title": "Transaction Pair Match"})
        return json.dumps({"error": "unrecognized"})

    client = MockLLMClient(response_fn=responder)
    result = client.call("Rewrite: Two Sum problem")
    parsed = json.loads(result.text)
    assert parsed["title"] == "Transaction Pair Match"


def test_get_default_client_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("TRANSFORM_LLM_MODE", raising=False)
    client = get_default_client()
    assert isinstance(client, MockLLMClient)


def test_get_default_client_falls_back_to_mock_on_unknown_mode(monkeypatch):
    monkeypatch.setenv("TRANSFORM_LLM_MODE", "not-a-real-mode")
    client = get_default_client()
    assert isinstance(client, MockLLMClient)


def test_get_default_client_selects_ollama_when_configured(monkeypatch):
    monkeypatch.setenv("TRANSFORM_LLM_MODE", "ollama")
    monkeypatch.setenv("TRANSFORM_LLM_MODEL", "custom-model")
    client = get_default_client()
    assert isinstance(client, OllamaLLMClient)
    assert client.model == "custom-model"


def test_ollama_client_extract_text_handles_documented_shape():
    client = OllamaLLMClient()
    assert client._extract_text({"response": "hello"}) == "hello"


def test_ollama_client_extract_text_handles_alternate_shapes():
    client = OllamaLLMClient()
    assert client._extract_text({"text": "hello"}) == "hello"
    assert client._extract_text({"results": [{"text": "hello"}]}) == "hello"


def test_ollama_client_extract_text_raises_on_unrecognized_shape():
    client = OllamaLLMClient()
    with pytest.raises(LLMCallError):
        client._extract_text({"unexpected_key": "value"})


def test_ollama_client_raises_llm_call_error_when_unreachable():
    # No local model server is running in this environment/CI — this
    # confirms the client fails loudly (LLMCallError) rather than hanging
    # or silently returning empty text. Uses a short timeout + 0 retries
    # so the test itself doesn't hang.
    client = OllamaLLMClient(
        base_url="http://localhost:1/api/generate",  # nothing listens here
        max_retries=0,
    )
    with pytest.raises(LLMCallError):
        client.call("prompt", timeout=2.0)
