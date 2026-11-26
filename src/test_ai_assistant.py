import json
import os
import sys
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_assistant


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def _set_env():
    return patch.dict(os.environ, {
        "LLM_API_URL": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "gemini-2.0-flash",
    }, clear=False)


def test_gemini_request_and_response():
    with _set_env(), patch.object(
        ai_assistant.urllib.request,
        "urlopen",
        return_value=FakeResponse({"candidates": [{"content": {"parts": [{"text": "Answer"}]}}]}),
    ) as urlopen:
        result, status = ai_assistant.run_assistant("summarize", "Plants need sunlight.")

    request = urlopen.call_args.args[0]
    assert status == 200
    assert result["answer"] == "Answer"
    assert request.headers["X-goog-api-key"] == "test-key"
    assert json.loads(request.data)["contents"][0]["parts"][0]["text"]
    assert "messages" not in json.loads(request.data)


def test_gemini_endpoint_uses_configured_model():
    endpoint = ai_assistant._gemini_endpoint(
        "https://generativelanguage.googleapis.com/v1beta/models/old-model:generateContent",
        "current-model",
    )
    assert endpoint.endswith("/models/current-model:generateContent")


def test_missing_key_keeps_fallback():
    with patch.dict(os.environ, {"LLM_API_URL": "https://example.test", "LLM_API_KEY": ""}, clear=False):
        result, status = ai_assistant.run_assistant("summarize", "content")
    assert status == 503
    assert result["fallback"] is True


def test_provider_error_categories():
    for code, expected_status in ((401, 502), (404, 502), (429, 503)):
        error = HTTPError("https://example.test", code, "failure", {}, None)
        with _set_env(), patch.object(ai_assistant.urllib.request, "urlopen", side_effect=error):
            result, status = ai_assistant.run_assistant("summarize", "content")
        assert status == expected_status
        assert result["fallback"] is True
        assert "test-key" not in result["message"]


def test_network_error_keeps_fallback():
    with _set_env(), patch.object(
        ai_assistant.urllib.request, "urlopen", side_effect=URLError("offline")
    ), patch.object(ai_assistant.time, "sleep"):
        result, status = ai_assistant.run_assistant("summarize", "content")
    assert status == 503
    assert "network" in result["message"].lower()


def test_flask_route_preserves_provider_message():
    from app import app

    with patch("app.run_assistant", return_value=(
        {"status": "unavailable", "message": "The AI provider rejected the API key.", "fallback": True},
        502,
    )):
        response = app.test_client().post(
            "/api/assistant",
            json={"action": "summarize", "lesson_content": "Plants need sunlight."},
        )
    assert response.status_code == 502
    assert response.get_json()["message"] == "The AI provider rejected the API key."


def test_temporary_provider_error_retries():
    error = HTTPError("https://example.test", 503, "unavailable", {}, None)
    with _set_env(), patch.object(
        ai_assistant.urllib.request, "urlopen", side_effect=error
    ), patch.object(ai_assistant.time, "sleep") as sleep:
        result, status = ai_assistant.run_assistant("summarize", "content")
    assert status == 503
    assert "temporarily unavailable" in result["message"].lower()
    assert sleep.call_count == 2
