"""Optional, server-side classroom assistant with Gemini and OpenAI-compatible support."""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("sign_ai.ai_assistant")
BASE_DIR = Path(__file__).resolve().parent.parent

SUPPORTED_ACTIONS = {
    "explain",
    "summarize",
    "quiz",
    "answer",
    "translate",
    "steps",
    "revision",
}


def _bounded_text(value, limit):
    return str(value or "").strip()[:limit]


def _load_local_env():
    """Load simple KEY=VALUE entries for local runs without overriding the process."""
    env_path = BASE_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("Unable to read local environment file: %s", exc)


def _provider_configuration():
    _load_local_env()
    return (
        os.environ.get("LLM_API_URL", "").strip(),
        os.environ.get("LLM_API_KEY", "").strip(),
        os.environ.get("LLM_MODEL", "").strip(),
    )


def _is_gemini_endpoint(endpoint):
    return "generativelanguage.googleapis.com" in urlparse(endpoint).netloc


def _gemini_endpoint(endpoint, model):
    """Keep the endpoint model path synchronized with LLM_MODEL."""
    if not _is_gemini_endpoint(endpoint):
        return endpoint
    return re.sub(
        r"(/models/)[^/:]+(:generateContent)$",
        rf"\g<1>{model}\g<2>",
        endpoint,
    )


def _provider_error(status):
    messages = {
        400: "The AI provider rejected the request. Check the model and request configuration.",
        401: "The AI provider rejected the API key.",
        403: "The AI provider denied access. Check the API key, project, or API permissions.",
        404: "The configured Gemini model or endpoint was not found.",
        429: "The AI provider quota or rate limit was reached.",
        500: "The AI provider encountered an internal error. Try again later.",
        502: "The AI provider gateway returned an error. Try again later.",
        503: "The AI provider is temporarily unavailable. Try again later.",
    }
    return messages.get(status, "The AI provider returned an error.")


def _build_prompt(action, lesson_content, question, target_language):
    action_instructions = {
        "explain": "Explain the difficult sentence in simple language.",
        "summarize": "Summarize the lesson in short, accurate bullet points.",
        "quiz": "Create five short questions from the lesson. Do not add facts not present in it.",
        "answer": "Answer the student's question only from the lesson content. If it is not answered there, say that the lesson does not provide enough information.",
        "translate": "Translate the lesson explanation into the requested language without adding facts.",
        "steps": "Convert the lesson instructions into short, numbered steps.",
        "revision": "Create concise revision notes and key points from the lesson.",
    }
    return (
        "You are an optional classroom assistant. Use only the supplied lesson content. "
        "Never invent lesson facts; explicitly say when the content is insufficient. "
        "Keep language age-appropriate and concise.\n\n"
        f"Task: {action_instructions[action]}\n"
        f"Target language: {target_language}\n"
        f"Lesson content:\n{lesson_content or '[No lesson content supplied]'}\n"
        f"Student question (if any): {question or '[None]'}"
    )


def run_assistant(action, lesson_content="", question="", target_language="en"):
    """Return a structured result and never expose provider errors or credentials."""
    if action not in SUPPORTED_ACTIONS:
        return {"status": "failed", "error": "Unsupported assistant action"}, 400

    endpoint, api_key, model = _provider_configuration()
    if not model:
        model = "gemini-3.1-flash-lite"
    if not endpoint or not api_key:
        return {
            "status": "unavailable",
            "message": "AI assistant is not configured. Existing classroom tools remain available.",
            "fallback": True,
        }, 503

    endpoint = _gemini_endpoint(endpoint, model)
    prompt = _build_prompt(
        action,
        _bounded_text(lesson_content, 12000),
        _bounded_text(question, 2000),
        _bounded_text(target_language, 20) or "en",
    )
    if _is_gemini_endpoint(endpoint):
        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        payload = json.dumps({
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Return only the classroom response, with no hidden reasoning."},
                {"role": "user", "content": prompt},
            ],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    provider_name = "gemini" if _is_gemini_endpoint(endpoint) else "openai-compatible"
    last_network_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", 200)
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (500, 502, 503, 504) and attempt < 2:
                logger.warning(
                    "Assistant provider temporary HTTP error: provider=%s status=%s attempt=%s/3",
                    provider_name, exc.code, attempt + 1,
                )
                time.sleep(2 ** attempt)
                continue
            logger.warning("Assistant provider HTTP error: provider=%s status=%s", provider_name, exc.code)
            message = _provider_error(exc.code)
            status_code = 503 if exc.code == 429 or exc.code >= 500 else 502
            return {"status": "unavailable", "message": message, "fallback": True}, status_code
        except (urllib.error.URLError, TimeoutError) as exc:
            last_network_error = exc
            if attempt < 2:
                logger.warning(
                    "Assistant provider network retry: provider=%s error=%s attempt=%s/3",
                    provider_name, type(exc).__name__, attempt + 1,
                )
                time.sleep(2 ** attempt)
                continue
            logger.warning("Assistant provider network error: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "message": "The AI provider could not be reached after 3 attempts. Check the server network and endpoint.",
                "fallback": True,
            }, 503
    else:
        logger.warning("Assistant provider request exhausted retries: provider=%s error=%s", provider_name, type(last_network_error).__name__)
        return {
            "status": "unavailable",
            "message": "The AI provider could not be reached after 3 attempts. Check the server network and endpoint.",
            "fallback": True,
        }, 503
    try:
        if _is_gemini_endpoint(endpoint):
            answer = "".join(
                part.get("text", "")
                for part in body["candidates"][0]["content"]["parts"]
            ).strip()
        else:
            answer = body["choices"][0]["message"]["content"].strip()
        if not answer:
            raise ValueError("Provider returned an empty answer")
        logger.info("Assistant provider request succeeded: provider=%s status=%s", provider_name, status)
        return {"status": "success", "answer": answer, "action": action}, 200
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Assistant provider response parsing error: %s", type(exc).__name__)
        return {
            "status": "unavailable",
            "message": "The AI provider returned an unexpected response. Check the configured model and endpoint.",
            "fallback": True,
        }, 503
