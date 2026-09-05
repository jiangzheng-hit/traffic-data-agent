from __future__ import annotations

import json
from typing import Any
from urllib import request


DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def parse_model_names(payload: dict[str, Any]) -> list[str]:
    """Return normalized model names from Ollama's /api/tags response."""
    names = []
    for item in payload.get("models", []):
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names))


def check_ollama(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> dict[str, Any]:
    """Check the local Ollama service without sending dataset contents."""
    try:
        with request.urlopen(f"{base_url}/api/tags", timeout=timeout) as response:
            tags = json.loads(response.read().decode("utf-8"))
        version = None
        try:
            with request.urlopen(f"{base_url}/api/version", timeout=timeout) as response:
                version = json.loads(response.read().decode("utf-8")).get("version")
        except Exception:
            pass
        return {
            "available": True,
            "version": version,
            "models": parse_model_names(tags),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "models": [],
            "error": type(exc).__name__,
        }
