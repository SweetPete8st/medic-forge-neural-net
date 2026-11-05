"""LLM backends for the parent agent.

- AnthropicBackend: Claude through the official SDK (default; Claude models only, per project decision).
- LocalOpenAICompatBackend: open-weight models served by Ollama / llama.cpp / vLLM, for the local-builder
  track. Never used for Claude.
- MockBackend: scripted replies for tests and offline runs.

Every backend exposes `complete(system, messages, max_tokens) -> str` and records usage/latency in
`.calls` so interviews can be stored with their cost.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CallRecord:
    backend: str
    model: str
    input_chars: int
    output_chars: int
    latency_s: float
    stop_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


class Backend:
    name = "base"
    model = ""
    def __init__(self):
        self.calls: List[CallRecord] = []
    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 4096) -> str:
        raise NotImplementedError


class AnthropicBackend(Backend):
    """Claude via the Anthropic SDK. Thinking is adaptive by default on current models; effort is explicit."""
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", use_fallbacks: bool = True):
        super().__init__()
        import anthropic  # imported here so the mock/local paths need no key
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.use_fallbacks = model, effort, use_fallbacks

    def complete(self, system, messages, max_tokens=4096):
        t0 = time.perf_counter()
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                      output_config={"effort": self.effort})
        if self.use_fallbacks:
            # server-side fallback routes a safety-classifier refusal to another model inside the same call
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if resp.stop_reason == "refusal":
            detail = getattr(resp, "stop_details", None)
            text = json.dumps({"say": "I can't continue with that request as phrased. Could you rephrase what the patient should do?",
                               "refusal": getattr(detail, "category", None) if detail else None, "slots": {}, "done": False})
        usage = getattr(resp, "usage", None)
        self.calls.append(CallRecord(self.name, resp.model, sum(len(m["content"]) for m in messages) + len(system), len(text),
                                     time.perf_counter() - t0, resp.stop_reason or "",
                                     {"input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)} if usage else {}))
        return text


class LocalOpenAICompatBackend(Backend):
    """Open-weight models behind an OpenAI-compatible endpoint (Ollama default port). Not for Claude."""
    name = "local"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/v1"):
        super().__init__(); self.model, self.base_url = model, base_url

    def complete(self, system, messages, max_tokens=4096):
        t0 = time.perf_counter()
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                           "max_tokens": max_tokens, "temperature": 0.2}).encode()
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            js = json.loads(r.read())
        msg = js["choices"][0]["message"]; text = msg.get("content") or ""
        self.calls.append(CallRecord(self.name, self.model, sum(len(m["content"]) for m in messages) + len(system), len(text),
                                     time.perf_counter() - t0, js["choices"][0].get("finish_reason", ""), js.get("usage", {})))
        return text


class MockBackend(Backend):
    """Scripted replies (a list of strings or callables taking the messages). For tests and offline demos."""
    name = "mock"
    model = "mock"

    def __init__(self, script):
        super().__init__(); self.script = list(script); self.i = 0

    def complete(self, system, messages, max_tokens=4096):
        item = self.script[min(self.i, len(self.script) - 1)]; self.i += 1
        text = item(messages) if callable(item) else item
        self.calls.append(CallRecord(self.name, self.model, 0, len(text), 0.0, "end_turn"))
        return text


def make_backend(kind: str, model: Optional[str] = None, base_url: Optional[str] = None, effort: str = "medium") -> Backend:
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-opus-5-5", effort=effort)
    if kind == "local":
        return LocalOpenAICompatBackend(model or "gpt-oss:20b", base_url or "http://127.0.0.1:11434/v1")
    if kind == "mock":
        raise ValueError("mock backend needs a script; construct MockBackend directly")
    raise ValueError(f"unknown backend {kind}")
