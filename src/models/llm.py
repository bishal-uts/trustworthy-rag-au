"""Ollama HTTP client.

Talks to a local Ollama daemon over HTTP. No SDK dependency — Ollama's REST
API is small and stable.

Methods:
 - health_check()         : is Ollama reachable?
 - list_local_models()    : what models has Ollama already pulled?
 - generate(prompt)       : one-shot text generation, returns full string
 - generate_stream(prompt): generator yielding token chunks for streaming UI
 - pull_model(stream=True): trigger Ollama to pull this model, with progress

JSON-mode is enabled per `settings.generation.ollama_format_json` and the
spec's metadata. If the model doesn't support JSON-mode, the call still
works but the output may be plain text — caller should defensively parse.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from src.config import LLMSpec, settings

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


@dataclass
class GenerationResult:
    text: str
    raw_response: dict
    model_called: str
    eval_count: int
    eval_duration_ms: int


class OllamaError(RuntimeError):
    pass


class OllamaNotRunning(OllamaError):
    pass


class OllamaModelMissing(OllamaError):
    def __init__(self, model: str):
        super().__init__(
            f"Ollama model '{model}' is not pulled. Run: ollama pull {model}"
        )
        self.model = model


class OllamaLLM:
    def __init__(self, spec: LLMSpec):
        self.spec = spec
        self.id = spec.id
        self.tag = spec.ollama_tag

    # ---------- Health ----------

    @staticmethod
    def health_check() -> bool:
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(f"{OLLAMA_HOST}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def list_local_models() -> list[str]:
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(f"{OLLAMA_HOST}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError as e:
            raise OllamaNotRunning(f"Ollama not reachable at {OLLAMA_HOST}: {e}") from e

    def is_pulled(self) -> bool:
        try:
            return self.tag in self.list_local_models()
        except OllamaNotRunning:
            return False

    # ---------- Generation ----------

    def _payload(self, prompt: str, *, stream: bool) -> dict:
        gen = settings.generation
        payload: dict = {
            "model": self.tag,
            "prompt": prompt,
            "stream": stream,
            # Disable thinking for models that support a toggle (Qwen 3 / 3.5).
            # No-op for models without thinking; ignored for models where
            # thinking is intrinsic (DeepSeek-R1).
            "think": gen.enable_thinking,
            "options": {
                "temperature": gen.temperature,
                "num_predict": gen.max_tokens,
            },
            "keep_alive": gen.keep_alive,
        }
        if gen.ollama_format_json:
            payload["format"] = "json"
        return payload

    def _handle_status(self, r: httpx.Response) -> None:
        if r.status_code == 404 and "model" in r.text.lower():
            raise OllamaModelMissing(self.tag)
        r.raise_for_status()

    def generate(self, prompt: str) -> GenerationResult:
        payload = self._payload(prompt, stream=False)
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
                r = c.post(f"{OLLAMA_HOST}/api/generate", json=payload)
                self._handle_status(r)
                data = r.json()
        except httpx.ConnectError as e:
            raise OllamaNotRunning(f"Ollama not reachable at {OLLAMA_HOST}: {e}") from e
        return GenerationResult(
            text=data.get("response", ""),
            raw_response=data,
            model_called=self.tag,
            eval_count=int(data.get("eval_count", 0)),
            eval_duration_ms=int(data.get("eval_duration", 0) // 1_000_000),
        )

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield token chunks (not single tokens — Ollama batches)."""
        payload = self._payload(prompt, stream=True)
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as c, c.stream(
                "POST", f"{OLLAMA_HOST}/api/generate", json=payload
            ) as r:
                self._handle_status(r)
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = chunk.get("response", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as e:
            raise OllamaNotRunning(f"Ollama not reachable at {OLLAMA_HOST}: {e}") from e

    # ---------- Pull ----------

    def pull_model(self) -> Iterator[dict]:
        """Stream Ollama's pull progress events. Caller drives the loop."""
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)) as c, c.stream(
                "POST", f"{OLLAMA_HOST}/api/pull", json={"name": self.tag, "stream": True}
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as e:
            raise OllamaNotRunning(f"Ollama not reachable at {OLLAMA_HOST}: {e}") from e

    # ---------- Manager-compatible no-op unload ----------

    def unload(self) -> None:
        """LLMs live in Ollama's address space, not Python's. No-op here.

        To actually evict from Ollama's memory, call generate with
        keep_alive=0 — but that requires changing settings.generation.keep_alive.
        """
        pass
