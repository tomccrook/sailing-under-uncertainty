"""Pluggable bounded inference; local weights and raw outputs stay out of Git."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from sail_research.contracts import Action

MODEL_REPOSITORY = "mlx-community/Qwen3.5-9B-4bit"
MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"


class Adapter(Protocol):
    def generate(
        self, messages: list[dict[str, str]], max_wall_s: float = 90
    ) -> dict[str, Any]: ...


def download_model(destination: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "purpose": "User-authorised local model evaluation",
    }
    (destination / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    try:
        folder = Path(
            snapshot_download(
                MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                token=False,
                local_dir=destination / "weights",
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md", "LICENSE*"],
                max_workers=2,
            )
        )
        files: dict[str, Any] = {
            p.relative_to(folder).as_posix(): {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in sorted(folder.rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        }
        report = {
            **request,
            "status": "succeeded",
            "download_wall_s": time.monotonic() - started,
            "files": files,
            "downloaded_bytes": sum(v["bytes"] for v in files.values()),
        }
        (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as error:
        (destination / "error.json").write_text(
            json.dumps({"error": str(error), "partial_files_retained": True}) + "\n"
        )
        raise


def verify_model(folder: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["repository"] != MODEL_REPOSITORY
        or manifest["revision"] != MODEL_REVISION
        or manifest["status"] != "succeeded"
    ):
        raise ValueError("Pinned local model provenance differs")
    for name, expected in manifest["files"].items():
        file = (folder / "weights" / name).resolve()
        if (
            not file.is_relative_to((folder / "weights").resolve())
            or hashlib.sha256(file.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Model/tokenizer/config bytes differ from the download manifest")
    return manifest


class MLXAdapter:
    def __init__(self, folder: Path, max_tokens: int = 700, max_context_tokens: int = 8192):
        # Optional platform-specific runtime; no imports at server startup.
        import mlx.core as mx
        from mlx_lm import load

        if not 1 <= max_tokens <= 1200 or not 1024 <= max_context_tokens <= 8192:
            raise ValueError("Model token budget outside allowed bounds")
        self.manifest = verify_model(folder)
        mx.random.seed(880044)
        started = time.monotonic()
        loaded = load(str(folder / "weights"), trust_remote_code=False)
        self.model = loaded[0]
        self.tokenizer: Any = loaded[1]
        self.load_wall_s = time.monotonic() - started
        self.max_tokens, self.max_context_tokens = max_tokens, max_context_tokens
        self.versions = {
            p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")
        }

    def generate(self, messages: list[dict[str, str]], max_wall_s: float = 90) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(prompt) > self.max_context_tokens:
            raise ValueError(
                "Model context budget exceeded; transcript retained, no silent truncation"
            )
        started = time.monotonic()
        chunks = []
        last = None
        wall_exhausted = False
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0),
        ):
            chunks.append(response.text)
            last = response
            if time.monotonic() - started > max_wall_s:
                wall_exhausted = True
                break
        if last is None:
            raise RuntimeError("Model produced no generation record")
        return {
            "raw": "".join(chunks),
            "wall_s": time.monotonic() - started,
            "prompt_tokens": last.prompt_tokens,
            "generation_tokens": last.generation_tokens,
            "generation_tps": last.generation_tps,
            "peak_memory_gb": last.peak_memory,
            "finish_reason": "wall_limit" if wall_exhausted else last.finish_reason,
            "model_revision": self.manifest["revision"],
            "versions": self.versions,
            "paid_service_cost_usd": 0,
            "allocation_bytes": mx.get_active_memory(),
        }


class OllamaAdapter:
    """API adapter for an owner-started local pretrained model; no download or paid call."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        client: httpx.Client | None = None,
    ):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.path != "/api/chat"
            or parsed.username
            or parsed.password
        ):
            raise ValueError("This API adapter permits only the local Ollama chat endpoint")
        self.model, self.endpoint = model, endpoint
        self.client = client or httpx.Client(timeout=90, follow_redirects=False, trust_env=False)

    def generate(self, messages: list[dict[str, str]], max_wall_s: float = 90) -> dict[str, Any]:
        started = time.monotonic()
        response = self.client.post(
            self.endpoint,
            timeout=max_wall_s,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": Action.model_json_schema(),
                "options": {"temperature": 0, "num_predict": 700, "num_ctx": 8192},
            },
        )
        response.raise_for_status()
        if len(response.content) > 100_000:
            raise ValueError("Model API response exceeds bound")
        data = response.json()
        return {
            "raw": data["message"]["content"],
            "wall_s": time.monotonic() - started,
            "prompt_tokens": data.get("prompt_eval_count"),
            "generation_tokens": data.get("eval_count"),
            "paid_service_cost_usd": 0,
            "model": data["model"],
            "runtime": "ollama-api",
            "raw_metadata": {
                k: data.get(k) for k in ("done_reason", "total_duration", "load_duration")
            },
        }
