"""Diagnose why Qwen3 leaks reasoning. Prints versions, model info and raw replies.

    python scripts/diagnose_ollama.py
    python scripts/diagnose_ollama.py --model qwen3:4b --prompt "Calculate 15% of 8500"

Needs only httpx (installed with langchain-ollama). Read-only: it never changes anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

import httpx

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[1].as_posix())
from config import settings  # noqa: E402


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def raw_chat(base: str, model: str, prompt: str, think: bool) -> None:
    body = {"model": model, "stream": False, "think": think,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": settings.ollama_num_ctx}}
    label = f"think={think}"
    try:
        r = httpx.post(f"{base}/api/chat", json=body, timeout=settings.ollama_timeout_seconds)
    except httpx.HTTPError as exc:
        print(f"[{label}] request failed: {exc}")
        return
    if r.status_code != 200:
        print(f"[{label}] HTTP {r.status_code}: {r.text[:300]}")
        return
    msg = r.json().get("message", {})
    content, thinking = msg.get("content", ""), msg.get("thinking", "")
    print(f"[{label}] content has '<think>': {'<think>' in content} | has '</think>': {'</think>' in content}"
          f" | separate 'thinking' field: {bool(thinking)}")
    print(f"[{label}] content (first 300 chars): {content[:300]!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--prompt", default="Calculate 15% of 8500")
    args = parser.parse_args()
    base = settings.ollama_base_url.rstrip("/")

    print("== Versions ==")
    print("langchain-ollama:", pkg_version("langchain-ollama"))
    print("langchain-core  :", pkg_version("langchain-core"))
    print("ollama (python) :", pkg_version("ollama"))
    try:
        print("Ollama server   :", httpx.get(f"{base}/api/version", timeout=5).json().get("version"))
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Cannot reach Ollama at {base}: {exc}")
        return 1

    print("\n== Model info ==")
    try:
        info = httpx.post(f"{base}/api/show", json={"model": args.model}, timeout=15).json()
        print("details     :", json.dumps(info.get("details", {})))
        print("capabilities:", info.get("capabilities", "not reported (older Ollama)"))
        template = info.get("template", "")
        print("template mentions '<think>':", "<think>" in template)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"/api/show failed: {exc}")

    print("\n== Raw replies (bypasses LangChain) ==")
    raw_chat(base, args.model, args.prompt, think=False)
    raw_chat(base, args.model, args.prompt + " /no_think", think=False)
    print("\nPaste this whole output back to me.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
