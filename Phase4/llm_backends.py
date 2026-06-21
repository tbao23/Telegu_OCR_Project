"""
phase4/llm_backends.py
━━━━━━━━━━━━━━━━━━━━━━
Shared LLM Backend Abstraction (not a pipeline step — imported by
2_llm_fluency_score.py and 3_llm_error_detection.py)

Lets the Phase 4 LLM validation scripts call ANY of four backends through
one consistent interface, selected at runtime via --backend:

  ollama     Local, free, no API key (default). Needs Ollama running.
  anthropic  Claude via the Anthropic API. Needs ANTHROPIC_API_KEY.
  openai     GPT via the OpenAI API. Needs OPENAI_API_KEY.
  gemini     Gemini via the Google AI API. Needs GOOGLE_API_KEY.

Each backend's SDK is imported lazily — you only need the one package
installed for whichever backend you actually use.

::: Known issue :::
Qwen3 (and other reasoning models like DeepSeek-R1) default to "thinking
mode" — their reasoning trace shares the same output-token budget as the
final answer. For tasks needing a short, direct response (our use case),
this can exhaust the budget on deliberation and leave an empty final
answer. This module passes "think": false to disable that. If you still
see empty/truncated responses, some Ollama builds have a reported bug
where this isn't fully respected (ollama/ollama#12917 on GitHub) — try
increasing max_tokens, or update Ollama to the latest version.

Usage (from another script)
----------------------------
  from llm_backends import call_llm, check_backend_ready, DEFAULT_MODELS

  model = args.judge_model or DEFAULT_MODELS[args.backend]
  check_backend_ready(args.backend)  # raises SystemExit with a clear message if not ready
  raw_text = call_llm(args.backend, model, prompt, json_mode=True, max_tokens=700)
"""

import os

NO_PROXY = {"http": None, "https": None}  # bypass any system proxy for localhost Ollama calls

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}

BACKENDS = list(DEFAULT_MODELS.keys())


def check_backend_ready(backend: str) -> None:
    """Raise SystemExit with a clear, actionable message if the chosen
    backend can't be used right now (missing key / server not running)."""
    if backend == "ollama":
        import requests
        try:
            requests.get("http://localhost:11434", timeout=3, proxies=NO_PROXY)
        except requests.exceptions.RequestException:
            raise SystemExit(
                "ERROR: Ollama is not reachable at http://localhost:11434.\n"
                "  1. Install: https://ollama.com/download\n"
                "  2. Pull a model: ollama pull qwen3:8b\n"
                "  3. Confirm it's running: curl http://localhost:11434"
            )

    elif backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ERROR: ANTHROPIC_API_KEY environment variable not set.\n"
                "  Get a key at https://console.anthropic.com (separate billing from claude.ai)\n"
                '  PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."'
            )

    elif backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "ERROR: OPENAI_API_KEY environment variable not set.\n"
                "  Get a key at https://platform.openai.com\n"
                '  PowerShell: $env:OPENAI_API_KEY="sk-..."'
            )

    elif backend == "gemini":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise SystemExit(
                "ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set.\n"
                "  Get a key at https://aistudio.google.com/app/apikey\n"
                '  PowerShell: $env:GOOGLE_API_KEY="AIza..."'
            )

    else:
        raise SystemExit(f"ERROR: Unknown backend '{backend}'. Choose from: {BACKENDS}")


def call_llm(backend: str, model: str, prompt: str, json_mode: bool = False, max_tokens: int = 700) -> str:
    """
    Call the given backend/model with prompt. Returns the raw text response.
    Raises an exception on failure — callers should catch and handle retries.
    """
    if backend == "ollama":
        import requests
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Qwen3/DeepSeek-R1 etc. default to thinking mode, which shares
                              # the num_predict budget with the final answer — for tasks like
                              # ours (short, direct outputs) this can exhaust the budget on
                              # reasoning and leave an empty response. Disable it explicitly.
            "options": {"num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(
            "http://localhost:11434/api/generate", json=payload, timeout=120, proxies=NO_PROXY
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        return raw.replace("```json", "").replace("```", "").strip() if json_mode else raw

    elif backend == "openai":
        import openai
        client = openai.OpenAI()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    elif backend == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        gen_config = {"max_output_tokens": max_tokens}
        if json_mode:
            gen_config["response_mime_type"] = "application/json"
        client = genai.GenerativeModel(model)
        response = client.generate_content(prompt, generation_config=gen_config)
        return response.text.strip()

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {BACKENDS}")
