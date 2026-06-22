"""
llm_backends.py  (project root — shared by Phase 3 and Phase 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared LLM Backend Abstraction

Lets any script in this project call ANY of four backends through one
consistent interface, selected at runtime via --backend:

  ollama     Local, free, no API key. Also supports vision (qwen3-vl:8b).
             Needs Ollama running. Can be slow on CPU-only hardware —
             if that's a bottleneck, switch the relevant phase to gemini.
  gemini     Gemini via Google AI Studio. FREE TIER, no credit card —
             see setup instructions below.
  anthropic  Claude via the Anthropic API. Paid. Needs ANTHROPIC_API_KEY.
  openai     GPT via the OpenAI API. Paid. Needs OPENAI_API_KEY.

Used by:
  - phase3/1_run_ocr.py            (vision: extracting text from images)
  - phase4/2_llm_fluency_score.py  (text: scoring OCR output quality)
  - phase4/3_llm_error_detection.py (text: flagging implausible words)
  - phase3/run_phase3.py, phase4/run_phase4.py (readiness checks)

Each backend's SDK is imported lazily — you only need the one package
installed for whichever backend you actually use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP: Getting a free Gemini API key (no credit card required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Go to https://aistudio.google.com/app/apikey
  2. Sign in with any Google account
  3. Click "Create API key" — no payment method needed
  4. Copy the key
  5. Set it in your terminal session (PowerShell):
       $env:GOOGLE_API_KEY="..."
  6. Install the SDK: pip install google-genai
     (Note: google-generativeai is deprecated — use google-genai instead)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP: 4-key rotation across multiple Gemini projects
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google's free-tier request cap is PER PROJECT PER DAY, not per account.
This project is set up for FOUR independent Google Cloud projects/keys:

  PowerShell:
    $env:GOOGLE_API_KEY_PHASE3="...key from project 1..."
    $env:GOOGLE_API_KEY_PHASE4="...key from project 2..."
    $env:GOOGLE_API_KEY_BACKUP1="...key from project 3..."
    $env:GOOGLE_API_KEY_BACKUP2="...key from project 4..."

Each Phase 3 call tries, in order: GOOGLE_API_KEY_PHASE3 → BACKUP1 →
BACKUP2 → plain GOOGLE_API_KEY. Each Phase 4 call tries:
GOOGLE_API_KEY_PHASE4 → BACKUP1 → BACKUP2 → GOOGLE_API_KEY. The two
backups are shared — whichever still has quota left gets used.

ROTATION IS AUTOMATIC and happens mid-call: if a key returns a quota
error (HTTP 429 / RESOURCE_EXHAUSTED), call_llm() immediately retries
the SAME request on the next candidate key — no waiting, since a
different project has a completely separate quota pool. It only falls
through to the caller's own retry/backoff loop if EVERY candidate key
is exhausted or unavailable. Non-quota errors (bad request, network
blip) do NOT trigger key rotation — switching keys wouldn't fix those,
so they're raised immediately for the caller's normal retry logic.

You don't have to set up all 4 — fewer keys just means less rotation
headroom. One single GOOGLE_API_KEY still works everywhere.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (from another script)
----------------------------
  from llm_backends import call_llm, check_backend_ready, DEFAULT_MODELS

  # Text-only (Phase 4 validation), with a phase-specific key env var:
  model = args.judge_model or DEFAULT_MODELS[args.backend]
  check_backend_ready(args.backend, api_key_env=args.api_key_env)
  raw_text = call_llm(args.backend, model, prompt, json_mode=True,
                       max_tokens=700, api_key_env=args.api_key_env)

  # Vision (Phase 3 OCR):
  model = args.vision_model or DEFAULT_VISION_MODELS[args.backend]
  check_backend_ready(args.backend, api_key_env=args.api_key_env)
  raw_text = call_llm(args.backend, model, prompt, image_path=image_path,
                       max_tokens=2000, api_key_env=args.api_key_env)

::: Known issue — thinking models :::
Qwen3 (and other reasoning models like DeepSeek-R1) default to "thinking
mode" — their reasoning trace shares the same output-token budget as the
final answer. This module passes "think": false to disable that for
Ollama. If you still see empty/truncated responses, some Ollama builds
have a reported bug where this isn't fully respected
(ollama/ollama#12917 on GitHub) — try increasing max_tokens, or update
Ollama to the latest version.
"""

import base64
import os

NO_PROXY = {"http": None, "https": None}  # bypass any system proxy for localhost Ollama calls

# Default models for TEXT-ONLY tasks (Phase 4 validation: scoring/judging OCR output)
DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

# Default models for VISION tasks (Phase 3 OCR: extracting text from images)
# Gemini 3 Flash wins ~65% of head-to-head OCR battles against 2.5 Flash per
# community benchmarks (ocrarena.ai, checked June 2026) and is still free-tier
# eligible. If "gemini-3-flash" 404s (naming varies / still rolling out), fall
# back to "gemini-2.5-flash" — both are valid via --vision-model override.
DEFAULT_VISION_MODELS = {
    "ollama": "qwen3-vl:8b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3-flash",
}

BACKENDS = list(DEFAULT_MODELS.keys())

# Which env var each backend's key naturally lives in, in priority order.
# A caller-supplied api_key_env (e.g. "GOOGLE_API_KEY_PHASE3") is always
# checked FIRST, ahead of these.
GENERIC_KEY_ENV_VARS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}

# Shared backup keys, tried after the phase-specific and generic ones,
# before finally giving up. Only meaningful for gemini (where quota
# exhaustion is the actual problem this solves).
BACKUP_KEY_ENV_VARS = {
    "gemini": ["GOOGLE_API_KEY_BACKUP1", "GOOGLE_API_KEY_BACKUP2"],
}

# Substrings that indicate "this key is out of quota, try another one" —
# as opposed to a genuine request error that switching keys won't fix.
QUOTA_ERROR_MARKERS = ["RESOURCE_EXHAUSTED", "429", "quota", "rate limit", "rate_limit"]


def _candidate_key_names(backend: str, api_key_env: str = None) -> list:
    """Ordered, de-duplicated list of env var NAMES to try for this backend."""
    names = ([api_key_env] if api_key_env else [])
    names += GENERIC_KEY_ENV_VARS.get(backend, [])
    names += BACKUP_KEY_ENV_VARS.get(backend, [])
    seen = set()
    ordered = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def resolve_api_key(backend: str, api_key_env: str = None):
    """
    Find the FIRST available API key value, checking candidate env vars
    in priority order. Returns (key_value, env_var_name_used) or
    (None, None) if nothing found. (Kept for backward compatibility /
    simple callers — call_llm() itself uses the full candidate list for
    rotation, not just the first match.)
    """
    for name in _candidate_key_names(backend, api_key_env):
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker.lower() in msg.lower() for marker in QUOTA_ERROR_MARKERS)


def check_backend_ready(backend: str, api_key_env: str = None) -> None:
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
                "  2. Pull a model: ollama pull qwen3:8b  (or qwen3-vl:8b for OCR)\n"
                "  3. Confirm it's running: curl http://localhost:11434"
            )
        return

    if backend not in GENERIC_KEY_ENV_VARS:
        raise SystemExit(f"ERROR: Unknown backend '{backend}'. Choose from: {BACKENDS}")

    names = _candidate_key_names(backend, api_key_env)
    available = [n for n in names if os.environ.get(n)]

    if not available:
        setup_hints = {
            "anthropic": 'Get a key at https://console.anthropic.com (separate billing from claude.ai)\n'
                         '  PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."',
            "openai": 'Get a key at https://platform.openai.com\n'
                      '  PowerShell: $env:OPENAI_API_KEY="sk-..."',
            "gemini": 'Get a FREE key (no credit card) at https://aistudio.google.com/app/apikey\n'
                      '  PowerShell: $env:GOOGLE_API_KEY="..."\n'
                      '  (or set up phase-specific + backup keys — see llm_backends.py docstring)',
        }
        raise SystemExit(
            f"ERROR: No API key found for backend '{backend}'.\n"
            f"  Checked environment variable(s): {', '.join(names)}\n"
            f"  {setup_hints[backend]}"
        )

    if backend == "gemini" and len(available) > 1:
        print(f"[llm_backends] {len(available)} Gemini key(s) available for rotation: {', '.join(available)}")


def call_llm(
    backend: str,
    model: str,
    prompt: str,
    image_path=None,
    json_mode: bool = False,
    max_tokens: int = 700,
    api_key_env: str = None,
) -> str:
    """
    Call the given backend/model with prompt, optionally attaching an
    image (for vision/OCR tasks). Returns the raw text response.

    For gemini specifically: automatically rotates through every
    available candidate key (phase-specific -> generic -> backups) on
    quota-exhaustion errors, retrying the SAME request on the next key
    with no delay (different project = different quota pool, so there's
    nothing to wait out). Raises only if every candidate is exhausted,
    or immediately on a non-quota error (callers handle those retries).
    """
    image_b64 = None
    if image_path is not None:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

    if backend == "ollama":
        import requests
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        if image_b64:
            payload["images"] = [image_b64]
        resp = requests.post(
            "http://localhost:11434/api/generate", json=payload, timeout=180, proxies=NO_PROXY
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    elif backend == "anthropic":
        import anthropic
        key, _ = resolve_api_key("anthropic", api_key_env)
        client = anthropic.Anthropic(api_key=key)
        content = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            })
        content.append({"type": "text", "text": prompt})
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()
        return raw.replace("```json", "").replace("```", "").strip() if json_mode else raw

    elif backend == "openai":
        import openai
        key, _ = resolve_api_key("openai", api_key_env)
        client = openai.OpenAI(api_key=key)
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    elif backend == "gemini":
        from google import genai
        from google.genai import types

        key_names = [n for n in _candidate_key_names("gemini", api_key_env) if os.environ.get(n)]
        if not key_names:
            raise RuntimeError(
                f"No Gemini API key available. Checked: {_candidate_key_names('gemini', api_key_env)}"
            )

        config_kwargs = {"max_output_tokens": max_tokens}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**config_kwargs)

        contents = [prompt]
        if image_b64:
            contents.append(types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/png"))

        last_exc = None
        for i, key_name in enumerate(key_names):
            try:
                client = genai.Client(api_key=os.environ[key_name])
                response = client.models.generate_content(model=model, contents=contents, config=config)
                return response.text.strip()
            except Exception as e:
                last_exc = e
                if _is_quota_error(e) and i < len(key_names) - 1:
                    print(f"[llm_backends] '{key_name}' exhausted (quota error) — "
                          f"rotating to '{key_names[i + 1]}'...")
                    continue
                raise
        raise last_exc

    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {BACKENDS}")
