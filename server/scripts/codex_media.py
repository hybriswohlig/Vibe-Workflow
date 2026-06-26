#!/usr/bin/env python3
"""OPENAI_MEDIA_COMMAND backend: GPT image via the ChatGPT/Codex subscription.

This calls the SAME Responses endpoint the Codex CLI uses, with the built-in
`image_generation` tool — a direct API call (fast, seconds) rather than driving
the Codex agent. No OPENAI_API_KEY needed; it uses the ~/.codex login token.

Ported from Open-Generative-AI (app/lib/codexImage.js + codexAuth.js).

Contract (matches workflow_helper._call_command):
  stdin : {"model": "gpt-image-2", "params": {"prompt","image_url","images_list","aspect_ratio"}}
  stdout: {"images": [{"url": "http://localhost:8000/uploads/<file>.png"}]}
On failure: write the reason to stderr and exit non-zero so the server surfaces it.
"""
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOADS_DIR = ROOT / "server" / "data" / "uploads"
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000")
CODEX_HOME = Path(os.getenv("CODEX_MEDIA_HOME") or os.path.expanduser("~/.codex"))
CODEX_BIN = os.getenv("CODEX_BIN", "/opt/homebrew/bin/codex")
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
TIMEOUT = int(os.getenv("CODEX_MEDIA_TIMEOUT", "300"))

# The image_generation tool only accepts these sizes.
VALID_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
ASPECT_TO_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1536x1024", "4:3": "1536x1024", "3:2": "1536x1024", "21:9": "1536x1024",
    "9:16": "1024x1536", "3:4": "1024x1536", "2:3": "1024x1536",
}


def die(message: str) -> None:
    sys.stderr.write(message.strip() + "\n")
    sys.exit(1)


def as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value or "")


def read_auth() -> dict | None:
    path = CODEX_HOME / "auth.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokens = data.get("tokens") or {}
    token = tokens.get("access_token") or data.get("access_token") or data.get("OPENAI_API_KEY")
    account = tokens.get("account_id") or data.get("account_id")
    if not token:
        return None
    return {"token": token, "account": account}


def read_codex_model() -> str | None:
    try:
        text = (CODEX_HOME / "config.toml").read_text(encoding="utf-8")
        m = re.search(r'^\s*model\s*=\s*"([^"]+)"', text, re.M)
        return m.group(1) if m else None
    except Exception:
        return None


def _pick_image(evt) -> tuple[str | None, str | None]:
    full = partial = None
    if not isinstance(evt, dict):
        return full, partial
    pi = evt.get("partial_image_b64")
    if isinstance(pi, str) and len(pi) > 100:
        partial = pi
    item = evt.get("item") or {}
    if item.get("type") == "image_generation_call" and isinstance(item.get("result"), str):
        full = item["result"]
    if isinstance(evt.get("result"), str) and len(evt["result"]) > 100:
        full = evt["result"]
    out = (evt.get("response") or {}).get("output") or evt.get("output")
    if isinstance(out, list):
        for o in out:
            if not isinstance(o, dict):
                continue
            if o.get("type") == "image_generation_call" and isinstance(o.get("result"), str):
                full = o["result"]
            for c in (o.get("content") or []):
                if not isinstance(c, dict):
                    continue
                iu = c.get("image_url")
                if isinstance(iu, str) and iu.startswith("data:") and "," in iu:
                    full = iu.split(",", 1)[1]
                if isinstance(c.get("b64_json"), str):
                    full = c["b64_json"]
    return full, partial


def extract_image_b64(raw: str) -> str | None:
    full = last_partial = None
    saw_sse = False
    for line in raw.split("\n"):
        t = line.strip()
        if not t.startswith("data:"):
            continue
        saw_sse = True
        js = t[5:].strip()
        if not js or js == "[DONE]":
            continue
        try:
            evt = json.loads(js)
        except Exception:
            continue
        f, p = _pick_image(evt)
        if f:
            full = f
        if p:
            last_partial = p
    if not saw_sse:
        try:
            f, p = _pick_image(json.loads(raw))
            full = f or full
            last_partial = p or last_partial
        except Exception:
            pass
    return full or last_partial


def _to_data_url(ref: str) -> str | None:
    ref = as_text(ref).strip()
    if not ref:
        return None
    local = None
    if ref.startswith(("http://", "https://")):
        if "/uploads/" in ref:
            cand = UPLOADS_DIR / ref.split("/uploads/", 1)[1].split("?", 1)[0]
            if cand.exists():
                local = cand
        if local is None:
            try:
                raw = urllib.request.urlopen(ref, timeout=30).read()
                return "data:image/png;base64," + base64.b64encode(raw).decode()
            except Exception:
                return None
    else:
        cand = Path(ref) if Path(ref).is_absolute() else UPLOADS_DIR / Path(ref).name
        if cand.exists():
            local = cand
    if local is not None:
        ext = local.suffix.lstrip(".") or "png"
        return f"data:image/{ext};base64," + base64.b64encode(local.read_bytes()).decode()
    return None


def build_input(prompt: str, image_refs: list) -> list:
    content = [{"type": "input_text", "text": prompt}]
    for ref in image_refs:
        data_url = _to_data_url(ref)
        if data_url:
            content.append({"type": "input_image", "image_url": data_url})
    return [{"type": "message", "role": "user", "content": content}]


def call_codex(auth: dict, model: str, payload_input: list, size: str) -> tuple[int, str]:
    payload = {
        "model": model,
        "instructions": (
            "You are an image generation assistant. Use the image_generation tool to "
            "create exactly the image the user describes. Do not ask clarifying "
            "questions; generate the image directly."
        ),
        "input": payload_input,
        "tools": [{"type": "image_generation", "size": size}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {auth['token']}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "User-Agent": "codex_cli_rs",
    }
    if auth.get("account"):
        headers["chatgpt-account-id"] = auth["account"]
    req = urllib.request.Request(
        CODEX_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"request failed: {exc}"


def refresh_codex_token() -> None:
    """Force the Codex CLI to refresh ~/.codex/auth.json (cheap ephemeral run)."""
    try:
        subprocess.run(
            [CODEX_BIN, "exec", "--ephemeral", "--sandbox", "read-only",
             "--cd", str(ROOT), "--skip-git-repo-check", "-"],
            input="reply with ok", text=True, capture_output=True, timeout=90,
            env={**os.environ, "CODEX_HOME": str(CODEX_HOME),
                 "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
        )
    except Exception:
        pass


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        die(f"codex_media: invalid JSON on stdin: {exc}")

    params = payload.get("params") or {}
    prompt = as_text(params.get("prompt")) or "a high quality image"
    size = ASPECT_TO_SIZE.get(str(params.get("aspect_ratio") or "1:1"), "auto")
    if size not in VALID_SIZES:
        size = "auto"
    image_refs = [r for r in [params.get("image_url"), *(params.get("images_list") or [])] if r]

    auth = read_auth()
    if not auth:
        die(f"codex_media: no Codex auth at {CODEX_HOME}/auth.json (run `codex login`)")

    # The Codex backend only accepts its account chat models (e.g. gpt-5.5), NOT
    # gpt-image-*; the image_generation tool is what produces the picture.
    model = read_codex_model() or "gpt-5.5"
    payload_input = build_input(prompt, image_refs)

    status, raw = call_codex(auth, model, payload_input, size)
    if status in (401, 403):
        refresh_codex_token()
        auth = read_auth() or auth
        status, raw = call_codex(auth, model, payload_input, size)

    if status != 200:
        die(f"codex_media: Codex backend error {status}: {raw[:500]}")

    b64 = extract_image_b64(raw)
    if not b64:
        die("codex_media: Codex returned no image (content moderation may have "
            f"blocked the prompt, or image_generation is unavailable). {raw[:400]}")

    try:
        data = base64.b64decode(b64)
    except Exception as exc:
        die(f"codex_media: invalid base64 from Codex: {exc}")
    if not data:
        die("codex_media: empty image data from Codex")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"gen_{uuid.uuid4().hex}.png"
    (UPLOADS_DIR / filename).write_bytes(data)
    url = f"{PUBLIC_API_BASE_URL.rstrip('/')}/uploads/{filename}"
    print(json.dumps({"images": [{"url": url}]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
