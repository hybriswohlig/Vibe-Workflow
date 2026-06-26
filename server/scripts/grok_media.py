#!/usr/bin/env python3
"""GROK_MEDIA_COMMAND backend: Grok Imagine image/video via the xAI API using
the local Grok Build session (~/.grok/auth.json) — your SuperGrok subscription,
no separate XAI_API_KEY required (falls back to XAI_API_KEY if set).

Ported from Open-Generative-AI (mcp/server.mjs grokImage/grokVideo + grokAuth.js).

Contract (matches workflow_helper._call_command):
  stdin : {"model": "grok-imagine-image-quality", "params": {"prompt","image_url","images_list","aspect_ratio"}}
  stdout (image): {"images": [{"url": "http://localhost:8000/uploads/<file>.png"}]}
  stdout (video): {"video": {"url": "http://localhost:8000/uploads/<file>.mp4"}}
On failure: write the reason to stderr and exit non-zero so the server surfaces it.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOADS_DIR = ROOT / "server" / "data" / "uploads"
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000")
GROK_HOME = Path(os.getenv("GROK_MEDIA_HOME") or os.path.expanduser("~/.grok"))
GROK_BIN = os.getenv("GROK_BIN") or str(GROK_HOME / "bin" / "grok")
XAI_BASE_URL = "https://api.x.ai/v1"


class AuthExpired(Exception):
    """Raised when xAI rejects the token (401/403) so the caller can refresh + retry."""
TIMEOUT = int(os.getenv("GROK_MEDIA_TIMEOUT", "120"))
VIDEO_POLL_ATTEMPTS = int(os.getenv("GROK_VIDEO_POLL_ATTEMPTS", "180"))
VIDEO_POLL_INTERVAL = int(os.getenv("GROK_VIDEO_POLL_INTERVAL", "5"))


def die(message: str) -> None:
    sys.stderr.write(message.strip() + "\n")
    sys.exit(1)


def as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value or "")


def read_grok_token() -> str | None:
    # Explicit key wins (paid xAI key).
    if os.getenv("XAI_API_KEY"):
        return os.getenv("XAI_API_KEY")
    path = GROK_HOME / "auth.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Legacy shapes.
    if data.get("access_token") or data.get("token"):
        return data.get("access_token") or data.get("token")
    # Current shape: session objects keyed by "<issuer>::<sessionId>" holding the JWT in `.key`.
    sessions = [v for v in data.values() if isinstance(v, dict) and isinstance(v.get("key"), str)]
    if not sessions:
        return None
    now = time.time()

    def not_expired(s: dict) -> bool:
        ea = s.get("expires_at")
        if not ea:
            return True
        try:
            return datetime.fromisoformat(ea.replace("Z", "+00:00")).timestamp() > now
        except Exception:
            return True

    chosen = next((s for s in sessions if not_expired(s)), sessions[0])
    return chosen.get("key")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _post(path: str, body: dict, token: str, timeout: int) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        f"{XAI_BASE_URL}{path}", data=json.dumps(body).encode("utf-8"),
        headers=_headers(token), method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"request failed: {exc}"


def _get(path: str, token: str, timeout: int) -> tuple[int, dict | str]:
    req = urllib.request.Request(f"{XAI_BASE_URL}{path}", headers=_headers(token), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"request failed: {exc}"


def resolve_image_ref(ref: str) -> str | None:
    """xAI accepts public URLs or base64 data URIs. Convert local files/urls to data URIs."""
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
            return ref  # assume publicly reachable
    else:
        cand = Path(ref) if Path(ref).is_absolute() else UPLOADS_DIR / Path(ref).name
        if cand.exists():
            local = cand
    if local is not None:
        ext = local.suffix.lstrip(".") or "png"
        return f"data:image/{ext};base64," + base64.b64encode(local.read_bytes()).decode()
    return None


_DOWNLOAD_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def save_from_url(url: str, suffix: str) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"gen_{uuid.uuid4().hex}.{suffix}"
    req = urllib.request.Request(url, headers={"User-Agent": _DOWNLOAD_UA, "Accept": "*/*"})
    try:
        data = urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError as exc:
        die(f"grok_media: failed to download result ({exc.code}) from {url[:80]}")
    (UPLOADS_DIR / filename).write_bytes(data)
    return f"{PUBLIC_API_BASE_URL.rstrip('/')}/uploads/{filename}"


def save_b64(b64: str, suffix: str) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"gen_{uuid.uuid4().hex}.{suffix}"
    (UPLOADS_DIR / filename).write_bytes(base64.b64decode(b64))
    return f"{PUBLIC_API_BASE_URL.rstrip('/')}/uploads/{filename}"


def do_image(model: str, params: dict, token: str) -> dict:
    prompt = as_text(params.get("prompt")) or "a high quality image"
    refs = [r for r in [params.get("image_url"), *(params.get("images_list") or [])] if r]
    body = {"model": model, "prompt": prompt}
    if params.get("aspect_ratio"):
        body["aspect_ratio"] = params["aspect_ratio"]
    if refs:  # edit / reference path
        image = resolve_image_ref(refs[0])
        if not image:
            die("grok_media: reference image could not be resolved for edit")
        body["image"] = {"url": image, "type": "image_url"}
        status, data = _post("/images/edits", body, token, TIMEOUT)
    else:
        status, data = _post("/images/generations", body, token, TIMEOUT)
    if status in (401, 403):
        raise AuthExpired(str(data)[:300])
    if status != 200:
        die(f"grok_media: xAI image error {status}: {str(data)[:400]} "
            "(set XAI_API_KEY or use fal if api.x.ai keeps rejecting the token)")
    entry = (data.get("data") or [{}])[0] if isinstance(data, dict) else {}
    suffix = "jpg" if "jpeg" in str(entry.get("mime_type") or "").lower() else "png"
    if entry.get("url"):
        return {"images": [{"url": save_from_url(entry["url"], suffix)}]}
    if entry.get("b64_json"):
        return {"images": [{"url": save_b64(entry["b64_json"], suffix)}]}
    die(f"grok_media: xAI returned no image: {str(data)[:300]}")


def do_video(model: str, params: dict, token: str) -> dict:
    prompt = as_text(params.get("prompt"))
    body = {"model": model, "prompt": prompt}
    if params.get("aspect_ratio"):
        body["aspect_ratio"] = params["aspect_ratio"]
    refs = [r for r in [params.get("image_url"), *(params.get("images_list") or [])] if r]
    if refs:  # image-to-video
        image = resolve_image_ref(refs[0])
        if image:
            body["image"] = {"url": image, "type": "image_url"}
    status, data = _post("/videos/generations", body, token, TIMEOUT)
    if status in (401, 403):
        raise AuthExpired(str(data)[:300])
    if status != 200:
        die(f"grok_media: xAI video error {status}: {str(data)[:400]}")
    if not isinstance(data, dict):
        die(f"grok_media: unexpected xAI video response: {str(data)[:300]}")
    url = (data.get("video") or {}).get("url") or data.get("url")
    request_id = data.get("request_id")
    if not url and request_id:
        for _ in range(VIDEO_POLL_ATTEMPTS):
            time.sleep(VIDEO_POLL_INTERVAL)
            st, pd = _get(f"/videos/{request_id}", token, 30)
            if st != 200 or not isinstance(pd, dict):
                continue
            status_str = str(pd.get("status") or "").lower()
            if status_str == "done":
                url = (pd.get("video") or {}).get("url") or pd.get("url")
                break
            if status_str in ("failed", "expired"):
                die(f"grok_media: xAI video {status_str}")
    if not url:
        die("grok_media: xAI video produced no URL (timed out)")
    return {"video": {"url": save_from_url(url, "mp4")}}


def refresh_grok_token() -> None:
    """Trigger the Grok CLI to refresh ~/.grok auth via a trivial agent call.
    `grok models` does NOT refresh; a real (single, 1-turn) invocation does."""
    if not Path(GROK_BIN).exists():
        return
    sock = Path(os.getenv("TMPDIR", "/tmp")) / "grok-media-refresh.sock"
    try:
        subprocess.run(
            [GROK_BIN, "-m", "grok-build", "--single", "ok",
             "--output-format", "plain", "--permission-mode", "dontAsk",
             "--max-turns", "1", "--no-plan", "--no-subagents", "--no-memory",
             "--no-alt-screen", "--no-leader", "--leader-socket", str(sock)],
            capture_output=True, text=True, timeout=90,
            env={**os.environ, "HOME": str(GROK_HOME.parent),
                 "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
        )
    except Exception:
        pass


def _generate(model: str, params: dict, token: str) -> dict:
    if "video" in model:
        return do_video(model, params, token)
    return do_image(model, params, token)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        die(f"grok_media: invalid JSON on stdin: {exc}")

    model = payload.get("model") or "grok-imagine-image-quality"
    params = payload.get("params") or {}

    token = read_grok_token()
    if not token:
        die(f"grok_media: no Grok session at {GROK_HOME}/auth.json (run `grok login`) "
            "and no XAI_API_KEY set")

    try:
        result = _generate(model, params, token)
    except AuthExpired:
        # Token expired — refresh via the Grok CLI and retry once.
        refresh_grok_token()
        token = read_grok_token() or token
        try:
            result = _generate(model, params, token)
        except AuthExpired as exc:
            die("grok_media: Grok auth expired and auto-refresh failed — run `grok login`. "
                f"({str(exc)[:200]})")

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
