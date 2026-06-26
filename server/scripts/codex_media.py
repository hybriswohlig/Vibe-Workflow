#!/usr/bin/env python3
"""OPENAI_MEDIA_COMMAND backend that generates images via the Codex built-in
image_gen tool (ChatGPT subscription auth, no OPENAI_API_KEY required).

Contract (matches workflow_helper._call_command):
  stdin : {"model": "gpt-image-2", "params": {"prompt": "...", "image_url": "...",
                                              "images_list": [...], "aspect_ratio": "1:1"}}
  stdout: {"images": [{"url": "http://localhost:8000/uploads/<file>.png"}]}

On failure: write the reason to stderr and exit non-zero so the server surfaces it.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOADS_DIR = ROOT / "server" / "data" / "uploads"
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000")

CODEX_BIN = os.getenv("CODEX_BIN", "/opt/homebrew/bin/codex")
# Use the real Codex home by default so auth stays fresh (the isolated
# .assistant-home copy goes stale and its refresh token gets revoked).
CODEX_HOME = os.getenv("CODEX_MEDIA_HOME") or os.path.expanduser("~/.codex")
TIMEOUT = int(os.getenv("CODEX_MEDIA_TIMEOUT", "600"))

ASPECT_HINTS = {
    "1:1": "square (1024x1024)",
    "16:9": "wide landscape 16:9 (1536x1024)",
    "9:16": "tall portrait 9:16 (1024x1536)",
    "4:3": "landscape 4:3",
    "3:4": "portrait 3:4",
}


def die(message: str) -> None:
    sys.stderr.write(message.strip() + "\n")
    sys.exit(1)


def as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value or "")


def resolve_local_image(ref: str) -> Path | None:
    """Turn an image_url (http URL, /uploads/<f>, or local path) into a local file."""
    if not ref:
        return None
    ref = ref.strip()
    # Already a local uploads file?
    if ref.startswith(("http://", "https://")):
        tail = ref.split("/uploads/", 1)
        if len(tail) == 2:
            candidate = UPLOADS_DIR / tail[1].split("?", 1)[0]
            if candidate.exists():
                return candidate
        # Download remote image.
        try:
            dst = UPLOADS_DIR / f"ref_{uuid.uuid4().hex}"
            urllib.request.urlretrieve(ref, dst)  # noqa: S310 (trusted local workflow)
            return dst
        except Exception:
            return None
    path = Path(ref)
    if not path.is_absolute():
        path = UPLOADS_DIR / path.name
    return path if path.exists() else None


def build_prompt(model: str, params: dict, target: Path, inputs: list[Path]) -> str:
    prompt = as_text(params.get("prompt")) or "a high quality image"
    aspect = ASPECT_HINTS.get(str(params.get("aspect_ratio") or "1:1"), "square (1024x1024)")
    lines = [
        "You MUST use your built-in image_gen tool. Do not write code and do not "
        "call any HTTP API.",
        f"Generate one {aspect} image for this request:",
        "",
        prompt,
        "",
    ]
    if inputs:
        lines.append(
            "Use the attached image(s) as the visual reference / source to edit."
        )
    lines += [
        f"Save the final PNG to exactly this absolute path: {target}",
        "Generate the image exactly once. Do not search the filesystem, do not "
        "run verification commands, and do not regenerate. As soon as the PNG is "
        "saved to the path, stop.",
        f"On the final line, print only this path: {target}",
    ]
    return "\n".join(lines)


def newest_generated(since: float) -> Path | None:
    gen_dir = Path(CODEX_HOME) / "generated_images"
    if not gen_dir.exists():
        return None
    candidates = [
        p for p in gen_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".webp", ".jpg", ".jpeg"}
        and p.stat().st_mtime >= since - 1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        die(f"codex_media: invalid JSON on stdin: {exc}")

    model = payload.get("model", "gpt-image-2")
    params = payload.get("params") or {}

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"gen_{uuid.uuid4().hex}.png"
    target = UPLOADS_DIR / filename

    inputs: list[Path] = []
    for ref in [params.get("image_url"), *(params.get("images_list") or [])]:
        local = resolve_local_image(as_text(ref)) if ref else None
        if local:
            inputs.append(local)

    prompt = build_prompt(model, params, target, inputs)

    if not Path(CODEX_BIN).exists():
        die(f"codex_media: codex binary not found at {CODEX_BIN}")
    if not (Path(CODEX_HOME) / "auth.json").exists():
        die(f"codex_media: no Codex auth at {CODEX_HOME}/auth.json (run `codex login`)")

    cmd = [
        CODEX_BIN, "exec",
        "--ephemeral",
        "--sandbox", "workspace-write",
        "--cd", str(ROOT),
        "--skip-git-repo-check",
    ]
    for img in inputs:
        cmd += ["--image", str(img)]
    cmd.append("-")

    env = {
        **os.environ,
        "CODEX_HOME": CODEX_HOME,
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }

    start = _now_mtime(target)
    try:
        completed = subprocess.run(
            cmd, input=prompt, text=True, capture_output=True,
            timeout=TIMEOUT, env=env,
        )
    except subprocess.TimeoutExpired:
        die(f"codex_media: codex image generation timed out after {TIMEOUT}s")

    produced = target if target.exists() else None
    if produced is None:
        fallback = newest_generated(start)
        if fallback is not None:
            shutil.copy2(fallback, target)
            produced = target

    if produced is None or not produced.exists() or produced.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "no output").strip()
        # Drop noisy auth-refresh lines unless they are the actual failure.
        die(f"codex_media: image was not produced.\n{detail[-1200:]}")

    url = f"{PUBLIC_API_BASE_URL.rstrip('/')}/uploads/{filename}"
    print(json.dumps({"images": [{"url": url}]}))
    return 0


def _now_mtime(target: Path) -> float:
    # Use the most recent mtime in generated_images as the "start" reference so a
    # brand-new file is reliably detected even if wall-clock skews.
    try:
        return os.path.getmtime(target)
    except OSError:
        import time
        return time.time()


if __name__ == "__main__":
    raise SystemExit(main())
