#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "server"
SCHEMA = SERVER / "scripts" / "workflow_assistant_schema.json"
CODEX = "/opt/homebrew/bin/codex"
LOCAL_HOME = SERVER / ".assistant-home"
LOCAL_CODEX_HOME = LOCAL_HOME / ".codex"
SOURCE_CODEX_HOME = Path("/Users/dariushk/.codex")


def load_payload() -> dict:
    raw = sys.stdin.read()
    return json.loads(raw or "{}")


def load_workflow(workflow_id: str | None):
    if not workflow_id:
        return None
    path = SERVER / "data" / "workflows" / f"{workflow_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compact(text: str, limit: int = 1200) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def emit_error(message: str, returncode: int = 1) -> int:
    trimmed = compact(message)
    print(json.dumps({"backend": "codex", "error": trimmed, "message": trimmed, "suggestions": [], "workflow": None}))
    return returncode


def prepare_local_home() -> Path:
    LOCAL_CODEX_HOME.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml", "version.json"):
        src = SOURCE_CODEX_HOME / name
        dst = LOCAL_CODEX_HOME / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    for name in ("sessions", "archived_sessions", "cache", "rules", "tmp"):
        (LOCAL_CODEX_HOME / name).mkdir(parents=True, exist_ok=True)
    return LOCAL_HOME


def main() -> int:
    payload = load_payload()
    workflow = load_workflow(payload.get("workflow_id"))
    local_home = prepare_local_home()
    prompt = json.dumps(
        {
            "task": (
                "You are the Vibe Workflow assistant. Help the user design or update "
                "a node-based image/video/audio workflow. Return only JSON matching "
                "the provided schema. If you are not fully confident about a valid "
                "workflow patch, set workflow to null and explain the next step."
            ),
            "user_prompt": payload.get("prompt", ""),
            "conversation_history": payload.get("history", []),
            "current_workflow": workflow,
            "response_contract": {
                "message": "User-facing assistant answer.",
                "suggestions": ["Short follow-up actions the UI can show."],
                "workflow": "Full replacement workflow object or null."
            },
        }
    )

    completed = subprocess.run(
        [
            CODEX,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--cd",
            str(ROOT),
            "--output-schema",
            str(SCHEMA),
            "-",
        ],
        text=True,
        input=prompt,
        capture_output=True,
        timeout=3600,
        env={
            **os.environ,
            "HOME": str(local_home),
            "CODEX_HOME": str(LOCAL_CODEX_HOME),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        },
    )
    if completed.returncode != 0:
        message = compact(completed.stderr or completed.stdout or "Codex failed")
        return emit_error(f"Codex assistant failed: {message}")
    stdout = (completed.stdout or "").strip()
    if not stdout:
        return emit_error("Codex assistant returned no output")
    try:
        json.loads(stdout)
    except json.JSONDecodeError:
        return emit_error(f"Codex assistant returned non-JSON output: {stdout}")
    print(stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
