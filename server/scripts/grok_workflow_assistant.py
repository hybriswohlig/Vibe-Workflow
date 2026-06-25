#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "server"
GROK = "/Users/dariushk/.grok/bin/grok"
GROK_MODEL = os.getenv("GROK_ASSISTANT_MODEL", "grok-build")
LOCAL_HOME = SERVER / ".assistant-home"
LOCAL_GROK_HOME = LOCAL_HOME / ".grok"
SOURCE_GROK_HOME = Path("/Users/dariushk/.grok")


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


def prepare_local_home() -> Path:
    LOCAL_GROK_HOME.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml", "version.json", "active_sessions.json"):
        src = SOURCE_GROK_HOME / name
        dst = LOCAL_GROK_HOME / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    for name in ("sessions", "logs", "memory", "worktrees", "upload_queue", "marketplace-cache", "downloads"):
        (LOCAL_GROK_HOME / name).mkdir(parents=True, exist_ok=True)
    return LOCAL_HOME


def parse_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            return {"message": text.strip(), "suggestions": [], "workflow": None}

    if isinstance(parsed, dict):
        if isinstance(parsed.get("text"), str):
            inner = parsed["text"].strip()
            try:
                inner_parsed = json.loads(inner)
                if isinstance(inner_parsed, dict):
                    return inner_parsed
            except json.JSONDecodeError:
                return {"message": inner, "suggestions": [], "workflow": None}
        if "message" in parsed or "suggestions" in parsed or "workflow" in parsed:
            return {
                "message": parsed.get("message", ""),
                "suggestions": parsed.get("suggestions", []),
                "workflow": parsed.get("workflow"),
                "backend": "grok",
            }
        if isinstance(parsed.get("text"), str):
            return {"message": parsed["text"].strip(), "suggestions": [], "workflow": None}

    return {"message": text.strip(), "suggestions": [], "workflow": None}


def emit_error(message: str, returncode: int = 1) -> int:
    trimmed = compact(message)
    print(json.dumps({"backend": "grok", "error": trimmed, "message": trimmed, "suggestions": [], "workflow": None}))
    return returncode


def main() -> int:
    payload = load_payload()
    workflow = load_workflow(payload.get("workflow_id"))
    local_home = prepare_local_home()
    prompt = json.dumps(
        {
            "task": (
                "You are the Vibe Workflow assistant. Help the user design or update "
                "a node-based image/video/audio workflow. Reply with valid JSON only: "
                "{\"message\": string, \"suggestions\": string[], \"workflow\": object|null}. "
                "If you are not fully confident about a valid workflow patch, set workflow to null."
            ),
            "user_prompt": payload.get("prompt", ""),
            "conversation_history": payload.get("history", []),
            "current_workflow": workflow,
        }
    )

    completed = subprocess.run(
        [
            GROK,
            "-m",
            GROK_MODEL,
            "--cwd",
            str(ROOT),
            "--single",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "dontAsk",
            "--max-turns",
            "4",
            "--no-plan",
            "--no-subagents",
            "--no-memory",
            "--no-alt-screen",
            "--no-leader",
            "--leader-socket",
            str((SERVER / ".assistant-home" / "grok-leader.sock").resolve()),
        ],
        text=True,
        capture_output=True,
        timeout=300,
        env={
            **os.environ,
            "HOME": str(local_home),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        },
    )
    if completed.returncode != 0:
        message = compact(completed.stderr or completed.stdout or "Grok failed")
        return emit_error(f"Grok assistant failed: {message}")
    result = parse_json(completed.stdout)
    result.setdefault("message", "")
    result.setdefault("suggestions", [])
    result.setdefault("workflow", None)
    result["backend"] = "grok"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
