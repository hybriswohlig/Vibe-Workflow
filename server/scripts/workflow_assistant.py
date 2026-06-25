#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "server" / "scripts"
CODEX_WRAPPER = SCRIPTS / "codex_workflow_assistant.py"
GROK_WRAPPER = SCRIPTS / "grok_workflow_assistant.py"
BACKEND_MAP = {
    "codex": CODEX_WRAPPER,
    "grok": GROK_WRAPPER,
}


def compact(text: str, limit: int = 1000) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def load_payload() -> str:
    return sys.stdin.read() or "{}"


def configured_backends() -> list[Path]:
    raw = os.getenv("WORKFLOW_ASSISTANT_BACKENDS", "codex").strip()
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    backends = []
    for name in requested:
        wrapper = BACKEND_MAP.get(name)
        if wrapper and wrapper not in backends:
            backends.append(wrapper)
    return backends or [CODEX_WRAPPER]


def run_wrapper(wrapper: Path, payload: str) -> dict | None:
    try:
        completed = subprocess.run(
            [str(wrapper)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"backend": wrapper.stem.replace("_workflow_assistant", ""), "error": f"{wrapper.name} timed out"}

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        try:
            parsed = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            parsed = {}
        return {
            "backend": wrapper.stem.replace("_workflow_assistant", ""),
            "error": parsed.get("error")
            or parsed.get("message")
            or f"{wrapper.name} exited {completed.returncode}",
            "message": parsed.get("message") or stdout or stderr,
            "stdout": compact(stdout),
            "stderr": compact(stderr),
            "returncode": completed.returncode,
        }
    try:
        data = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {
            "backend": wrapper.stem.replace("_workflow_assistant", ""),
            "error": f"{wrapper.name} returned non-JSON output",
            "message": stdout or stderr,
            "stdout": compact(stdout),
            "stderr": compact(stderr),
            "returncode": completed.returncode,
        }
    data["backend"] = wrapper.stem.replace("_workflow_assistant", "")
    return data


def main() -> int:
    payload = load_payload()
    errors = []

    for wrapper in configured_backends():
        result = run_wrapper(wrapper, payload)
        if result and not result.get("error"):
            print(json.dumps(result))
            return 0
        if result and result.get("error"):
            errors.append(result)

    failure_message = "Workflow assistant failed to run Codex and Grok backends."
    backend_names = [error.get("backend", "backend") for error in errors]
    if backend_names:
        failure_message = f"Workflow assistant failed to run {', '.join(backend_names)} backend(s)."
    if errors:
        summary = []
        for error in errors:
            backend = error.get("backend", "backend")
            detail = compact(error.get("error") or error.get("message") or "unknown error")
            summary.append(f"{backend}: {detail}")
        failure_message = f"{failure_message} " + " | ".join(summary)
    print(
        json.dumps(
            {
                "message": failure_message,
                "details": errors,
                "suggestions": [],
                "workflow": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
