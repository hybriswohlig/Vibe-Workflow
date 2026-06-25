#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "server"
SCHEMA = SERVER / "scripts" / "workflow_assistant_schema.json"
CODEX_BIN = os.getenv("CODEX_BIN", "/opt/homebrew/bin/codex")
CODEX_MODEL = os.getenv("CODEX_BRIDGE_MODEL", "gpt-5.5")
CODEX_SANDBOX = os.getenv("CODEX_BRIDGE_SANDBOX", "read-only")


def load_workflow(workflow_id: str | None):
    if not workflow_id:
        return None
    path = SERVER / "data" / "workflows" / f"{workflow_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(payload: dict) -> str:
    return json.dumps(
        {
            "task": (
                "You are the Vibe Workflow assistant. Help the user design or update "
                "a node-based image/video/audio workflow. Return only JSON matching "
                "the provided schema. If you are not fully confident about a valid "
                "workflow patch, set workflow to null and explain the next step."
            ),
            "user_prompt": payload.get("prompt", ""),
            "conversation_history": payload.get("history", []),
            "current_workflow": load_workflow(payload.get("workflow_id")),
            "response_contract": {
                "message": "User-facing assistant answer.",
                "suggestions": ["Short follow-up actions the UI can show."],
                "workflow": "Full replacement workflow object or null.",
            },
        }
    )


def run_codex(payload: dict) -> dict:
    prompt = build_prompt(payload)
    with tempfile.NamedTemporaryFile(prefix="codex-last-", suffix=".json", delete=False) as handle:
        last_message_path = Path(handle.name)

    completed = subprocess.run(
        [
            CODEX_BIN,
            "exec",
            "--model",
            CODEX_MODEL,
            "--ephemeral",
            "--sandbox",
            CODEX_SANDBOX,
            "--cd",
            str(ROOT),
            "--output-schema",
            str(SCHEMA),
            "--output-last-message",
            str(last_message_path),
            "-",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=3600,
        env={**os.environ, "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
    )

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Codex failed").strip()
        if len(detail) > 1600:
            detail = detail[:1597].rstrip() + "..."
        return {
            "backend": "codex",
            "error": f"Codex assistant failed: {detail}",
            "message": f"Codex assistant failed: {detail}",
            "suggestions": [],
            "workflow": None,
        }

    result_text = ""
    if last_message_path.exists():
        result_text = last_message_path.read_text(encoding="utf-8").strip()
    if not result_text:
        result_text = (completed.stdout or "").strip()
    if not result_text:
        return {
            "backend": "codex",
            "error": "Codex assistant returned no output",
            "message": "Codex assistant returned no output",
            "suggestions": [],
            "workflow": None,
        }

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        return {
            "backend": "codex",
            "error": "Codex assistant returned non-JSON output",
            "message": result_text,
            "suggestions": [],
            "workflow": None,
        }

    result["backend"] = "codex"
    return result


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/workflow-assistant":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        try:
            result = run_codex(payload)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"backend": "codex", "error": str(exc), "message": str(exc), "suggestions": [], "workflow": None})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Codex bridge listening on http://{args.host}:{args.port}/workflow-assistant")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
