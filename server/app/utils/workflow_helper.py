import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
WORKFLOWS_DIR = DATA_DIR / "workflows"
RUNS_DIR = DATA_DIR / "runs"
UPLOADS_DIR = DATA_DIR / "uploads"

FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000")
OPENAI_MEDIA_COMMAND = os.getenv("OPENAI_MEDIA_COMMAND") or os.getenv("OPENAI_IMAGE_COMMAND")
GROK_MEDIA_COMMAND = os.getenv("GROK_MEDIA_COMMAND")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    for path in (WORKFLOWS_DIR, RUNS_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return deepcopy(default)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data) -> None:
    _ensure_dirs()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    tmp_path.replace(path)


def _workflow_path(workflow_id: str) -> Path:
    return WORKFLOWS_DIR / f"{workflow_id}.json"


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _load_workflow(workflow_id: str) -> dict:
    workflow = _read_json(_workflow_path(workflow_id), None)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


def _load_run(run_id: str) -> dict:
    run = _read_json(_run_path(run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _schema_property(
    title: str,
    field_type: str = "string",
    default=None,
    field: Optional[str] = None,
    enum: Optional[list[str]] = None,
    max_items: Optional[int] = None,
) -> dict:
    prop = {
        "title": title,
        "name": title.lower().replace(" ", "_"),
        "type": field_type,
        "description": title,
    }
    if default is not None:
        prop["default"] = default
    if field:
        prop["field"] = field
    if enum:
        prop["enum"] = enum
    if field_type == "array":
        prop["items"] = {"type": "string"}
        prop["maxItems"] = max_items or 10
    return prop


PROMPT_SCHEMA = {
    "prompt": _schema_property("Prompt", "string", ""),
}
IMAGE_INPUTS = {
    **PROMPT_SCHEMA,
    "image_url": _schema_property("Image URL", "string", "", "image"),
    "images_list": _schema_property("Images", "array", [], "images_list", max_items=8),
    "aspect_ratio": _schema_property(
        "Aspect Ratio",
        "string",
        "1:1",
        enum=["1:1", "16:9", "9:16", "4:3", "3:4"],
    ),
}
VIDEO_INPUTS = {
    **PROMPT_SCHEMA,
    "image_url": _schema_property("Image URL", "string", "", "image"),
    "last_image": _schema_property("Last Image", "string", "", "image"),
    "video_url": _schema_property("Video URL", "string", "", "video"),
    "audio_url": _schema_property("Audio URL", "string", "", "audio"),
    "images_list": _schema_property("Images", "array", [], "images_list", max_items=8),
    "videos_list": _schema_property("Videos", "array", [], "videos_list", max_items=8),
    "audios_list": _schema_property("Audios", "array", [], "audios_list", max_items=8),
    "aspect_ratio": _schema_property(
        "Aspect Ratio",
        "string",
        "16:9",
        enum=["16:9", "9:16", "1:1", "4:3", "3:4"],
    ),
}
TEXT_INPUTS = {
    **PROMPT_SCHEMA,
    "system_prompt": _schema_property("System Prompt", "string", ""),
    "image_url": _schema_property("Image URL", "string", "", "image"),
    "images_list": _schema_property("Images", "array", [], "images_list", max_items=8),
}
AUDIO_INPUTS = {
    **PROMPT_SCHEMA,
    "audio_url": _schema_property("Audio URL", "string", "", "audio"),
    "video_url": _schema_property("Video URL", "string", "", "video"),
    "image_url": _schema_property("Image URL", "string", "", "image"),
}
VIDEO_COMBINER_INPUTS = {
    "videos_list": _schema_property("Video Clips", "array", [], "videos_list", max_items=20),
    "aspect_ratio": _schema_property(
        "Aspect Ratio",
        "string",
        "auto",
        enum=["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"],
    ),
}

IMAGE_MODELS = [
    "image-passthrough",
    "gpt-image-1.5",
    "nano-banana",
    "nano-banana-edit",
    "nano-banana-pro",
    "nano-banana-pro-edit",
    "flux-schnell",
    "flux-2-dev",
    "flux-2-dev-edit",
    "flux-2-flex",
    "flux-2-flex-edit",
    "flux-2-pro",
    "flux-2-pro-edit",
    "bytedance-seedream-v4",
    "bytedance-seedream-edit-v4",
    "bytedance-seedream-v4.5",
    "bytedance-seedream-v4.5-edit",
    "wan2.5-text-to-image",
    "wan2.5-image-edit",
    "wan2.6-text-to-image",
    "wan2.6-image-edit",
    "qwen-image",
    "qwen-image-edit-2511",
    "qwen-image-edit",
    "qwen-image-edit-plus",
    "qwen-image-edit-plus-lora",
    "z-image-turbo",
    "chroma-image",
    "kling-o1-text-to-image",
    "kling-o1-edit-image",
    "grok-imagine-text-to-image",
    "hunyuan-image-2.1",
    "hunyuan-image-3.0",
    "google-imagen4",
    "google-imagen4-fast",
    "google-imagen4-ultra",
    "midjourney-v7-text-to-image",
    "midjourney-v7-image-to-image",
    "midjourney-v7-omni-reference",
    "midjourney-v7-style-reference",
    "vidu-q2-text-to-image",
    "vidu-q2-reference-to-image",
]
VIDEO_MODELS = [
    "video-passthrough",
    "seedance-lite-i2v",
    "seedance-lite-t2v",
    "seedance-pro-t2v",
    "seedance-pro-i2v",
    "seedance-v1.5-pro-i2v",
    "seedance-v1.5-pro-t2v",
    "veo3.1-image-to-video",
    "veo3.1-text-to-video",
    "wan2.2-text-to-video",
    "wan2.2-image-to-video",
    "wan2.5-text-to-video",
    "wan2.5-image-to-video",
    "wan2.6-text-to-video",
    "wan2.6-image-to-video",
    "openai-sora",
    "openai-sora-2-text-to-video",
    "openai-sora-2-image-to-video",
    "openai-sora-2-pro-text-to-video",
    "openai-sora-2-pro-image-to-video",
    "kling-o1-text-to-video",
    "kling-o1-image-to-video",
    "grok-imagine-text-to-video",
    "grok-imagine-image-to-video",
    "hunyuan-text-to-video",
    "hunyuan-image-to-video",
    "midjourney-v7-image-to-video",
    "vidu-q2-reference",
    "luma-modify-video",
    "luma-flash-reframe",
]
TEXT_MODELS = ["text-passthrough", "any-llm", "openrouter-vision", "gpt-5-nano", "gpt-5-mini"]
AUDIO_MODELS = [
    "audio-passthrough",
    "suno-create-music",
    "suno-extend-music",
    "suno-remix-music",
    "minimax-voice-clone",
    "minimax-speech-2.6-hd",
    "minimax-speech-2.6-turbo",
]

FAL_MODEL_MAP = {
    "flux-schnell": "fal-ai/flux/schnell",
    "flux-2-dev": "fal-ai/flux-2/dev",
    "flux-2-dev-edit": "fal-ai/flux-2/dev/image-to-image",
    "flux-2-flex": "fal-ai/flux-2/flex",
    "flux-2-flex-edit": "fal-ai/flux-2/flex/image-to-image",
    "flux-2-pro": "fal-ai/flux-2/pro",
    "flux-2-pro-edit": "fal-ai/flux-2/pro/image-to-image",
    "nano-banana": "fal-ai/nano-banana",
    "nano-banana-edit": "fal-ai/nano-banana/edit",
    "nano-banana-pro": "fal-ai/nano-banana-pro",
    "nano-banana-pro-edit": "fal-ai/nano-banana-pro/edit",
    "veo3.1-text-to-video": "fal-ai/veo3.1/text-to-video",
    "veo3.1-image-to-video": "fal-ai/veo3.1/image-to-video",
}


def _model_name(model_id: str) -> str:
    specials = {
        "text-passthrough": "Input Text",
        "image-passthrough": "Input Image",
        "video-passthrough": "Input Video",
        "audio-passthrough": "Input Audio",
    }
    if model_id in specials:
        return specials[model_id]
    return " ".join(word.capitalize() for word in model_id.replace("-", " ").split())


def _schema_for(properties: dict, required: Optional[list[str]] = None) -> dict:
    return {
        "input_schema": {
            "schemas": {
                "input_data": {
                    "properties": deepcopy(properties),
                    "required": required or ["prompt"],
                }
            }
        }
    }


def _model_dict(model_ids: list[str], properties: dict) -> dict:
    return {
        model_id: {
            "id": model_id,
            "name": _model_name(model_id),
            **_schema_for(properties if "passthrough" not in model_id else _passthrough_schema(model_id)),
        }
        for model_id in model_ids
    }


def _passthrough_schema(model_id: str) -> dict:
    if model_id.startswith("image"):
        return {"image_url": _schema_property("Image URL", "string", "", "image")}
    if model_id.startswith("video"):
        return {"video_url": _schema_property("Video URL", "string", "", "video")}
    if model_id.startswith("audio"):
        return {"audio_url": _schema_property("Audio URL", "string", "", "audio")}
    return {"prompt": _schema_property("Prompt", "string", "")}


def _node_schemas() -> dict:
    return {
        "categories": {
            "text": {"models": _model_dict(TEXT_MODELS, TEXT_INPUTS)},
            "image": {"models": _model_dict(IMAGE_MODELS, IMAGE_INPUTS)},
            "video": {"models": _model_dict(VIDEO_MODELS, VIDEO_INPUTS)},
            "audio": {"models": _model_dict(AUDIO_MODELS, AUDIO_INPUTS)},
            "utility": {
                "models": {
                    "prompt-concatenator": {
                        "id": "prompt-concatenator",
                        "name": "Prompt Concatenator",
                        **_schema_for(PROMPT_SCHEMA),
                    },
                    "video-combiner": {
                        "id": "video-combiner",
                        "name": "Video Combiner",
                        **_schema_for(VIDEO_COMBINER_INPUTS, ["videos_list"]),
                    },
                }
            },
            "api": {"models": {}},
        }
    }


def _resolve_templates(value, node_results: dict):
    if isinstance(value, list):
        return [_resolve_templates(item, node_results) for item in value]
    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"\{\{\s*([^.]+)\.outputs\[0\]\.value\s*\}\}", value)
    if match:
        source_id = match.group(1)
        return node_results.get(source_id, "")
    return value


def _result_from_output(value, output_type: str) -> dict:
    result_id = str(uuid.uuid4())
    return {
        "id": result_id,
        "outputs": [{"type": output_type, "value": value}],
    }


def _failure_result(message: str) -> dict:
    return _result_from_output({"error": message}, "error")


def _run_record(node_id: str, status: str, result: dict, run_id: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "node_run_id": str(uuid.uuid4()),
        "node_id": node_id,
        "run_id": run_id,
        "status": status,
        "started_at": _now(),
        "completed_at": _now() if status in {"succeeded", "failed"} else None,
        "result": result,
    }


def _normalize_fal_result(payload: dict, category: str) -> dict:
    if "images" in payload and payload["images"]:
        outputs = [
            {"type": "image_url", "value": image.get("url", image)}
            for image in payload["images"]
        ]
    elif "image" in payload:
        image = payload["image"]
        outputs = [{"type": "image_url", "value": image.get("url", image)}]
    elif "video" in payload:
        video = payload["video"]
        outputs = [{"type": "video_url", "value": video.get("url", video)}]
    elif "audio" in payload:
        audio = payload["audio"]
        outputs = [{"type": "audio_url", "value": audio.get("url", audio)}]
    elif "audio_url" in payload:
        outputs = [{"type": "audio_url", "value": payload["audio_url"]}]
    elif "output" in payload:
        output_type = "text" if category == "text" else f"{category}_url"
        outputs = [{"type": output_type, "value": payload["output"]}]
    else:
        outputs = [{"type": "text", "value": json.dumps(payload)}]
    return {"id": str(uuid.uuid4()), "outputs": outputs}


async def _call_fal(model: str, params: dict, category: str) -> dict:
    if not FAL_KEY:
        raise HTTPException(status_code=400, detail="Set FAL_KEY or FAL_API_KEY in server/.env")

    endpoint = FAL_MODEL_MAP.get(model, f"fal-ai/{model}")
    url = f"https://queue.fal.run/{endpoint}"
    headers = {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        submit = await client.post(url, headers=headers, json=params)
        if submit.status_code >= 400:
            raise HTTPException(status_code=submit.status_code, detail=submit.text)
        submitted = submit.json()
        status_url = submitted.get("status_url")
        response_url = submitted.get("response_url")

        for _ in range(240):
            status = await client.get(status_url, headers=headers)
            status.raise_for_status()
            status_data = status.json()
            if status_data.get("status") == "COMPLETED":
                if status_data.get("error"):
                    raise HTTPException(status_code=502, detail=status_data["error"])
                response = await client.get(response_url, headers=headers)
                response.raise_for_status()
                return _normalize_fal_result(response.json(), category)
            await asyncio.sleep(2)

    raise HTTPException(status_code=504, detail="fal.ai request timed out")


async def _call_command(command: str, payload: dict, category: str) -> dict:
    try:
        completed = subprocess.run(
            shlex.split(command),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=3600,
            check=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Fallback command not found: {command}") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=502, detail=exc.stderr or exc.stdout) from exc

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        data = {"output": completed.stdout.strip()}
    return _normalize_fal_result(data, category)


async def _execute_node(node: dict, run_id: str, node_results: dict) -> dict:
    category = node.get("category")
    model = node.get("model", "")
    params = {
        key: _resolve_templates(value, node_results)
        for key, value in (node.get("params") or node.get("input_params") or {}).items()
    }

    try:
        if model.endswith("passthrough"):
            value = (
                params.get("prompt")
                or params.get("image_url")
                or params.get("video_url")
                or params.get("audio_url")
                or ""
            )
            output_type = "text" if category == "text" else f"{category}_url"
            result = _result_from_output(value, output_type)
        elif category == "utility" and model == "prompt-concatenator":
            prompt = params.get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(str(item) for item in prompt)
            result = _result_from_output(prompt, "text")
        elif category == "utility" and model == "video-combiner":
            videos = params.get("videos_list") or params.get("video_files") or []
            result = _result_from_output(videos[0] if videos else "", "video_url")
        elif model.startswith(("gpt-image", "openai-sora")):
            if not OPENAI_MEDIA_COMMAND:
                raise HTTPException(
                    status_code=400,
                    detail="Set OPENAI_MEDIA_COMMAND to use OpenAI subscription fallback",
                )
            result = await _call_command(OPENAI_MEDIA_COMMAND, {"model": model, "params": params}, category)
        elif model.startswith("grok-"):
            if not GROK_MEDIA_COMMAND:
                raise HTTPException(
                    status_code=400,
                    detail="Set GROK_MEDIA_COMMAND to use Grok subscription fallback",
                )
            result = await _call_command(GROK_MEDIA_COMMAND, {"model": model, "params": params}, category)
        else:
            result = await _call_fal(model, params, category)
        node_results[node["id"]] = result["outputs"][0].get("value")
        return _run_record(node["id"], "succeeded", result, run_id)
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        return _run_record(node["id"], "failed", _failure_result(detail), run_id)


async def create_or_update_workflow(payload: dict):
    _ensure_dirs()
    workflow_id = payload.get("workflow_id") or str(uuid.uuid4())
    workflow = {
        **payload,
        "workflow_id": workflow_id,
        "id": workflow_id,
        "name": payload.get("name") or "Untitled",
        "is_owner": True,
        "is_published": payload.get("is_published", False),
        "is_template": payload.get("is_template", False),
        "show_temp_button": True,
        "updated_at": _now(),
        "created_at": payload.get("created_at") or _now(),
    }
    _write_json(_workflow_path(workflow_id), workflow)
    return {"workflow_id": workflow_id, **workflow}


async def get_node_schemas_helper(workflow_id: str):
    return _node_schemas()


async def get_api_node_schemas_helper(workflow_id: str):
    return _node_schemas()


async def get_workflow_def_helper(workflow_id: str):
    workflow = _load_workflow(workflow_id)
    run_history = {}
    last_run_id = workflow.get("run_id")
    for path in RUNS_DIR.glob("*.json"):
        run = _read_json(path, {})
        if run.get("workflow_id") == workflow_id:
            last_run_id = run.get("run_id", last_run_id)
            for node_id, records in run.get("nodes", {}).items():
                run_history.setdefault(node_id, []).extend(records)
    return {**workflow, "run_id": last_run_id, "run_history": run_history}


async def get_workflow_defs_helper():
    _ensure_dirs()
    workflows = [_read_json(path, {}) for path in WORKFLOWS_DIR.glob("*.json")]
    return sorted(workflows, key=lambda item: item.get("updated_at", ""), reverse=True)


async def delete_workflow_def_by_id(workflow_id: str):
    path = _workflow_path(workflow_id)
    if path.exists():
        path.unlink()
    return {"success": True}


async def update_workflow_name_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    workflow["name"] = payload.get("name", workflow.get("name", "Untitled"))
    workflow["updated_at"] = _now()
    _write_json(_workflow_path(workflow_id), workflow)
    return workflow


async def run_workflow_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    run_id = str(uuid.uuid4())
    run = {"run_id": run_id, "workflow_id": workflow_id, "nodes": {}, "status": "running", "created_at": _now()}
    node_results = {}
    _write_json(_run_path(run_id), run)

    for node in workflow.get("data", {}).get("nodes", []):
        record = await _execute_node(node, run_id, node_results)
        run["nodes"].setdefault(node["id"], []).append(record)
        _write_json(_run_path(run_id), run)

    statuses = [records[-1]["status"] for records in run["nodes"].values() if records]
    run["status"] = "failed" if "failed" in statuses else "succeeded"
    _write_json(_run_path(run_id), run)
    workflow["run_id"] = run_id
    _write_json(_workflow_path(workflow_id), workflow)
    return {"run_id": run_id, "status": run["status"]}


async def get_run_status_helper(run_id: str):
    return _load_run(run_id)


async def run_node_helper(workflow_id: str, node_id: str, payload: dict):
    run_id = payload.get("run_id") or str(uuid.uuid4())
    run = _read_json(_run_path(run_id), {"run_id": run_id, "workflow_id": workflow_id, "nodes": {}, "status": "running", "created_at": _now()})
    node = {
        "id": node_id,
        "category": _category_from_node_id(payload.get("node_id", ""), payload.get("model", "")),
        "model": payload.get("model"),
        "params": payload.get("params", {}),
    }
    record = await _execute_node(node, run_id, {})
    run["nodes"].setdefault(node_id, []).append(record)
    run["status"] = "failed" if record["status"] == "failed" else "succeeded"
    _write_json(_run_path(run_id), run)
    return {"run_id": run_id, "node_run_id": record["node_run_id"]}


def _category_from_node_id(label: str, model: str) -> str:
    label = label.lower()
    if "image" in label or "image" in model:
        return "image"
    if "video" in label or "video" in model or model.startswith("openai-sora"):
        return "video"
    if "audio" in label:
        return "audio"
    return "text"


async def publish_workflow_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    workflow["is_published"] = bool(payload.get("publish"))
    _write_json(_workflow_path(workflow_id), workflow)
    return {"publish": workflow["is_published"]}


async def template_workflow_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    workflow["is_template"] = bool(payload.get("is_template"))
    _write_json(_workflow_path(workflow_id), workflow)
    return {"is_template": workflow["is_template"]}


async def cloudfront_signed_url_helper(payload: dict):
    return {"url": payload.get("url")}


async def generate_thumbnail_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    workflow["thumbnail"] = payload.get("thumbnail") or payload.get("url")
    _write_json(_workflow_path(workflow_id), workflow)
    return {"success": True, "thumbnail": workflow.get("thumbnail")}


async def get_file_upload_url_helper(params: dict):
    _ensure_dirs()
    filename = Path(params.get("filename", "upload.bin")).name
    key = f"{uuid.uuid4()}-{filename}"
    return {"url": "/api/app/upload", "fields": {"key": key}}


async def get_workflow_last_run(workflow_id: str):
    workflow = _load_workflow(workflow_id)
    run_id = workflow.get("run_id")
    return _load_run(run_id) if run_id else {}


async def architect_workflow_helper(payload: dict):
    return {
        "request_id": str(uuid.uuid4()),
        "status": "completed",
        "message": "Local workflow architect is not configured after removing MuAPI.",
        "suggestions": [],
        "workflow": None,
    }


async def poll_architect_result_helper(id: str):
    return {
        "status": "completed",
        "message": "Local workflow architect is not configured after removing MuAPI.",
        "suggestions": [],
        "workflow": None,
    }


async def delete_node_run_by_id_helper(node_run_id: str):
    for path in RUNS_DIR.glob("*.json"):
        run = _read_json(path, {})
        changed = False
        for node_id, records in run.get("nodes", {}).items():
            filtered = [record for record in records if record.get("node_run_id") != node_run_id]
            if len(filtered) != len(records):
                run["nodes"][node_id] = filtered
                changed = True
        if changed:
            _write_json(path, run)
            return {"success": True}
    return {"success": False}


async def update_workflow_category_helper(workflow_id: str, payload: dict):
    workflow = _load_workflow(workflow_id)
    workflow["category"] = payload.get("category", workflow.get("category", "General"))
    _write_json(_workflow_path(workflow_id), workflow)
    return workflow


async def get_workflow_api_inputs_helper(workflow_id: str):
    workflow = _load_workflow(workflow_id)
    return workflow.get("data", {}).get("nodes", [])


async def execute_workflow_via_api_helper(workflow_id: str, payload: dict):
    return await run_workflow_helper(workflow_id, payload)


async def get_workflow_api_outputs_helper(run_id: str):
    run = _load_run(run_id)
    outputs = {}
    for node_id, records in run.get("nodes", {}).items():
        if records:
            outputs[node_id] = records[-1].get("result", {}).get("outputs", [])
    return {"run_id": run_id, "outputs": outputs}
