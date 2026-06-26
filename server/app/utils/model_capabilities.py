"""Per-model input capabilities loaded from app/model_capabilities.json.

Turns the share-ready capability profiles into the input-property shape the node
schema (and UI) expects, and provides validation helpers. Fully defensive: if the
JSON is missing or malformed, every lookup returns None so callers fall back to the
generic per-category schema and nothing breaks.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CAPS_PATH = Path(__file__).resolve().parent.parent / "model_capabilities.json"

try:
    _DATA = json.loads(_CAPS_PATH.read_text(encoding="utf-8"))
    _FIELDS = _DATA.get("field_library", {})
    _PROFILES = _DATA.get("profiles", {})
    _MODELS = _DATA.get("models", {})
except Exception as exc:  # pragma: no cover - defensive
    logger.warning("model_capabilities: could not load %s (%s); using generic schemas", _CAPS_PATH, exc)
    _DATA, _FIELDS, _PROFILES, _MODELS = {}, {}, {}, {}


def _resolve_field(entry) -> Optional[tuple[str, dict, bool]]:
    """Return (param_name, property_dict, is_required) for one profile field entry."""
    if isinstance(entry, str):
        lib_key, overrides = entry, {}
    elif isinstance(entry, dict) and entry.get("ref"):
        lib_key, overrides = entry["ref"], entry
    else:
        return None
    lib = _FIELDS.get(lib_key)
    if not lib:
        return None

    name = overrides.get("name") or lib.get("name") or lib_key
    title = lib.get("title", name)
    prop = {"title": title, "name": name, "type": "string", "description": title}
    if "default" in lib:
        prop["default"] = lib["default"]
    if lib.get("field"):
        prop["field"] = lib["field"]
    if lib.get("enum"):
        prop["enum"] = lib["enum"]
        prop["type"] = "string"
    elif lib.get("type") == "array":
        prop["type"] = "array"
        prop["items"] = {"type": "string"}
        prop["maxItems"] = overrides.get("max") or lib.get("max") or 10
    else:
        prop["type"] = lib.get("type", "string")

    return name, prop, bool(overrides.get("required"))


def _profile_for(model_id: str) -> Optional[dict]:
    key = _MODELS.get(model_id)
    return _PROFILES.get(key) if key else None


def properties_and_required(model_id: str) -> Optional[tuple[dict, list[str]]]:
    """(properties, required) for a mapped model, or None to use the generic schema."""
    profile = _profile_for(model_id)
    if not profile:
        return None
    properties: dict = {}
    required: list[str] = []
    for entry in profile.get("fields", []):
        resolved = _resolve_field(entry)
        if not resolved:
            continue
        name, prop, is_required = resolved
        properties[name] = prop
        if is_required:
            required.append(name)
    if not properties:
        return None
    return properties, required


def allowed_keys(model_id: str) -> Optional[set[str]]:
    spec = properties_and_required(model_id)
    return set(spec[0].keys()) if spec else None


def required_keys(model_id: str) -> list[str]:
    spec = properties_and_required(model_id)
    return list(spec[1]) if spec else []


def missing_input_message(model_id: str, missing: list[str]) -> str:
    spec = properties_and_required(model_id)
    titles = []
    props = spec[0] if spec else {}
    for key in missing:
        titles.append(props.get(key, {}).get("title", key))
    label = ", ".join(titles) if titles else "required input"
    return f"This model requires: {label}."
