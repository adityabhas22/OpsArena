from __future__ import annotations

from typing import Any, get_args, get_origin

from .models import OPS_ACTION_ADAPTER


def _schema_to_properties(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    cleaned = {}
    for name, prop in properties.items():
        if name == "metadata":
            continue
        cleaned[name] = {
            "type": prop.get("type", "string"),
            "description": prop.get("description", ""),
        }
        if "enum" in prop:
            cleaned[name]["enum"] = prop["enum"]
        if "anyOf" in prop:
            cleaned[name]["anyOf"] = prop["anyOf"]
    return cleaned, [item for item in required if item != "metadata"]


def iter_action_models() -> list[type]:
    annotated = OPS_ACTION_ADAPTER.annotation
    inner = get_args(annotated)[0]
    if get_origin(inner) is None:
        return [inner]
    return list(get_args(inner))


def render_action_catalog_for_llm() -> str:
    lines: list[str] = []
    for model in iter_action_models():
        schema = model.model_json_schema()
        props, required = _schema_to_properties(schema)
        fields = ", ".join(
            f"{name}{'*' if name in required else ''}" for name in props if name != "action_type"
        )
        action_name = model.model_fields["action_type"].default
        lines.append(f"- {action_name}: {fields or 'no parameters'}")
    return "\n".join(lines)


def render_action_catalog_as_json() -> list[dict[str, Any]]:
    tools = []
    for model in iter_action_models():
        schema = model.model_json_schema()
        props, required = _schema_to_properties(schema)
        action_name = model.model_fields["action_type"].default
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": action_name,
                    "description": schema.get("description", f"Execute {action_name}"),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools
