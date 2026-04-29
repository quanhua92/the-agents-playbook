from typing import Any


def flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $ref/$defs in a JSON Schema for provider compatibility.

    Pydantic emits $ref pointers under a $defs key for nested models.
    Some providers (especially via OpenRouter) can't resolve $ref — they
    need a flat, fully inlined schema. This utility walks the schema and
    replaces all $ref entries with the actual definitions from $defs.

    Also applies OpenAI strict object rules: adds additionalProperties: false
    and auto-populates required for all nested objects.

    Note: The OpenAI API itself expects a raw JSON Schema dict — it does not
    accept Pydantic models. However, the OpenAI Python SDK
    (client.beta.chat.completions.parse) can accept a Pydantic model
    directly and handles $ref flattening + strict mode compliance for you.
    Since we're using raw httpx, we must do this ourselves.
    """
    defs: dict[str, Any] = schema.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                key: str = obj["$ref"].split("/")[-1]
                return _resolve(defs[key])

            if obj.get("type") == "object":
                obj["additionalProperties"] = False

                if "properties" in obj:
                    obj["required"] = list(obj["properties"].keys())

            return {k: _resolve(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [_resolve(v) for v in obj]

        return obj

    return _resolve(schema)
