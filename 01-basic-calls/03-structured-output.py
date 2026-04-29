import asyncio
import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from the_agents_playbook import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Reviewer(BaseModel):
    name: str
    publication: str


class MovieReview(BaseModel):
    title: str
    year: int
    reviewer: Reviewer
    rating: float = Field(ge=0.0, le=10.0)
    summary: str
    genre: list[str]
    recommended: bool


# Pydantic emits $ref pointers under a $defs key for nested models.
# Some providers (especially via OpenRouter) can't resolve $ref — they
# need a flat, fully inlined schema. This utility walks the schema and
# replaces all $ref entries with the actual definitions from $defs.
#
# Note: The OpenAI API itself expects a raw JSON Schema dict — it does not
# accept Pydantic models. However, the OpenAI Python SDK
# (client.beta.chat.completions.parse) can accept a Pydantic model
# directly and handles $ref flattening + strict mode compliance for you.
# Since we're using raw httpx, we must do this ourselves.
def flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    # Extract definitions safely (default to empty dict if missing)
    defs: dict[str, Any] = schema.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            # 1. Resolve $ref links
            if "$ref" in obj:
                key: str = obj["$ref"].split("/")[-1]
                return _resolve(defs[key])

            # 2. Apply OpenAI strict object rules
            if obj.get("type") == "object":
                obj["additionalProperties"] = False

                # All properties must be declared as required
                if "properties" in obj:
                    obj["required"] = list(obj["properties"].keys())

            return {k: _resolve(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [_resolve(v) for v in obj]

        return obj

    return _resolve(schema)


async def run():
    review_text = "Avatar (2009) by James Cameron. A paraplegic marine on the alien planet Pandora. Great visuals. 7.5/10. Sci-fi action. Recommended. — Reviewed by Alex Chen for Rotten Tomatoes."

    # json_schema vs json_object: json_object only guarantees valid JSON output;
    # json_schema enforces conformance to a specific schema (strictly more powerful).
    # strict: True requires all fields present, correct types, no extra fields.
    # Not all models support this — unsupported models either ignore it or error.
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "MovieReview",
            "strict": True,
            # model_json_schema() generates a JSON Schema dict from the Pydantic model.
            # flatten_json_schema() inlines all $ref/$defs so providers that can't
            # resolve references (e.g. some via OpenRouter) still work correctly.
            "schema": flatten_json_schema(MovieReview.model_json_schema()),
        },
    }

    body = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": "Extract movie review data from the user's text into the requested JSON schema.",
            },
            {"role": "user", "content": review_text.strip()},
        ],
        "response_format": response_format,
    }

    logging.info("Request body: %s", json.dumps(body, indent=2))

    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.openai_base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

        data = response.json()
        logging.info("Raw response: %s", json.dumps(data, indent=2))

        message = data["choices"][0]["message"]
        usage = data["usage"]

        print(f"Model:  {data['model']}")
        print(f"Finish: {data['choices'][0].get('finish_reason')}")

        # model_validate_json() parses the JSON string back into a typed Pydantic instance.
        # Validates all fields — if the LLM returned "two thousand nine" for year,
        # this would raise a ValidationError.
        review = MovieReview.model_validate_json(message["content"])
        print(review.model_dump_json(indent=2))

        print(
            f"Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total"
        )
        print(f"Cost:   ${usage['cost']:.6f}")

    # Limitations:
    # - Not all models support response_format: json_schema. Unsupported models
    #   either ignore it (returning freeform text) or return a 400 error.
    # - OpenRouter routes to different providers — json_schema support depends
    #   on the underlying provider, not OpenRouter itself. The response may
    #   silently degrade to unstructured text if routed to an incompatible provider.


def main():
    asyncio.run(run())


main()
