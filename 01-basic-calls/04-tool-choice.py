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


async def run():
    review_text = "Avatar (2009) by James Cameron. A paraplegic marine on the alien planet Pandora. Great visuals. 7.5/10. Sci-fi action. Recommended. — Reviewed by Alex Chen for Rotten Tomatoes."

    # Define a tool spec using MovieReview as the parameter schema.
    # The model will be forced to call this tool, returning structured
    # data in tool_calls[0].function.arguments instead of message.content.
    tool_spec = {
        "type": "function",
        "function": {
            "name": "submit_movie_review",
            "description": "Submit a structured movie review",
            "parameters": flatten_json_schema(MovieReview.model_json_schema()),
        },
    }

    # Force the model to call this specific tool.
    # Some models don't support tool_choice with a specific function name.
    # If the request fails, we fall back to tool_choice: "required" with a
    # single tool defined — this has the same forcing effect.
    tool_choice: dict[str, Any] | str = {
        "type": "function",
        "function": {"name": "submit_movie_review"},
    }

    body = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": "Extract movie review data from the user's text by calling the submit_movie_review tool.",
            },
            {"role": "user", "content": review_text.strip()},
        ],
        "tools": [tool_spec],
        "tool_choice": tool_choice,
        "temperature": 0.7,
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

        # The tool_calls response structure: message.tool_calls is a list,
        # each item has id, type, and function (with name + arguments as a
        # JSON string). This is the same shape in real tool-use flows —
        # the difference is we're not actually executing the tool.
        #
        # Note: arguments is a JSON string, not a dict. Parse it carefully.
        # In later chapters the agent will dispatch to a real Tool.execute()
        # and feed the result back as a tool message. This demo shows only
        # the first half of that flow.
        if "tool_calls" in message:
            tool_call = message["tool_calls"][0]
            logging.info(
                "Tool call: id=%s name=%s arguments=%s",
                tool_call["id"],
                tool_call["function"]["name"],
                tool_call["function"]["arguments"],
            )
            raw_json = tool_call["function"]["arguments"]
        else:
            # Model didn't return tool_calls — may not support tools at all.
            # Fall back to parsing content as JSON.
            logging.warning("No tool_calls in response, falling back to content parsing")
            raw_json = message["content"]

        review = MovieReview.model_validate_json(raw_json)
        print(review.model_dump_json(indent=2))

        print(
            f"Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total"
        )
        print(f"Cost:   ${usage['cost']:.6f}")

    # Tool choice vs response_format: json_schema
    # - Some models support tool_choice but not response_format: json_schema.
    #   Tool choice is the more widely supported mechanism.
    # - Tool choice is the natural bridge to real tool use — the same format
    #   you learn here is the format used when agents actually call tools.
    # - Not all models support tool_choice: { type: "function" }. Some only
    #   support "auto" or "required". The try/except fallback handles this.
    # - Very old or non-OpenAI-compatible models may silently ignore the
    #   tools parameter entirely.


def main():
    asyncio.run(run())


main()
