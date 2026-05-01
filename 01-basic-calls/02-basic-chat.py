import httpx
import asyncio

from the_agents_playbook import settings


async def run():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.openai_base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
            },
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]
        usage = data["usage"]

        print(f"Model:     {data['model']}")
        print(f"Provider:  {data.get('provider', 'N/A')}")
        print(f"Finish:    {data['choices'][0].get('finish_reason')}")
        print(f"Content:   {message['content']}")
        if message.get("reasoning"):
            print(f"Reasoning: {message['reasoning']}")
        print(
            f"Tokens:    {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total"
        )
        print(f"Cost:      ${usage['cost']:.6f}")


def main():
    asyncio.run(run())


main()
