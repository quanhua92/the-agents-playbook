"""01_basic_chat.py -- Basic ChatOpenAI/ChatAnthropic call.

Replaces the root project's raw httpx POST in 02-basic-chat.py.
No raw HTTP, no manual JSON parsing -- LangChain handles it all.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from shared import get_openai_llm, get_anthropic_llm, settings


def main():
    # --- OpenAI via OpenRouter ---
    print("=== ChatOpenAI ===")
    llm = get_openai_llm()

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ])

    print(f"Content:   {response.content}")
    usage = response.usage_metadata or {}
    print(f"Tokens:    {usage}")
    print(f"Model:     {response.response_metadata.get('model', 'N/A')}")
    print(f"Finish:    {response.response_metadata.get('finish_reason', 'N/A')}")

    # --- Anthropic ---
    if settings.anthropic_api_key:
        print("\n=== ChatAnthropic ===")
        claude = get_anthropic_llm()
        response2 = claude.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2 + 2?"),
        ])
        print(f"Content:   {response2.content}")
        usage2 = response2.usage_metadata or {}
        print(f"Tokens:    {usage2}")
    else:
        print("\n=== ChatAnthropic === (skipped, no ANTHROPIC_API_KEY)")


if __name__ == "__main__":
    main()
