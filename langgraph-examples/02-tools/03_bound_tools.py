"""03_bound_tools.py -- tool_choice controls when and which tools the LLM must use.

In the root project:
  request = MessageRequest(
      model=settings.openai_model,
      messages=messages,
      tools=registry.get_specs(),
      tool_choice=ToolChoice(type="auto"),      # model decides
      tool_choice=ToolChoice(type="required"),    # must call a tool
      tool_choice=ToolChoice(type="none"),        # must not call any tool
  )

In LangChain (via bind_tools or bind):
  llm.bind_tools(tools, tool_choice="auto")       # default — model decides
  llm.bind_tools(tools, tool_choice="required")   # must call at least one tool
  llm.bind_tools(tools, tool_choice="none")       # must not call any tool
  llm.bind_tools(tools, tool_choice="any")        # must call a tool, picks which
  llm.bind_tools(tools, tool_choice="calculate")  # must call this specific tool
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from shared import settings

MODEL = settings.openai_model


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a science or math topic (e.g., pi, speed of light, earth circumference)."""
    facts = {
        "pi": "3.14159...",
        "speed of light": "299,792,458 m/s",
        "earth circumference": "40,075 km",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'")


def show(choice: str, query: str, model: str = MODEL) -> None:
    """Invoke LLM with a given tool_choice and print what happened."""
    llm = ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
        base_url=settings.openai_base_url,
    ).bind_tools(
        [calculate, lookup_fact],
        tool_choice=choice,
    )
    response = llm.invoke([HumanMessage(content=query)])

    print(f"  Query: {query}")
    if response.tool_calls:
        for tc in response.tool_calls:
            result = {
                "calculate": calculate,
                "lookup_fact": lookup_fact,
            }[tc["name"]].invoke(tc["args"])
            print(f"  -> Tool: {tc['name']}({tc['args']}) => {result}")
    else:
        print(f"  -> Direct: {response.content[:100]}")
    print()


def main():
    query = "What is 15 * 27?"

    print('=== tool_choice: "auto" (default, model decides) ===\n')
    show("auto", query)
    print('=== tool_choice: "required" (must call at least one tool) ===\n')
    show("required", query)
    print('=== tool_choice: "calculate" (must call this specific tool) ===\n')
    show("calculate", query)
    print('=== tool_choice: "calculate" on a non-math query ===\n')
    show("calculate", "What is the capital of France?")

    print('=== tool_choice: "none" (must not call any tool) ===\n')
    print('  Note: openai/gpt-oss-20b:free does not support tool_choice="none".')
    try:
        show("none", query)
    except Exception as e:
        print(f"  ERROR: {e}\n")

    print("=== Takeaways ===\n")
    print('1. tool_choice="auto"       — model decides (default)')
    print('2. tool_choice="required"   — must call at least one tool')
    print('3. tool_choice="calculate"  — must call this specific tool')
    print(
        '4. tool_choice="none"       — must not call any tool (not all models support it)'
    )
    print()
    print("Caveat: tool_choice support depends on the model and provider.")
    print('  - gpt-oss-20b:free: supports auto/required/specific, not "none"')
    print('  - gpt-4.1-mini: supports all modes including "none"')


if __name__ == "__main__":
    main()
