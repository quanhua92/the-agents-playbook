"""02_command_resuming.py -- Command-based resumption after interrupt.

Key points:
  - Any node can call interrupt(value) to pause graph execution
  - Caller checks result["__interrupt__"] to see what the graph is waiting for
  - Caller calls invoke(Command(resume=...)) with the same config to resume
  - The resume value becomes the return value of interrupt() inside the node
  - The resume value can influence routing via conditional edges

In the root project, human-in-the-loop happens BEFORE execution:
  if middleware.should_prompt("edit"):
      response = prompter.ask("Approve?")
  result = await registry.dispatch("edit", args)

In LangGraph, human-in-the-loop happens MID-EXECUTION:
  graph runs -> hits interrupt() -> PAUSES -> human responds -> RESUMES
  The human's response can influence subsequent routing via conditional edges.
"""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    decision: str


def gate(state: State) -> dict:
    """Ask user which approach to take."""
    response = interrupt(
        {
            "question": "Which approach do you want?",
            "options": {
                "a": "aggressive (fast, risky)",
                "b": "conservative (slow, safe)",
            },
        }
    )
    decision = response.get("choice", "a")
    return {"decision": decision}


def aggressive(state: State) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Taking aggressive approach: moving fast, accepting risk."
            )
        ]
    }


def conservative(state: State) -> dict:
    return {
        "messages": [
            AIMessage(content="Taking conservative approach: careful, step-by-step.")
        ]
    }


def route_by_decision(state: State) -> Literal["aggressive", "conservative"]:
    return "aggressive" if state["decision"] == "a" else "conservative"


def main():
    graph = StateGraph(State)
    graph.add_node("gate", gate)
    graph.add_node("aggressive", aggressive)
    graph.add_node("conservative", conservative)

    graph.add_edge(START, "gate")
    graph.add_conditional_edges(
        "gate", route_by_decision, ["aggressive", "conservative"]
    )
    graph.add_edge("aggressive", END)
    graph.add_edge("conservative", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # --- Scenario 1: Choose conservative ---
    print("=== Scenario 1: Choose Conservative ===\n")
    config1: RunnableConfig = {"configurable": {"thread_id": "conservative-path"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Fix the bug in auth.py")], "decision": ""},
        config1,
    )
    if "__interrupt__" in result:
        for i in result["__interrupt__"]:
            print(f"Graph interrupted: {i.value}")
        print()

    result1 = app.invoke(Command(resume={"choice": "b"}), config1)
    print(f"Decision: {result1['decision']}")
    print(f"Response: {result1['messages'][-1].content}")

    # --- Scenario 2: Choose aggressive ---
    print("\n=== Scenario 2: Choose Aggressive ===\n")
    config2: RunnableConfig = {"configurable": {"thread_id": "aggressive-path"}}

    result = app.invoke(
        {"messages": [HumanMessage(content="Fix the bug in auth.py")], "decision": ""},
        config2,
    )
    if "__interrupt__" in result:
        for i in result["__interrupt__"]:
            print(f"Graph interrupted: {i.value}")
        print()

    result2 = app.invoke(Command(resume={"choice": "a"}), config2)
    print(f"Decision: {result2['decision']}")
    print(f"Response: {result2['messages'][-1].content}")

    print("\n=== Key Insight ===")
    print("Root: permission check before execution")
    print("Here: interrupt mid-execution, human response influences routing")
    print("The graph topology itself changes based on human input!")


if __name__ == "__main__":
    main()
