"""01_typed_state.py -- TypedDict state vs root's WorkflowState.

In the root project:
  state = WorkflowState()           # mutable dataclass
  state.set_context("plan", plan)   # direct mutation
  state.merge_context(updates)      # dict-level merge

In LangGraph:
  class AgentState(TypedDict):
      messages: Annotated[list[BaseMessage], add_messages]
      steps_completed: int

Nodes return partial updates. Reducers merge them into immutable state.
"""

from typing import Any, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """LangGraph state definition.

    Contrast with root's WorkflowState dataclass:
    - Root: mutable dataclass with .set_context(), .get_context()
    - LangGraph: nodes return partial updates, reducers merge them
    """

    messages: Annotated[list[BaseMessage], add_messages]
    steps_completed: int
    metadata: dict[str, Any]


def step_a(state: AgentState) -> dict:
    steps = state.get("steps_completed", 0)
    print(f"  step_a: steps so far = {steps}")
    return {
        "messages": [HumanMessage(content="Step A executed")],
        "steps_completed": steps + 1,
    }


def step_b(state: AgentState) -> dict:
    steps = state["steps_completed"]
    print(f"  step_b: steps_completed = {steps}")
    return {"metadata": {"last_step": "b"}}


def step_c(state: AgentState) -> dict:
    meta = state.get("metadata", {})
    print(f"  step_c: metadata = {meta}")
    return {"steps_completed": state["steps_completed"] + 1}


def main():
    graph = StateGraph(AgentState)
    graph.add_node("a", step_a)
    graph.add_node("b", step_b)
    graph.add_node("c", step_c)

    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)

    app = graph.compile()

    print("=== Executing Graph ===")
    result = app.invoke(
        {
            "messages": [],
            "steps_completed": 0,
            "metadata": {},
        }
    )

    print("\n=== Final State ===")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Messages: {len(result['messages'])}")
    for msg in result["messages"]:
        print(f"  [{msg.type}] {msg.content}")


if __name__ == "__main__":
    main()
