"""02_reducer_state.py -- State reducers: add_messages, operator.add, custom.

In the root project, WorkflowState.merge_context() does dict-level merging.
There is no field-level merge strategy.

In LangGraph, each state field can have its own reducer:
  - Annotated[list, add_messages]: appends, deduplicates by ID
  - Annotated[int, add]: numeric accumulation (0 + 1 + 1 = 2)
  - No Annotated: last-write-wins
"""

from operator import add
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AccumulatorState(dict):
    messages: Annotated[list, add_messages]
    counter: Annotated[int, add]
    label: str


def node_1(state: AccumulatorState) -> dict:
    print("  node_1: adding message, incrementing counter")
    return {
        "messages": [HumanMessage(content="Hello from node 1")],
        "counter": 1,
        "label": "after_node_1",
    }


def node_2(state: AccumulatorState) -> dict:
    print(f"  node_2: counter={state['counter']}, label={state['label']}")
    return {
        "messages": [AIMessage(content="Hello from node 2")],
        "counter": 1,
        "label": "after_node_2",
    }


def node_3(state: AccumulatorState) -> dict:
    print(f"  node_3: {len(state['messages'])} messages so far")
    return {
        "messages": [HumanMessage(content="Hello from node 3")],
        "label": "final",
    }


def main():
    graph = StateGraph(AccumulatorState)
    graph.add_node("n1", node_1)
    graph.add_node("n2", node_2)
    graph.add_node("n3", node_3)

    graph.add_edge(START, "n1")
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    graph.add_edge("n3", END)

    app = graph.compile()

    print("=== Executing Graph ===")
    result = app.invoke({"messages": [], "counter": 0, "label": "start"})

    print(f"\n=== Final State ===")
    print(f"Messages ({len(result['messages'])} -- accumulated by add_messages):")
    for m in result["messages"]:
        print(f"  [{m.type}] {m.content}")

    print(f"\nCounter: {result['counter']} (0 + 1 + 1 + 0 = 2 -- node_3 didn't add)")
    print(f"Label: '{result['label']}' (last-write-wins -- 'final' overwrote 'after_node_2')")

    # Demonstrate reducer behavior
    print("\n=== Reducer Behavior Summary ===")
    print("add_messages: appends new messages, deduplicates by ID")
    print("operator.add: numeric accumulation (initial + node_1 + node_2 + ...)")
    print("(no reducer): last-write-wins")


if __name__ == "__main__":
    main()
