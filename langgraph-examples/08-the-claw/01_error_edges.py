"""01_error_edges.py -- Error handling via conditional edge routing.

In the root project:
  loop = RepairLoop(registry, max_retries=3)
  result = await loop.repair("shell", {"command": "ls"})
  # RepairLoop wraps dispatch with try/except and retry counter

In LangGraph:
  graph.add_edge("error_handler", "risky")  # loop back for retry
  graph.add_conditional_edges("risky", check_result, {"error_handler": ..., "end": END})

The graph topology IS the retry mechanism. A conditional edge from the
risky node routes to an error handler (which loops back) or END (on success).
No separate RepairLoop wrapper needed.
"""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ErrorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    errors: list[str]
    retry_count: int


def risky_operation(state: ErrorState) -> dict:
    """A node that simulates failure on first attempts."""
    attempt = state["retry_count"]

    if attempt < 2:
        error_msg = f"Attempt {attempt + 1}: connection timeout"
        print(f"  [RISKY] {error_msg}")
        return {
            "errors": state["errors"] + [error_msg],
            "retry_count": attempt + 1,
        }

    print(f"  [RISKY] Operation succeeded on attempt {attempt + 1}")
    return {
        "messages": [AIMessage(content="Operation succeeded!")],
        "retry_count": attempt + 1,
    }


def error_handler(state: ErrorState) -> dict:
    """Handle errors and prepare for retry."""
    last_error = state["errors"][-1] if state["errors"] else "Unknown error"
    print(f"  [HANDLER] Handling: {last_error} -> will retry")
    return {
        "messages": [AIMessage(content=f"Error handled: {last_error}. Retrying...")],
    }


def check_result(state: ErrorState) -> Literal["error_handler", "end"]:
    """Route based on whether the operation succeeded."""
    # Check if we got a success message
    has_success = any(
        "succeeded" in getattr(m, "content", "") for m in state["messages"]
    )
    if has_success:
        return "end"
    if state["retry_count"] >= 3:
        return "end"
    return "error_handler"


def main():
    graph = StateGraph(ErrorState)
    graph.add_node("risky", risky_operation)
    graph.add_node("error", error_handler)

    graph.add_edge(START, "risky")
    graph.add_conditional_edges(
        "risky", check_result, {"error_handler": "error", "end": END}
    )
    graph.add_edge("error", "risky")  # Retry loop!

    app = graph.compile()

    print("=== Error Edge Routing ===\n")
    result = app.invoke(
        {
            "messages": [],
            "errors": [],
            "retry_count": 0,
        }
    )

    print("\n=== Final State ===")
    print(f"Retry count: {result['retry_count']}")
    print(f"Errors: {result['errors']}")
    print(f"Final: {result['messages'][-1].content}")

    print("\n=== Graph Topology ===")
    print("  START -> risky -> check_result")
    print("                      |-> error_handler -> risky (retry loop)")
    print("                      |-> END")
    print("\nThe retry is encoded in the graph edges, not in a wrapper class.")


if __name__ == "__main__":
    main()
