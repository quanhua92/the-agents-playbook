"""02_parallel_agents.py -- Parallel fan-out with LangGraph Send() API.

Uses Send() to fan out tasks to multiple agent workers in parallel,
then aggregate the results. This pattern is useful when subtasks
are independent and can run concurrently.

Pattern:
  1. Supervisor identifies parallel subtasks
  2. Send() fans out to multiple workers simultaneously
  3. Each worker processes its subtask independently
  4. Results are aggregated into a final answer
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    subtasks: list[str]


def supervisor_node(state: State) -> dict:
    """Identify subtasks and store them in state.

    The conditional_edges on this node will fan out via Send().
    """
    task = state["messages"][-1].content if state["messages"] else ""

    subtasks = [
        f"Research the positive aspects of: {task}",
        f"Research the challenges of: {task}",
    ]

    return {"subtasks": subtasks}


def fan_out(state: State) -> list[Send]:
    """Return Send() objects for parallel fan-out from conditional_edges."""
    return [
        Send("research_worker", {"messages": [HumanMessage(content=st)]})
        for st in state["subtasks"]
    ]


def research_worker(state: State) -> dict:
    """A worker that processes a single research subtask."""
    last_msg = state["messages"][-1].content if state["messages"] else ""
    # In production, this would invoke a real agent with tools
    return {
        "messages": [
            HumanMessage(
                content=f"[Worker] Processed: {last_msg[:60]}...\nResult: Key findings gathered."
            )
        ],
    }


def aggregator_node(state: State) -> dict:
    """Combine all worker outputs into a final response."""
    messages = state.get("messages", [])
    worker_count = sum(1 for m in messages if "[Worker]" in m.content)
    return {
        "messages": [
            HumanMessage(
                content=f"[Aggregator] Combined {worker_count} worker results into final answer."
            )
        ],
    }


def build_graph():
    """Build the parallel fan-out graph."""
    graph = StateGraph(State)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_worker", research_worker)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "supervisor")
    # conditional_edges returns Send() for parallel fan-out
    graph.add_conditional_edges("supervisor", fan_out)
    graph.add_edge("research_worker", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


def main():
    print("=== Parallel Agents with LangGraph Send() ===\n")

    graph = build_graph()

    task = "renewable energy adoption in 2026"
    print(f"Task: {task}\n")
    print("Supervisor fans out to 2 parallel workers:\n")

    result = graph.invoke({"messages": [HumanMessage(content=task)], "subtasks": []})

    print("Execution trace:")
    for msg in result["messages"]:
        if msg.content and ("[Worker]" in msg.content or "[Aggregator]" in msg.content):
            print(f"  {msg.content}")

    print("\n=== Pattern Summary ===\n")
    print("1. Supervisor receives task and identifies parallel subtasks")
    print("2. Send() fans out to N workers simultaneously")
    print("3. Each worker processes independently (true parallelism)")
    print("4. All worker outputs are collected in the shared message state")
    print("5. Aggregator combines results into a final answer")
    print("\nKey difference from sequential dispatch:")
    print("  - Sequential: task1 -> task2 -> task3 (one at a time)")
    print("  - Parallel:   task1 | task2 | task3 (concurrent, faster)")


if __name__ == "__main__":
    main()
