"""01_supervisor.py -- Supervisor pattern with LangGraph conditional edges.

A supervisor agent routes tasks to specialized worker agents using
conditional edges. Each worker is a create_react_agent with its own
scoped tool set.

Pattern:
  1. Supervisor classifies the task
  2. Conditional edge routes to the appropriate worker
  3. Worker executes with its scoped tools
  4. All workers converge back to supervisor for synthesis
"""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from shared import get_openai_llm


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Simulated results
    results = {
        "python": "Python is a versatile programming language. Latest version: 3.13.",
        "rust": "Rust is a systems language focused on safety and performance.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


def supervisor_node(state: State) -> dict:
    """Classify the task and route to the appropriate worker."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Simple keyword-based routing
    lower_msg = last_message.lower()
    if any(word in lower_msg for word in ["search", "find", "look up", "research"]):
        next_agent = "researcher"
    elif any(word in lower_msg for word in ["calculate", "math", "compute", "how much"]):
        next_agent = "calculator"
    else:
        next_agent = "writer"

    return {
        "messages": [
            HumanMessage(content=f"[Supervisor] Routing to {next_agent}: {last_message[:100]}")
        ],
        "next_agent": next_agent,
    }


def researcher_node(state: State) -> dict:
    """Research worker — uses web search."""
    return {
        "messages": [
            HumanMessage(content="[Researcher] Using web search to find information...")
        ],
    }


def calculator_node(state: State) -> dict:
    """Calculator worker — uses math tools."""
    return {
        "messages": [
            HumanMessage(content="[Calculator] Computing mathematical result...")
        ],
    }


def writer_node(state: State) -> dict:
    """Writer worker — generates text, no tools."""
    return {
        "messages": [
            HumanMessage(content="[Writer] Composing response...")
        ],
    }


def synthesize_node(state: State) -> dict:
    """Combine worker output into final response."""
    return {
        "messages": [
            HumanMessage(content="[Supervisor] Synthesized final response from worker output.")
        ],
    }


def route_to_worker(state: State) -> str:
    """Conditional edge: route to the correct worker."""
    return state.get("next_agent", "writer")


def build_graph():
    """Build the supervisor graph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("writer", writer_node)
    graph.add_node("synthesize", synthesize_node)

    # Edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_worker,
        ["researcher", "calculator", "writer"],
    )
    # All workers converge to synthesize
    graph.add_edge("researcher", "synthesize")
    graph.add_edge("calculator", "synthesize")
    graph.add_edge("writer", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def main():
    print("=== Supervisor Pattern with LangGraph ===\n")

    graph = build_graph()

    tasks = [
        "Search for information about Python programming",
        "Calculate 15 * 23 + 7",
        "Write a haiku about coding",
    ]

    for task in tasks:
        print(f"--- Task: {task} ---")
        result = graph.invoke({"messages": [HumanMessage(content=task)]})
        for msg in result["messages"]:
            content = msg.content
            if content and ("[Supervisor]" in content or "[Researcher]" in content
                           or "[Calculator]" in content or "[Writer]" in content):
                print(f"  {content}")
        print()


if __name__ == "__main__":
    main()
