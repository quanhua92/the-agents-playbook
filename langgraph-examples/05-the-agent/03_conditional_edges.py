"""03_conditional_edges.py -- Conditional routing replaces entropy scoring.

In the root project:
  score = score_tools(tool_probs)  # Shannon entropy
  if score < threshold:  # Low uncertainty -> use best tool
  elif score < high:     # Medium uncertainty -> ask
  else:                  # High uncertainty -> delegate

In LangGraph:
  def route_message(state) -> "question" | "exclamation" | "statement":
      last = state["messages"][-1].content
      if "?" in last: return "question"
      ...
  graph.add_conditional_edges("classifier", route_message, [...])

Any function can be a conditional edge -- branching logic is explicit
and declarative, encoded in the graph topology.
"""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class RouterState(dict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str


def classifier(state: RouterState) -> dict:
    """Classify the input and set route."""
    last = state["messages"][-1].content if state["messages"] else ""
    if "?" in last:
        return {"route": "question"}
    elif "!" in last or last.isupper():
        return {"route": "exclamation"}
    else:
        return {"route": "statement"}


def handle_question(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="I see you have a question. Let me help you with that.")]}


def handle_exclamation(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="No need to shout! I'm right here and ready to help.")]}


def handle_statement(state: RouterState) -> dict:
    return {"messages": [AIMessage(content="Thanks for the information. Noted.")]}


def route_message(state: RouterState) -> Literal["question", "exclamation", "statement"]:
    return state["route"]


def main():
    graph = StateGraph(RouterState)
    graph.add_node("classifier", classifier)
    graph.add_node("question", handle_question)
    graph.add_node("exclamation", handle_exclamation)
    graph.add_node("statement", handle_statement)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_message,
        ["question", "exclamation", "statement"],
    )
    graph.add_edge("question", END)
    graph.add_edge("exclamation", END)
    graph.add_edge("statement", END)

    app = graph.compile()

    print("=== Conditional Edge Routing ===\n")

    test_inputs = [
        "What is AI?",
        "THIS IS AMAZING!",
        "The sky is blue today.",
    ]

    for inp in test_inputs:
        result = app.invoke({"messages": [HumanMessage(content=inp)], "route": ""})
        reply = result["messages"][-1]
        print(f"  '{inp}'")
        print(f"    -> route={result['route']}")
        print(f"    -> '{reply.content}'")
        print()

    print("=== Root Comparison ===")
    print("Root:  Shannon entropy score -> if/elif branching in Agent.run()")
    print("Here:  classifier node -> conditional_edges() in graph topology")
    print("Both:  route based on input characteristics")


if __name__ == "__main__":
    main()
