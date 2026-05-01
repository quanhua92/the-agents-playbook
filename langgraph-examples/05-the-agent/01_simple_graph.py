"""01_simple_graph.py -- Basic StateGraph with nodes and edges.

This is the most fundamental LangGraph concept. Every LangGraph agent is
a StateGraph under the hood -- nodes do work, edges define flow.

In the root project, the Agent class hand-rolls the entire loop:
  while iteration < max_iterations:
      response = await provider.send_message(request)
      if tool_calls: dispatch -> feed results back
      else: return response

In LangGraph, this is declarative:
  graph = StateGraph(AgentState)
  graph.add_node("chatbot", chatbot)
  graph.add_edge(START, "chatbot")
  graph.add_edge("chatbot", END)
  app = graph.compile()
"""

from typing import Annotated, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: GraphState) -> dict:
    """A simple node that echoes back the last message."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        return {"messages": [AIMessage(content=f"You said: {last_msg.content}")]}
    return {"messages": [AIMessage(content="(no user message)")]}


def main():
    graph = StateGraph(GraphState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    app = graph.compile()

    print("=== Simple Graph ===\n")

    inputs = [
        {"messages": [HumanMessage(content="Hello, world!")]},
        {"messages": [HumanMessage(content="What is LangGraph?")]},
    ]

    for inp in inputs:
        print(f"Input: {inp['messages'][0].content}")
        result = app.invoke(cast(GraphState, inp))
        print(f"Output: {result['messages'][-1].content}")
        print()

    # Visualize the graph structure
    print("=== Graph Structure ===")
    print("START -> chatbot -> END")
    print("\nIn subsequent examples, we'll add:")
    print("  - ToolNode (automatic tool dispatch)")
    print("  - Conditional edges (routing based on state)")
    print("  - Checkpointing (memory via thread_id)")
    print("  - interrupt() (human-in-the-loop)")


if __name__ == "__main__":
    main()
