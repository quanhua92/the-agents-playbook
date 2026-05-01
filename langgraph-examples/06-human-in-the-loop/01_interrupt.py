"""01_interrupt.py -- interrupt() replaces PermissionMiddleware.

Key points:
  - Any node can call interrupt(value) to pause graph execution
  - Caller checks result["__interrupt__"] to see what the graph is waiting for
  - Caller calls invoke(Command(resume=...)) with the same config to resume
  - The resume value becomes the return value of interrupt() inside the node
  - Requires a checkpointer (MemorySaver or SqliteSaver) to persist state

In the root project:
  middleware = PermissionMiddleware()
  middleware.annotate("edit", RiskLevel.WORKSPACE_WRITE)
  if middleware.should_prompt("edit"):
      user_response = prompter.ask("Approve edit?")
  if user_response.approved:
      result = await registry.dispatch("edit", args)

In LangGraph:
  def review_action(state):
      user_response = interrupt({"question": "Approve?", "risk": "DANGER"})
      if user_response["approved"]:
          return {"messages": [...success...]}
      else:
          return {"messages": [...cancelled...]}
"""

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool


def review_action(state: AgentState) -> dict:
    """A node that needs human approval before proceeding.

    interrupt() PAUSES graph execution here. The graph state is saved
    to the checkpoint. Execution resumes when Command(resume=...) is called.
    """
    user_response = interrupt(
        {
            "question": "Should the agent proceed with editing README.md?",
            "risk_level": "WORKSPACE_WRITE",
            "action": "edit",
            "target": "README.md",
        }
    )

    if user_response.get("approved", False):
        return {
            "messages": [AIMessage(content="File edited successfully.")],
            "approved": True,
        }
    else:
        return {
            "messages": [AIMessage(content="Action cancelled by user.")],
            "approved": False,
        }


def main():
    graph = StateGraph(AgentState)
    graph.add_node("review", review_action)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": "approval-test"}}

    # --- Step 1: invoke (graph will interrupt) ---
    print("=== Step 1: Invoke (graph interrupts) ===")
    print("Input: request to edit README.md\n")

    result = app.invoke(
        {
            "messages": [HumanMessage(content="Please edit README.md")],
            "approved": False,
        },
        config,
    )

    if "__interrupt__" in result:
        for i in result["__interrupt__"]:
            print(f"Graph interrupted: {i.value}")
        print()

    # --- Inspect checkpoint state ---
    state = app.get_state(config)
    print("=== Checkpoint State ===")
    print(f"Next nodes: {state.next}")
    print(f"Tasks: {state.tasks}")
    print()

    # --- Step 2: Resume with approval ---
    print("=== Step 2: Resume with approval ===")
    result = app.invoke(
        Command(resume={"approved": True}),
        config,
    )

    print(f"Approved: {result['approved']}")
    print(f"Message: {result['messages'][-1].content}")

    # --- Step 3: Resume with denial (different thread) ---
    print("\n=== Step 3: Deny (different thread) ===")
    config2: RunnableConfig = {"configurable": {"thread_id": "deny-test"}}

    partial = app.invoke(
        {"messages": [HumanMessage(content="Delete old_cache.py")], "approved": False},
        config2,
    )
    if "__interrupt__" in partial:
        for i in partial["__interrupt__"]:
            print(f"Graph interrupted: {i.value}")
        print()

    result2 = app.invoke(Command(resume={"approved": False}), config2)
    print(f"Approved: {result2['approved']}")
    print(f"Message: {result2['messages'][-1].content}")


if __name__ == "__main__":
    main()
