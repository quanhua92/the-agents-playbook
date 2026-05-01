"""03_draft_approval.py -- Draft/approval pattern with LangGraph interrupt().

Workers stage actions as drafts instead of executing them directly.
The graph pauses (interrupt) to show the draft for human review,
then resumes with the approval decision.

Pattern:
  1. Worker agent decides to send an email
  2. Worker creates a draft instead of sending
  3. Graph interrupts for human approval
  4. Human approves or rejects via Command(resume=...)
  5. Graph proceeds based on the decision
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft_approved: bool | None


def worker_node(state: State) -> dict:
    """Worker proposes an action as a draft for review."""
    # In a real agent, this would involve tool calls.
    # Here we simulate the worker deciding to send an email.

    draft = {
        "kind": "email",
        "summary": "Send project update to stakeholders",
        "to": "stakeholders@example.com",
        "subject": "Weekly Project Update",
        "body": "This week we completed the API redesign and started load testing.",
    }

    # Interrupt for human approval
    decision = interrupt(
        {
            "question": "Worker wants to send an email. Approve?",
            "draft": draft,
        }
    )

    # decision is the value passed via Command(resume=...)
    if isinstance(decision, str) and decision.lower() == "approve":
        return {
            "messages": [HumanMessage(content="Email sent successfully!")],
            "draft_approved": True,
        }
    else:
        return {
            "messages": [HumanMessage(content="Draft was rejected.")],
            "draft_approved": False,
        }


def summarizer_node(state: State) -> dict:
    """Summarize the outcome."""
    if state.get("draft_approved"):
        return {
            "messages": [
                HumanMessage(content="Summary: Action was approved and executed.")
            ]
        }
    else:
        return {
            "messages": [HumanMessage(content="Summary: Action was rejected by human.")]
        }
    return {}


def build_graph():
    """Build the draft approval graph."""
    graph = StateGraph(State)
    graph.add_node("worker", worker_node)
    graph.add_node("summarizer", summarizer_node)

    graph.add_edge(START, "worker")
    graph.add_conditional_edges(
        "worker",
        lambda state: "summarizer",
    )
    graph.add_edge("summarizer", END)

    return graph.compile(checkpointer=MemorySaver())


def main():
    print("=== Draft/Approval with LangGraph interrupt() ===\n")

    graph = build_graph()

    # Step 1: Run the graph — it will interrupt at the worker node
    print("Step 1: Running graph (will pause for approval)...")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Send the weekly update")],
            "draft_approved": None,
        },
        config={"configurable": {"thread_id": "1"}},
    )
    print(f"Messages: {[m.content for m in result['messages']]}\n")

    # Step 2: Resume with approval
    print("Step 2: Resuming with approval...")
    result_approved = graph.invoke(
        Command(resume="approve"),
        config={"configurable": {"thread_id": "1"}},
    )
    print(f"Messages: {[m.content for m in result_approved['messages']]}\n")

    # Step 3: Resume with rejection (new thread)
    print("Step 3: New conversation, rejecting this time...")
    graph.invoke(
        {
            "messages": [HumanMessage(content="Send the weekly update")],
            "draft_approved": None,
        },
        config={"configurable": {"thread_id": "2"}},
    )
    result_rejected = graph.invoke(
        Command(resume="reject"),
        config={"configurable": {"thread_id": "2"}},
    )
    print(f"Messages: {[m.content for m in result_rejected['messages']]}\n")

    # Show the pattern explanation
    print("=== Pattern Summary ===\n")
    print("1. Worker node creates a draft action")
    print("2. interrupt() pauses the graph and returns the draft to the caller")
    print("3. Caller reviews the draft and decides")
    print("4. Command(resume='approve') or Command(resume='reject') continues")
    print("5. The graph proceeds based on the decision")
    print()
    print("This ensures no irreversible action is taken without explicit approval.")


if __name__ == "__main__":
    main()
