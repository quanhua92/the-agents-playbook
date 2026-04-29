"""01_plan_execute.py -- Plan-and-execute pattern (native LangGraph).

In the root project:
  workflow = Workflow(steps=[plan_step, build_step])
  errors = workflow.validate()  # DAG validation (Kahn's algorithm)
  async for event in workflow.run("Fix the bug"):
      if event.type == "step_completed": ...

In LangGraph:
  graph = StateGraph(PlanExecuteState)
  graph.add_node("planner", planner)
  graph.add_node("executor", executor)
  graph.add_conditional_edges("executor", should_continue, ...)
  # No manual DAG validation -- edges ARE the DAG

The graph naturally encodes the plan-execute loop: plan -> execute -> check
if more steps -> loop back or summarize.
"""

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class PlanExecuteState(dict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    steps: list[str]
    current_step: int
    results: list[str]


def planner(state: PlanExecuteState) -> dict:
    """Generate a plan as a numbered list of steps."""
    task = state["messages"][-1].content
    steps = [
        "Research the topic and gather requirements",
        "Draft an outline",
        "Write the content following the outline",
        "Review and refine the final output",
    ]
    plan = f"Plan for: {task}\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    print(f"  [PLANNER] Generated plan with {len(steps)} steps")
    return {"plan": plan, "steps": steps, "current_step": 0, "results": []}


def executor(state: PlanExecuteState) -> dict:
    """Execute the current step."""
    idx = state["current_step"]
    step = state["steps"][idx]
    result = f"[Executed] {step}"
    results = state.get("results", []) + [result]
    print(f"  [EXECUTOR] Step {idx + 1}/{len(state['steps'])}: {step}")
    return {"current_step": idx + 1, "results": results}


def should_continue(state: PlanExecuteState) -> str:
    """Check if all steps are done."""
    if state["current_step"] < len(state["steps"]):
        return "executor"
    return "summarizer"


def summarizer(state: PlanExecuteState) -> dict:
    """Summarize execution results."""
    summary = (
        f"Completed {len(state['results'])} steps:\n"
        + "\n".join(f"- {r}" for r in state["results"])
    )
    return {"messages": [AIMessage(content=summary)]}


def main():
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {"executor": "executor", "summarizer": "summarizer"},
    )
    graph.add_edge("summarizer", END)

    app = graph.compile()

    print("=== Plan-and-Execute Workflow ===\n")
    result = app.invoke({
        "messages": [HumanMessage(content="Write a blog post about LangGraph")],
        "plan": "",
        "steps": [],
        "current_step": 0,
        "results": [],
    })

    print(f"\n=== Plan ===\n{result['plan']}")

    print(f"\n=== Results ===")
    for r in result["results"]:
        print(f"  {r}")

    print(f"\n=== Summary ===\n{result['messages'][-1].content}")

    print("\n=== Root Comparison ===")
    print("Root: Workflow(DAG) + Kahn's algorithm + asyncio.gather + hooks")
    print("Here: StateGraph + conditional_edges (loop back or finish)")
    print("The graph topology IS the workflow -- no separate DAG validation needed.")


if __name__ == "__main__":
    main()
