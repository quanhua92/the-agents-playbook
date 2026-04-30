"""04_llm_judge.py -- LLM-as-Judge evaluation for LangGraph agents.

Instead of brittle substring matching, use a second LLM call to evaluate
agent output quality with an explicit rubric. This enables evaluating
open-ended tasks where there's no single correct answer.

Pattern:
  1. Run the agent on a task
  2. Collect the final response
  3. Send task + response + rubric to a judge LLM
  4. Parse structured scores from the judge's response
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared import get_openai_llm


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a given topic. Returns a brief summary."""
    facts = {
        "python": "Python is a high-level programming language created by Guido van Rossum, first released in 1991.",
        "rust": "Rust is a systems programming language focused on safety, concurrency, and performance, created by Mozilla.",
        "async": "Async/await is a programming pattern for writing non-blocking code using coroutines and an event loop.",
    }
    return facts.get(topic.lower(), f"No specific fact found about '{topic}'.")


def run_agent_and_collect(agent, task: str) -> str:
    """Run a LangGraph agent and return its final response text."""
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    return result["messages"][-1].content


def judge_with_llm(llm, task: str, response: str, criteria: dict[str, str]) -> dict:
    """Use an LLM to judge the quality of an agent's response.

    Returns a dict with 'scores' and 'reasoning' keys.
    """
    criteria_text = "\n".join(
        f"- {name}: {desc} (0.0-1.0)"
        for name, desc in criteria.items()
    )

    judge_prompt = (
        f"## Task\n{task}\n\n"
        f"## Agent Response\n{response}\n\n"
        f"## Evaluation Criteria\n{criteria_text}\n\n"
        "Evaluate the response. Return JSON with:\n"
        '{"scores": {"criterion": 0.0}, "reasoning": "...", "overall": 0.0}'
    )

    from langchain_core.messages import SystemMessage
    result = llm.invoke([
        SystemMessage(content="You are an impartial evaluator. Return ONLY valid JSON."),
        HumanMessage(content=judge_prompt),
    ])

    import json
    try:
        text = result.content
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith(("json", "JSON")):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return {"scores": {}, "reasoning": "Failed to parse judge response", "overall": 0.0}


def main():
    print("=== LLM-as-Judge Evaluation ===\n")

    llm = get_openai_llm()

    # Build a ReAct agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[calculate, lookup_fact],
    )

    # Define evaluation criteria
    criteria = {
        "accuracy": "Are the facts and calculations correct?",
        "completeness": "Does the response fully address the task?",
        "clarity": "Is the response clear and well-structured?",
    }

    # Define tasks to evaluate
    tasks = [
        {
            "task": "What is 15 * 23? Show your work.",
            "description": "Math task — factual correctness check",
        },
        {
            "task": "Tell me about Python and Rust programming languages.",
            "description": "Knowledge task — completeness and accuracy check",
        },
        {
            "task": "Explain async programming and give me a calculation: 100 / 7.",
            "description": "Multi-step task — both tool use and knowledge",
        },
    ]

    from langchain_core.tools import tool as lc_tool

    for task_info in tasks:
        task = task_info["task"]
        description = task_info["description"]

        print(f"--- {description} ---")
        print(f"Task: {task}\n")

        # Run the agent
        response = run_agent_and_collect(agent, task)
        print(f"Agent response: {response[:200]}...\n")

        # Judge the response
        judge_result = judge_with_llm(llm, task, response, criteria)
        print("Judge evaluation:")
        overall = judge_result.get("overall", 0)
        for criterion, score in judge_result.get("scores", {}).items():
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            print(f"  {criterion:15s} [{bar}] {score:.2f}")
        overall_bar = "#" * int(overall * 20) + "-" * (20 - int(overall * 20))
        print(f"  {'overall':15s} [{overall_bar}] {overall:.2f}")
        print(f"  Reasoning: {judge_result.get('reasoning', 'N/A')[:150]}")
        print()


if __name__ == "__main__":
    main()
