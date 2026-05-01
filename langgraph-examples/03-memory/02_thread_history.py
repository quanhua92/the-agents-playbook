"""02_thread_history.py -- Thread-based conversation management.

In the root project, separate sessions are stored as separate JSONL files.
In LangGraph, thread_id replaces file paths -- each thread is an independent
conversation context backed by the checkpointer.

This example shows multiple users/threads managed simultaneously.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from shared import get_openai_llm


def main():
    llm = get_openai_llm()
    checkpointer = MemorySaver()
    agent = create_agent(llm, [], checkpointer=checkpointer)

    threads = {
        "alice": [
            "I'm Alice, I'm a data scientist.",
            "What's my name and what do I do?",
        ],
        "bob": [
            "I'm Bob, I work as a chef.",
            "What do I do for a living?",
        ],
    }

    for thread_id, messages in threads.items():
        print(f"=== Thread: {thread_id} ===")
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        for msg in messages:
            result = agent.invoke(
                {"messages": [HumanMessage(content=msg)]},
                config,
            )
            print(f"  Q: {msg}")
            answer = result["messages"][-1].content
            print(f"  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print()

    # Cross-thread isolation check
    print("=== Cross-Thread Isolation ===")
    config_alice: RunnableConfig = {"configurable": {"thread_id": "alice"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="Do you know anyone named Bob?")]},
        config_alice,
    )
    print(f"  Alice's thread: {result['messages'][-1].content[:120]}...")
    print("  (Alice's thread should NOT know about Bob)")


if __name__ == "__main__":
    main()
