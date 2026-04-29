"""05-tool-chaining.py — Multi-step tool chains with re-scoring.

ToolChainer executes sequential tool calls, re-scoring after each step.
The chain stops when entropy drops below threshold (clear next step),
a tool returns an error, or max chain length is reached.

This is what separates "an agent that can call a tool" from
"an agent that can accomplish multi-step tasks."
"""

import asyncio

from the_agents_playbook.loop import ToolChain, ToolChainer, score_tools
from the_agents_playbook.tools import Tool, ToolResult, ToolRegistry


class AddTool(Tool):
    """Adds two numbers."""

    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Add two numbers."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        }

    async def execute(self, a: float, b: float, **kwargs) -> ToolResult:
        return ToolResult(output=str(a + b))


class DoubleTool(Tool):
    """Doubles a number."""

    @property
    def name(self) -> str:
        return "double"

    @property
    def description(self) -> str:
        return "Double a number."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"],
        }

    async def execute(self, n: float, **kwargs) -> ToolResult:
        return ToolResult(output=str(n * 2))


class FailTool(Tool):
    """Always fails — for testing error handling."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="Something went wrong", error=True)


async def main():
    registry = ToolRegistry()
    registry.register(AddTool())
    registry.register(DoubleTool())
    registry.register(FailTool())

    # --- Basic chain ---

    chainer = ToolChainer(registry, max_chain_length=3, entropy_threshold=1.0)

    print("=== Single-step chain ===")
    chain = await chainer.execute_chain({
        "tool_name": "add",
        "arguments": {"a": 10, "b": 20},
    })
    print(f"  Steps:    {len(chain.steps)}")
    print(f"  Output:   {chain.final_output}")
    print(f"  Confidence: {chain.confidence}")
    print()

    # --- Chain with error ---

    print("=== Chain with error ===")
    chain = await chainer.execute_chain({
        "tool_name": "fail",
        "arguments": {},
    })
    print(f"  Steps:    {len(chain.steps)}")
    print(f"  Output:   {chain.final_output}")
    print(f"  Confidence: {chain.confidence}")
    print()

    # --- should_chain decisions ---

    print("=== Chain decisions ===")
    ok_result = ToolResult(output="success")
    err_result = ToolResult(output="fail", error=True)

    # Low entropy → stop chaining
    print(f"  Low entropy (0.3):  should_chain(ok) = {chainer.should_chain(ok_result, 0.3)}")
    # High entropy → continue chaining
    print(f"  High entropy (2.0): should_chain(ok) = {chainer.should_chain(ok_result, 2.0)}")
    # Error → always stop
    print(f"  Error result:       should_chain(err) = {chainer.should_chain(err_result, 2.0)}")
    print()

    # --- Entropy-aware tool scoring ---

    print("=== Entropy-based routing ===")
    tool_scores = {"add": 0.8, "double": 0.15, "fail": 0.05}
    entropy = score_tools(tool_scores)
    print(f"  Scores:    {tool_scores}")
    print(f"  Entropy:   {entropy:.4f} bits")
    print(f"  Threshold: {chainer.max_chain_length}")

    if entropy < 1.0:
        print(f"  → Low uncertainty: proceed with top tool (add)")
    else:
        print(f"  → High uncertainty: consider asking user")


if __name__ == "__main__":
    asyncio.run(main())
