"""Tests for loop.chains — ToolChain and ToolChainer."""

import pytest

from the_agents_playbook.loop.chains import ToolChain, ToolChainer
from the_agents_playbook.tools.protocol import Tool, ToolResult
from the_agents_playbook.tools.registry import ToolRegistry


class EchoTool(Tool):
    """A simple tool that returns its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes back the input."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=kwargs.get("message", ""))


class FailTool(Tool):
    """A tool that always fails."""

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
        return ToolResult(output="error", error=True)


class TestToolChain:
    def test_defaults(self):
        chain = ToolChain()
        assert chain.steps == []
        assert chain.final_output is None
        assert chain.confidence == 1.0

    def test_with_steps(self):
        chain = ToolChain(
            steps=[
                {"tool_name": "echo", "arguments": {"msg": "hi"}, "result": ToolResult(output="hi")},
            ],
            final_output="hi",
            confidence=0.9,
        )
        assert len(chain.steps) == 1
        assert chain.final_output == "hi"
        assert chain.confidence == 0.9


class TestToolChainer:
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(FailTool())
        return reg

    @pytest.fixture
    def chainer(self, registry):
        return ToolChainer(registry, max_chain_length=3, entropy_threshold=1.0)

    def test_max_chain_length(self, chainer):
        assert chainer.max_chain_length == 3

    def test_should_chain_stops_on_error(self, chainer):
        result = ToolResult(output="failed", error=True)
        assert chainer.should_chain(result, 2.0) is False

    def test_should_chain_stops_on_low_entropy(self, chainer):
        result = ToolResult(output="ok")
        assert chainer.should_chain(result, 0.3) is False  # below threshold 1.0

    def test_should_chain_continues_on_high_entropy(self, chainer):
        result = ToolResult(output="ok")
        assert chainer.should_chain(result, 2.0) is True  # above threshold 1.0

    async def test_execute_chain_single_step(self, chainer):
        chain = await chainer.execute_chain({
            "tool_name": "echo",
            "arguments": {"message": "hello"},
        })
        assert len(chain.steps) == 1
        assert chain.final_output == "hello"
        assert chain.steps[0]["tool_name"] == "echo"

    async def test_execute_chain_with_error_tool(self, chainer):
        chain = await chainer.execute_chain({
            "tool_name": "fail",
            "arguments": {},
        })
        assert len(chain.steps) == 1
        assert chain.steps[0]["result"].error is True
        assert chain.confidence == 0.0

    async def test_execute_chain_unknown_tool(self, chainer):
        chain = await chainer.execute_chain({
            "tool_name": "nonexistent",
            "arguments": {},
        })
        assert len(chain.steps) == 1
        assert chain.steps[0]["result"].error is True
        assert chain.confidence == 0.0

    async def test_execute_chain_respects_max_length(self, registry):
        chainer = ToolChainer(registry, max_chain_length=1)
        chain = await chainer.execute_chain({
            "tool_name": "echo",
            "arguments": {"message": "test"},
        })
        assert len(chain.steps) == 1

    async def test_should_chain_no_scores_stops_on_error(self, chainer):
        result = ToolResult(output="err", error=True)
        assert chainer.should_chain(result, 0.0) is False
