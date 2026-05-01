"""Tests for guardrails.ask_user — AskUserQuestion tool."""

from the_agents_playbook.guardrails.ask_user import AskUserQuestion
from the_agents_playbook.guardrails.prompter import DenyAllPrompter, SilentPrompter


class TestAskUserQuestion:
    def test_tool_protocol(self):
        tool = AskUserQuestion()
        assert tool.name == "ask_user"
        assert (
            "question" in tool.description.lower()
            or "clarification" in tool.description.lower()
        )
        assert "question" in tool.parameters["required"]

    def test_parameters_schema(self):
        tool = AskUserQuestion()
        props = tool.parameters["properties"]
        assert "question" in props
        assert "options" in props
        assert props["question"]["type"] == "string"
        assert props["options"]["type"] == "array"

    async def test_execute_basic_question(self):
        tool = AskUserQuestion(prompter=SilentPrompter())
        result = await tool.execute(question="What is your name?")
        assert result.error is False
        assert "What is your name?" in result.output

    async def test_execute_with_options(self):
        tool = AskUserQuestion(prompter=SilentPrompter())
        result = await tool.execute(
            question="Which file?",
            options=["auth.py", "user.py"],
        )
        assert result.error is False
        assert "options" in result.output

    async def test_execute_user_declines(self):
        tool = AskUserQuestion(prompter=DenyAllPrompter())
        result = await tool.execute(question="Continue?")
        assert result.error is True
        assert "declined" in result.output.lower()

    async def test_default_prompter_is_silent(self):
        tool = AskUserQuestion()
        result = await tool.execute(question="test")
        assert result.error is False
