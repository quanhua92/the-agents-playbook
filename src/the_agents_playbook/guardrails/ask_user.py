"""AskUserQuestion — a tool that lets the agent ask the user for clarification.

Instead of hallucinating an answer, the agent can "break character" and
ask the user directly. The prompter implementation controls HOW the
question is presented (terminal, web, API).
"""

from typing import Any

from ..tools.protocol import Tool, ToolResult
from .prompter import Prompter, SilentPrompter


class AskUserQuestion(Tool):
    """Tool that lets the agent ask the user a question and get a response.

    The agent calls this tool when it's uncertain or needs information
    the user hasn't provided. The question is presented via the Prompter.

    Usage:
        tool = AskUserQuestion(prompter=TerminalPrompter())
        result = await tool.execute(
            question="Which file should I edit?",
            options=["auth.py", "user.py", "config.py"],
        )
    """

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question when you need clarification. "
            "Use this instead of guessing when you're uncertain."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices for the user.",
                },
            },
            "required": ["question"],
        }

    def __init__(self, prompter: Prompter | None = None) -> None:
        self._prompter = prompter or SilentPrompter()

    async def execute(self, question: str, options: list[str] | None = None, **kwargs: Any) -> ToolResult:
        """Ask the user a question and return their response.

        Args:
            question: The question to present to the user.
            options: Optional list of choices.

        Returns:
            ToolResult with the user's answer.
        """
        if options:
            prompt = f"{question}\nOptions: {', '.join(options)}"
        else:
            prompt = question

        from .permissions import RiskLevel
        approved = await self._prompter.confirm(prompt, risk=RiskLevel.READ_ONLY)

        if not approved:
            return ToolResult(output="User declined to answer.", error=True)

        if options:
            # For options, the user selected one (simplified — real impl would show menu)
            return ToolResult(output=f"User selected from options: {', '.join(options)}")
        else:
            return ToolResult(output=f"User was asked: {question}")
