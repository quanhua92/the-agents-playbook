"""Prompter — decouple permission UI from agent logic.

The Prompter ABC defines how the agent asks for user confirmation.
Implementations handle different UI contexts (terminal, web, API)
without the agent loop knowing about them.
"""

from abc import ABC, abstractmethod

from .permissions import RiskLevel


class Prompter(ABC):
    """Abstract prompter for user confirmation requests.

    Implementations decide HOW to ask the user. The agent loop
    only knows THAT it needs to ask.
    """

    @abstractmethod
    async def confirm(
        self, message: str, risk: RiskLevel = RiskLevel.WORKSPACE_WRITE
    ) -> bool:
        """Ask the user to confirm an action.

        Args:
            message: Description of the action to confirm.
            risk: Risk level of the action being confirmed.

        Returns:
            True if the user approves, False otherwise.
        """
        ...


class TerminalPrompter(Prompter):
    """Reads confirmation from stdin, writes prompts to stdout.

    Usage in tests:
        prompter = TerminalPrompter(input_fn=AsyncMock(return_value="y"))
    """

    def __init__(self, input_fn=None) -> None:
        self._input_fn = input_fn

    async def confirm(
        self, message: str, risk: RiskLevel = RiskLevel.WORKSPACE_WRITE
    ) -> bool:
        risk_label = risk.value.upper()
        prompt = f"[{risk_label}] {message} (y/n): "

        if self._input_fn:
            answer = await self._input_fn(prompt)
        else:
            answer = input(prompt)

        return answer.strip().lower().startswith("y")


class SilentPrompter(Prompter):
    """Auto-approves everything. Used in tests and headless mode."""

    async def confirm(
        self, message: str, risk: RiskLevel = RiskLevel.WORKSPACE_WRITE
    ) -> bool:
        return True


class DenyAllPrompter(Prompter):
    """Auto-denies everything. Useful for testing refusal behavior."""

    async def confirm(
        self, message: str, risk: RiskLevel = RiskLevel.WORKSPACE_WRITE
    ) -> bool:
        return False
