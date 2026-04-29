"""Agent — the core ReAct loop composing tools, memory, context, and provider.

The agent is not a "thing" — it's a process. It ties together everything
built so far (providers, tools, memory, context) into a recursive
decision-making loop: think → act → observe → repeat.

Usage:
    agent = Agent(
        provider=OpenAIProvider(settings=settings),
        registry=registry,
        memory=DualFileMemory(directory=Path(".memory")),
        context_builder=builder,
        config=AgentConfig(max_tool_iterations=10),
    )
    async for event in agent.run("Fix the bug in auth.py"):
        print(event)
    await agent.close()
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..context.builder import ContextBuilder
from ..memory.protocol import BaseMemoryProvider, Fact
from ..providers.base import BaseProvider
from ..providers.types import InputMessage, MessageRequest, ToolChoice
from ..tools.registry import ToolRegistry
from .chains import ToolChainer
from ..settings import settings as app_settings
from .config import AgentConfig
from .protocol import AgentEvent, TurnResult
from .scoring import score_tools

logger = logging.getLogger(__name__)


class Agent:
    """The core ReAct agent loop.

    Composes a provider, tool registry, memory, and context builder into
    a streaming autonomous loop that yields AgentEvent objects.

    The loop:
    1. Store user message in memory
    2. Recall relevant memories
    3. Build context (static + semi-stable + dynamic)
    4. Send to LLM with tool specs
    5. If LLM requests tool calls → dispatch → feed results back
    6. Repeat until LLM responds with text (no tool calls)
    7. Update memory with final response
    """

    def __init__(
        self,
        provider: BaseProvider,
        registry: ToolRegistry,
        memory: BaseMemoryProvider | None = None,
        context_builder: ContextBuilder | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._memory = memory
        self._context_builder = context_builder
        self._config = config or AgentConfig()
        self._chainer = ToolChainer(
            registry,
            max_chain_length=self._config.max_chain_length,
            entropy_threshold=self._config.entropy_threshold,
        )

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute the agent loop for a user prompt.

        Yields AgentEvent objects as the agent works through the ReAct loop.
        Each event carries a type (text, tool_call, tool_result, status, error)
        and associated data.

        Args:
            prompt: The user's message or task description.

        Yields:
            AgentEvent objects representing the agent's progress.
        """
        # 1. Store user message in memory
        if self._memory:
            await self._memory.store(Fact(
                content=prompt,
                source="user",
            ))

        # 2. Recall relevant memories
        memory_context = ""
        if self._memory:
            facts = await self._memory.recall(prompt, top_k=5)
            if facts:
                memory_context = "\n".join(f"- {f.content}" for f in facts)

        # 3. Build context
        system_prompt = ""
        if self._context_builder:
            if memory_context:
                from ..context.layers import ContextLayer, LayerPriority
                from ..context.builder import ContextBuilder
                # Add memory as a dynamic layer
                self._context_builder.add_dynamic(
                    ContextLayer(
                        name="memory",
                        content=f"Relevant memories:\n{memory_context}",
                        priority=LayerPriority.SEMI_STABLE,
                    )
                )
            system_prompt = self._context_builder.build()

        # 4. Start the ReAct loop
        messages: list[InputMessage] = [InputMessage(role="user", content=prompt)]
        tool_specs = self._registry.get_specs()

        for iteration in range(self._config.max_tool_iterations):
            yield AgentEvent(type="status", data={
                "message": f"Thinking (iteration {iteration + 1})...",
            })

            # Build request
            request = MessageRequest(
                model=app_settings.openai_model,
                system=system_prompt or "You are a helpful assistant with access to tools.",
                messages=messages,
                tools=tool_specs if tool_specs else [],
                tool_choice=ToolChoice(type="auto"),
            )

            # 5. Call provider
            try:
                response = await self._provider.send_message(request)
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Provider error in agent loop: %s", error_msg)

                if self._config.on_error == "raise":
                    raise
                elif self._config.on_error == "abort":
                    yield AgentEvent(type="error", data={"message": error_msg})
                    return
                else:  # yield_and_continue
                    yield AgentEvent(type="error", data={"message": error_msg})
                    return

            # 6. Check for tool calls
            tool_calls = response.message.tool_calls

            if not tool_calls:
                # LLM returned text — final response
                final_text = response.message.content or ""
                yield AgentEvent(type="text", data={"text": final_text})

                # 7. Update memory with final response
                if self._memory and final_text:
                    await self._memory.store(Fact(
                        content=final_text,
                        source="assistant",
                    ))
                return

            # Add assistant message with tool calls to conversation
            messages.append(InputMessage(
                role="assistant",
                content=response.message.content or "",
            ))

            # 7. Dispatch tool calls
            for call in tool_calls:
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                arguments = json.loads(fn.get("arguments", "{}"))

                yield AgentEvent(type="tool_call", data={
                    "tool_name": tool_name,
                    "arguments": arguments,
                })

                try:
                    result = await self._registry.dispatch(tool_name, arguments)
                except Exception as exc:
                    result_str = f"Error: {exc}"
                    yield AgentEvent(type="tool_result", data={
                        "output": result_str,
                        "error": True,
                    })
                    messages.append(InputMessage(
                        role="user",
                        content=f"Tool {tool_name} failed: {result_str}",
                    ))
                    continue

                yield AgentEvent(type="tool_result", data={
                    "output": result.output,
                    "error": result.error,
                })

                messages.append(InputMessage(
                    role="user",
                    content=result.output,
                ))

        # Hit max iterations
        yield AgentEvent(type="status", data={
            "message": f"Reached max tool iterations ({self._config.max_tool_iterations})",
        })

    async def run_turn(self, prompt: str) -> TurnResult:
        """Run the agent loop and collect all events into a TurnResult.

        Convenience method for cases where streaming is not needed.

        Args:
            prompt: The user's message or task description.

        Returns:
            TurnResult with all events and summary data.
        """
        events: list[AgentEvent] = []
        tool_calls_made = 0
        final_response = None
        error = None

        async for event in self.run(prompt):
            events.append(event)
            if event.type == "tool_call":
                tool_calls_made += 1
            elif event.type == "text":
                final_response = event.data.get("text")
            elif event.type == "error":
                error = event.data.get("message")

        return TurnResult(
            events=events,
            tool_calls_made=tool_calls_made,
            final_response=final_response,
            error=error,
        )

    async def close(self) -> None:
        """Clean up provider resources."""
        await self._provider.close()
