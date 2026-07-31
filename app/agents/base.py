"""Agent execution loop over the Claude API.

Hand-written instead of using the SDK's tool runner for two reasons: the tool
runner is a beta API, and the loop is precisely the part of the project worth
showing — keeping it explicit costs about seventy lines and makes the flow
auditable.

The loop is the same for both agents; what changes is the tool set and the
prompt. Each agent declares a *terminal tool*: when the model calls it, the loop
ends and its arguments are the structured result. That trades free-text parsing
for a contract validated by the tool's own schema, and gives a deterministic
stopping criterion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """The agent could not produce a usable result."""


class BudgetExceeded(AgentError):
    """The request's LLM call budget was reached."""


class LLMBudget:
    """Cap on LLM calls per HTTP request.

    Without it, an adversarially built batch — many IPs, each with one suspicious
    event — would produce one incident per IP and turn a single request into
    hundreds of model calls. The excess falls back to the deterministic path.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"budget of {self.limit} LLM calls reached")
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MessagesClient(Protocol):
    """The minimal surface of the Anthropic client used here.

    Declared as a Protocol so tests can inject a double without depending on the
    SDK or on the network.
    """

    def create(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class AgentResult:
    output: dict[str, Any]
    iterations: int
    tool_calls: list[str]
    llm_calls: int


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        tools: list[Tool],
        terminal_tool: str,
        max_tokens: int = 8_000,
        effort: str = "medium",
        max_iterations: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.terminal_tool = terminal_tool
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_iterations = max_iterations

        if terminal_tool not in self.tools:
            raise ValueError(f"terminal tool {terminal_tool!r} was not declared")

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # Tool definitions and the system prompt are identical across every
            # call; the cache breakpoint at the end of the system block covers both,
            # since tools render before system.
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[tool.definition() for tool in self.tools.values()],
            messages=messages,
        )

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool. Returns (content, is_error)."""
        tool = self.tools.get(name)
        if tool is None:
            return f"unknown tool: {name}", True
        try:
            result = tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - the error goes back to the model
            logger.warning("tool %s failed: %s", name, exc)
            return f"the tool failed: {exc}", True
        if isinstance(result, str):
            return result, False
        return json.dumps(result, ensure_ascii=False, default=str), False

    def run(self, user_message: str, budget: LLMBudget | None = None) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        called: list[str] = []
        llm_calls = 0

        for iteration in range(1, self.max_iterations + 1):
            if budget is not None:
                budget.consume()
            response = self._request(messages)
            llm_calls += 1

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "refusal":
                raise AgentError("the request was refused by the model filters")

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [
                block for block in response.content
                if getattr(block, "type", None) == "tool_use"
            ]
            if not tool_uses:
                if stop_reason == "max_tokens":
                    raise AgentError("response truncated by max_tokens")
                raise AgentError(
                    f"the agent finished without calling {self.terminal_tool!r}"
                )

            results = []
            for block in tool_uses:
                called.append(block.name)
                if block.name == self.terminal_tool:
                    return AgentResult(
                        output=dict(block.input),
                        iterations=iteration,
                        tool_calls=called,
                        llm_calls=llm_calls,
                    )
                content, is_error = self._dispatch(block.name, dict(block.input))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        raise AgentError(
            f"reached the limit of {self.max_iterations} iterations without calling "
            f"{self.terminal_tool!r}"
        )
