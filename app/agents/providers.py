"""Provider adapters for the agent loop.

The loop in `app.agents.base` is written against the Anthropic Messages surface:
`client.messages.create(...)` returning an object with `.stop_reason` and a
`.content` list of blocks carrying `.type`, `.name`, `.input` and `.id`.

That surface is a seam, not a dependency. Anything that speaks it can drive the
agents, so supporting another provider means writing an adapter rather than
touching the loop, the tools or the prompts.

`GeminiClient` is such an adapter, over the Gemini REST API. The translation is
bidirectional and the outgoing half is the awkward one: the loop builds its message
history in Anthropic shape, so every call re-renders that history into Gemini
`contents`.

| loop sends | Gemini expects |
|---|---|
| `system=[{type, text, cache_control}]` | `systemInstruction.parts[].text` |
| `tools=[{name, description, input_schema}]` | `tools[].function_declarations[]` |
| `{"role": "assistant", "content": [blocks]}` | `{"role": "model", "parts": [...]}` |
| `{"type": "tool_result", "tool_use_id": ...}` | `parts[].functionResponse` |

Gemini function calls carry no call id, while the loop needs one to match a result
back to its call. The adapter synthesizes ids that embed the function name
(`call::name::n`) so the mapping stays stateless — no per-conversation bookkeeping
that could leak between concurrent requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 120.0
ID_SEPARATOR = "::"

# Gemini reports why generation stopped with its own vocabulary. Only three
# outcomes matter to the loop: it called a tool, it stopped talking, or it was cut
# off / refused.
FINISH_REASONS = {
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
    "SPII": "refusal",
    "RECITATION": "refusal",
}


@dataclass(slots=True)
class Block:
    """One content block, shaped like an Anthropic response block.

    `signature` is the extra field Gemini needs and Anthropic does not: from
    Gemini 3 onward, a function call carries a `thoughtSignature` that must be
    echoed back verbatim when the conversation continues, or the API rejects the
    next request. It rides along here so the agent loop never has to know.
    """

    type: str
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    signature: str = ""


@dataclass(slots=True)
class Response:
    content: list[Block]
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)


def _tool_id(name: str, index: int) -> str:
    return f"call{ID_SEPARATOR}{name}{ID_SEPARATOR}{index}"


def _name_from_id(tool_id: str) -> str:
    parts = tool_id.split(ID_SEPARATOR)
    return parts[1] if len(parts) >= 3 else tool_id


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce a JSON Schema to the subset Gemini accepts.

    Gemini rejects `additionalProperties` and `$schema`, and rejects an object
    declaring no properties at all — which several of our read-only tools do, since
    every argument is optional.
    """
    allowed = {"type", "description", "enum", "properties", "required", "items"}
    cleaned = {key: value for key, value in schema.items() if key in allowed}
    properties = cleaned.get("properties")
    if cleaned.get("type") == "object" and not properties:
        return None
    if isinstance(properties, dict):
        cleaned["properties"] = {
            key: _clean_schema(value) or {"type": "string"}
            for key, value in properties.items()
        }
    return cleaned


def _render_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the loop's Anthropic-shaped history into Gemini `contents`."""
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            contents.append({"role": "user", "parts": [{"text": content}]})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    part: dict[str, Any] = {
                        "functionCall": {
                            "name": getattr(block, "name", ""),
                            "args": dict(getattr(block, "input", {}) or {}),
                        }
                    }
                    signature = getattr(block, "signature", "")
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif block_type == "text" and getattr(block, "text", ""):
                    parts.append({"text": block.text})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        # Tool results come back as a user turn holding tool_result blocks.
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(
                    {
                        "functionResponse": {
                            "name": _name_from_id(str(block.get("tool_use_id", ""))),
                            "response": {"result": str(block.get("content", ""))},
                        }
                    }
                )
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append({"text": block.get("text", "")})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return contents


class _Messages:
    """Implements the single method the agent loop calls."""

    def __init__(self, api_key: str, base_url: str, client: httpx.Client) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int = 8_000,
        system: list[dict[str, Any]] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        **ignored: Any,
    ) -> Response:
        """Anthropic-shaped call, Gemini-shaped request underneath.

        `thinking` and `output_config` arrive from the loop and are dropped: they are
        Anthropic parameters, and Gemini would reject the request rather than ignore
        them.
        """
        body: dict[str, Any] = {
            "contents": _render_contents(messages or []),
            "generationConfig": {"maxOutputTokens": max_tokens},
        }

        if system:
            text = (
                system
                if isinstance(system, str)
                else "\n".join(block.get("text", "") for block in system)
            )
            body["systemInstruction"] = {"parts": [{"text": text}]}

        if tools:
            declarations = []
            for tool in tools:
                declaration: dict[str, Any] = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                schema = _clean_schema(tool.get("input_schema", {}))
                if schema:
                    declaration["parameters"] = schema
                declarations.append(declaration)
            body["tools"] = [{"function_declarations": declarations}]
            # Without this the model tends to answer in prose; the loop treats a
            # response with no tool call as an error, and rightly so.
            body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}

        name = model if model.startswith("models/") else f"models/{model}"
        response = self._client.post(
            f"{self._base_url}/{name}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini returned {response.status_code}: {response.text[:400]}"
            )
        return self._parse(response.json())

    @staticmethod
    def _parse(payload: dict[str, Any]) -> Response:
        candidates = payload.get("candidates") or []
        if not candidates:
            feedback = payload.get("promptFeedback", {})
            reason = feedback.get("blockReason", "no candidate returned")
            return Response(content=[], stop_reason="refusal", usage={"blocked": reason})

        candidate = candidates[0]
        finish = str(candidate.get("finishReason", "STOP")).upper()
        blocks: list[Block] = []
        for index, part in enumerate(candidate.get("content", {}).get("parts", [])):
            call = part.get("functionCall")
            if call:
                # Gemini sometimes namespaces the call under a synthetic module.
                name = call.get("name", "").split(":")[-1]
                blocks.append(
                    Block(
                        type="tool_use",
                        id=_tool_id(name, index),
                        name=name,
                        input=dict(call.get("args") or {}),
                        signature=part.get("thoughtSignature", ""),
                    )
                )
            elif part.get("text"):
                blocks.append(Block(type="text", text=part["text"]))

        has_tool_use = any(block.type == "tool_use" for block in blocks)
        stop_reason = FINISH_REASONS.get(
            finish, "tool_use" if has_tool_use else "end_turn"
        )
        return Response(
            content=blocks,
            stop_reason=stop_reason,
            usage=payload.get("usageMetadata", {}),
        )


class GeminiClient:
    """Drop-in replacement for the Anthropic client, backed by the Gemini API."""

    def __init__(self, api_key: str, base_url: str = GEMINI_BASE_URL) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to build a Gemini client")
        self._http = httpx.Client()
        self.messages = _Messages(api_key, base_url, self._http)

    def close(self) -> None:  # pragma: no cover - lifecycle helper
        self._http.close()
