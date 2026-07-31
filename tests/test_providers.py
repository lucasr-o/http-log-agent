"""Tests for the Gemini adapter's translation, without touching the network.

The adapter is the riskiest piece of provider support, because the rest of the
suite cannot catch a bug in it: `FakeClient` is already Anthropic-shaped, so every
agent test would pass against a completely broken adapter. These tests exercise the
translation itself, in both directions, against payloads copied from real Gemini
responses.
"""

from __future__ import annotations

import pytest

from app.agents.providers import (
    Block,
    GeminiClient,
    _clean_schema,
    _name_from_id,
    _render_contents,
    _tool_id,
)


class TestToolIds:
    def test_the_id_carries_the_function_name(self):
        """Gemini sends no call id, so the loop's id has to be self-describing.

        Keeping the name inside the id makes the mapping stateless — no
        per-conversation bookkeeping that could leak between concurrent requests.
        """
        assert _name_from_id(_tool_id("get_ip_history", 2)) == "get_ip_history"

    def test_an_unknown_id_shape_returns_itself(self):
        assert _name_from_id("whatever") == "whatever"


class TestSchemaCleaning:
    def test_unsupported_keys_are_dropped(self):
        cleaned = _clean_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {"ip": {"type": "string", "description": "target"}},
                "required": ["ip"],
            }
        )
        assert cleaned == {
            "type": "object",
            "properties": {"ip": {"type": "string", "description": "target"}},
            "required": ["ip"],
        }

    def test_an_object_without_properties_becomes_none(self):
        """Gemini rejects a parameter object declaring no properties at all."""
        assert _clean_schema({"type": "object", "properties": {}}) is None

    def test_enums_and_nested_items_survive(self):
        cleaned = _clean_schema(
            {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": ["status", "method"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            }
        )
        assert cleaned["properties"]["field"]["enum"] == ["status", "method"]
        assert cleaned["properties"]["tags"]["items"] == {"type": "string"}


class TestOutgoingTranslation:
    def test_a_plain_user_turn_becomes_a_text_part(self):
        contents = _render_contents([{"role": "user", "content": "triage this"}])
        assert contents == [{"role": "user", "parts": [{"text": "triage this"}]}]

    def test_the_assistant_turn_becomes_a_model_turn(self):
        block = Block(type="tool_use", id=_tool_id("count_by_field", 0),
                      name="count_by_field", input={"field": "status"})
        contents = _render_contents([{"role": "assistant", "content": [block]}])
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"][0]["functionCall"] == {
            "name": "count_by_field",
            "args": {"field": "status"},
        }

    def test_the_thought_signature_is_echoed_back(self):
        """Gemini 3 rejects the next request when the signature is not returned.

        This is the one field Anthropic has no equivalent for, and dropping it cost a
        400 on the first real run.
        """
        block = Block(type="tool_use", id=_tool_id("t", 0), name="t", signature="sig-abc")
        parts = _render_contents([{"role": "assistant", "content": [block]}])[0]["parts"]
        assert parts[0]["thoughtSignature"] == "sig-abc"

    def test_a_block_without_a_signature_omits_the_field(self):
        block = Block(type="tool_use", id=_tool_id("t", 0), name="t")
        parts = _render_contents([{"role": "assistant", "content": [block]}])[0]["parts"]
        assert "thoughtSignature" not in parts[0]

    def test_a_tool_result_becomes_a_function_response(self):
        contents = _render_contents(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": _tool_id("get_ip_history", 1),
                            "content": '{"previous_incidents": 0}',
                            "is_error": False,
                        }
                    ],
                }
            ]
        )
        response = contents[0]["parts"][0]["functionResponse"]
        assert response["name"] == "get_ip_history"
        assert response["response"] == {"result": '{"previous_incidents": 0}'}

    def test_an_empty_assistant_turn_is_skipped(self):
        assert _render_contents([{"role": "assistant", "content": []}]) == []


class TestIncomingTranslation:
    def _parse(self, payload):
        from app.agents.providers import _Messages

        return _Messages._parse(payload)

    def test_a_function_call_becomes_a_tool_use_block(self):
        response = self._parse(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "submit_verdict",
                                        "args": {"verdict": "malicious"},
                                    },
                                    "thoughtSignature": "sig-1",
                                }
                            ]
                        },
                    }
                ]
            }
        )
        assert response.stop_reason == "tool_use"
        block = response.content[0]
        assert block.type == "tool_use"
        assert block.name == "submit_verdict"
        assert block.input == {"verdict": "malicious"}
        assert block.signature == "sig-1"

    def test_a_namespaced_call_name_is_stripped(self):
        """Gemini sometimes prefixes the call with a synthetic module name."""
        response = self._parse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "default_api:block_ip", "args": {}}}
                            ]
                        }
                    }
                ]
            }
        )
        assert response.content[0].name == "block_ip"

    def test_text_only_response_ends_the_turn(self):
        response = self._parse(
            {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "hi"}]}}]}
        )
        assert response.stop_reason == "end_turn"
        assert response.content[0].type == "text"

    def test_truncation_maps_to_max_tokens(self):
        response = self._parse(
            {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}
        )
        assert response.stop_reason == "max_tokens"

    def test_a_safety_stop_maps_to_refusal(self):
        response = self._parse(
            {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}
        )
        assert response.stop_reason == "refusal"

    def test_a_blocked_prompt_with_no_candidate_maps_to_refusal(self):
        """The loop must see a refusal, not an IndexError."""
        response = self._parse({"promptFeedback": {"blockReason": "OTHER"}})
        assert response.stop_reason == "refusal"
        assert response.content == []


class TestClientConstruction:
    def test_an_empty_key_is_refused(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient("")

    def test_the_client_exposes_the_messages_surface(self):
        """What the agent loop needs is exactly `client.messages.create`."""
        client = GeminiClient("fake-key")
        assert callable(client.messages.create)
        client.close()


class TestProviderSelection:
    def test_gemini_is_selected_when_only_its_key_is_present(self):
        from app.config import Settings

        settings = Settings(anthropic_api_key="", gemini_api_key="k", llm_provider="auto")
        assert settings.active_provider == "gemini"
        assert settings.active_model == "gemini-3.5-flash"

    def test_anthropic_wins_when_both_keys_are_present(self):
        from app.config import Settings

        settings = Settings(anthropic_api_key="a", gemini_api_key="g", llm_provider="auto")
        assert settings.active_provider == "anthropic"
        assert settings.active_model == "claude-opus-5"

    def test_an_explicit_provider_without_its_key_disables_the_agents(self):
        from app.config import Settings

        settings = Settings(anthropic_api_key="a", gemini_api_key="", llm_provider="gemini")
        assert settings.active_provider == ""
        assert settings.llm_enabled is False

    def test_an_explicit_model_overrides_the_provider_default(self):
        from app.config import Settings

        settings = Settings(gemini_api_key="k", llm_provider="gemini", llm_model="gemini-3.6-flash")
        assert settings.active_model == "gemini-3.6-flash"
