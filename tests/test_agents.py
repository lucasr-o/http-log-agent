import pytest

from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.agents.action import ActionAgent
from app.agents.base import Agent, AgentError, BudgetExceeded, LLMBudget, Tool
from app.agents.triage import TriageAgent, build_tools
from tests.conftest import (
    VERDICT_BENIGN,
    VERDICT_MALICIOUS,
    FakeClient,
    FakeResponse,
    text_response,
    tool_use,
)


def terminal_tool(name: str = "submit") -> Tool:
    return Tool(
        name=name,
        description="finishes",
        input_schema={"type": "object", "properties": {}},
        handler=lambda arguments: arguments,
    )


class TestLLMBudget:
    def test_consumes_up_to_the_limit(self):
        budget = LLMBudget(2)
        budget.consume()
        budget.consume()
        assert budget.exhausted
        with pytest.raises(BudgetExceeded):
            budget.consume()

    def test_remaining_never_goes_negative(self):
        budget = LLMBudget(1)
        budget.consume()
        assert budget.remaining == 0


class TestAgentLoop:
    def test_terminal_tool_ends_the_loop_and_returns_its_arguments(self):
        client = FakeClient([tool_use("submit", {"verdict": "malicious"})])
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        result = agent.run("analyze")
        assert result.output == {"verdict": "malicious"}
        assert result.iterations == 1
        assert result.llm_calls == 1

    def test_intermediate_tool_feeds_the_next_call(self):
        calls = []

        def handler(arguments):
            calls.append(arguments)
            return {"result": "ok"}

        lookup = Tool("lookup", "query", {"type": "object", "properties": {}}, handler)
        client = FakeClient(
            [
                tool_use("lookup", {"ip": "1.2.3.4"}, block_id="tu_a"),
                tool_use("submit", {"verdict": "benign"}, block_id="tu_b"),
            ]
        )
        agent = Agent(client, "m", "prompt", [lookup, terminal_tool()], "submit")
        result = agent.run("analyze")

        assert calls == [{"ip": "1.2.3.4"}]
        assert result.tool_calls == ["lookup", "submit"]
        assert result.llm_calls == 2
        # The second call must carry the tool_result from the first.
        second = client.messages.calls[1]["messages"]
        assert second[-1]["content"][0]["type"] == "tool_result"
        assert second[-1]["content"][0]["tool_use_id"] == "tu_a"

    def test_unknown_tool_returns_as_an_error_without_crashing(self):
        client = FakeClient(
            [tool_use("nonexistent", {}), tool_use("submit", {"verdict": "benign"})]
        )
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        result = agent.run("analyze")
        tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
        assert tool_result["is_error"] is True
        assert "unknown" in tool_result["content"]
        assert result.output == {"verdict": "benign"}

    def test_exception_inside_a_tool_returns_as_an_error(self):
        def explode(arguments):
            raise RuntimeError("database is down")

        broken = Tool("breaks", "fails", {"type": "object", "properties": {}}, explode)
        client = FakeClient(
            [tool_use("breaks", {}), tool_use("submit", {"verdict": "benign"})]
        )
        agent = Agent(client, "m", "prompt", [broken, terminal_tool()], "submit")
        agent.run("analyze")
        tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
        assert tool_result["is_error"] is True
        assert "database is down" in tool_result["content"]

    def test_response_without_a_tool_call_is_an_error(self):
        client = FakeClient([text_response("I think it is malicious")])
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        with pytest.raises(AgentError, match="without calling"):
            agent.run("analyze")

    def test_model_refusal_is_an_error(self):
        client = FakeClient([FakeResponse(content=[], stop_reason="refusal")])
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        with pytest.raises(AgentError, match="refused"):
            agent.run("analyze")

    def test_max_tokens_truncation_is_an_error(self):
        client = FakeClient([text_response("...", stop_reason="max_tokens")])
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        with pytest.raises(AgentError, match="truncated"):
            agent.run("analyze")

    def test_iteration_limit_ends_the_loop(self):
        """A model that never concludes must not spin forever."""
        loop = Tool("loop", "spins", {"type": "object", "properties": {}}, lambda a: "ok")
        client = FakeClient([tool_use("loop", {}) for _ in range(5)])
        agent = Agent(client, "m", "prompt", [loop, terminal_tool()], "submit", max_iterations=3)
        with pytest.raises(AgentError, match="iterations"):
            agent.run("analyze")
        assert len(client.messages.calls) == 3

    def test_budget_stops_before_calling_the_model(self):
        client = FakeClient([tool_use("submit", {})])
        agent = Agent(client, "m", "prompt", [terminal_tool()], "submit")
        budget = LLMBudget(0)
        with pytest.raises(BudgetExceeded):
            agent.run("analyze", budget)
        assert client.messages.calls == []

    def test_terminal_tool_must_be_declared(self):
        with pytest.raises(ValueError):
            Agent(FakeClient([]), "m", "prompt", [terminal_tool()], "nonexistent")

    def test_request_uses_caching_and_opus5_parameters(self):
        client = FakeClient([tool_use("submit", {})])
        Agent(client, "claude-opus-5", "prompt", [terminal_tool()], "submit").run("x")
        call = client.messages.calls[0]
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["thinking"] == {"type": "adaptive"}
        assert "temperature" not in call  # rejected by Opus 5


class TestTriageTools:
    def test_search_events_finds_the_payload(self, attack_dossier, database):
        tools = {tool.name: tool for tool in build_tools(attack_dossier, database)}
        result = tools["search_events"].handler({"pattern": "SELECT"})
        assert result["match_count"] >= 1

    def test_search_events_with_invalid_regex_returns_an_error(self, attack_dossier, database):
        tools = {tool.name: tool for tool in build_tools(attack_dossier, database)}
        assert "error" in tools["search_events"].handler({"pattern": "["})

    def test_count_by_field_aggregates(self, attack_dossier, database):
        tools = {tool.name: tool for tool in build_tools(attack_dossier, database)}
        result = tools["count_by_field"].handler({"field": "status"})
        assert sum(result["counts"].values()) == attack_dossier.event_count

    def test_count_by_field_rejects_an_invalid_field(self, attack_dossier, database):
        tools = {tool.name: tool for tool in build_tools(attack_dossier, database)}
        assert "error" in tools["count_by_field"].handler({"field": "nonexistent"})

    def test_get_sample_events_honors_the_limit(self, attack_dossier, database):
        tools = {tool.name: tool for tool in build_tools(attack_dossier, database)}
        result = tools["get_sample_events"].handler({"limit": 2})
        assert result["count"] <= 2


class TestTriageAgent:
    def test_returns_a_structured_verdict(self, attack_dossier, database, settings):
        client = FakeClient([tool_use("submit_verdict", VERDICT_MALICIOUS)])
        agent = TriageAgent(client, database, settings)
        verdict = agent.analyze(attack_dossier, LLMBudget(5))
        assert verdict["verdict"] == "malicious"
        assert verdict["attack_types"] == ["sqli"]
        assert verdict["_meta"]["llm_calls"] == 1

    def test_dossier_reaches_the_user_prompt(self, attack_dossier, database, settings):
        client = FakeClient([tool_use("submit_verdict", VERDICT_MALICIOUS)])
        TriageAgent(client, database, settings).analyze(attack_dossier, LLMBudget(5))
        prompt = client.messages.calls[0]["messages"][0]["content"]
        assert attack_dossier.ip in prompt


class TestActionAgent:
    def _agent(self, client, database, settings):
        return ActionAgent(
            client, database, IPBlocker("dry_run"), Notifier(), settings
        )

    def test_block_records_in_the_blocklist(self, attack_dossier, database, settings):
        client = FakeClient(
            [
                tool_use("block_ip", {"reason": "confirmed sqli"}, block_id="tu_a"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "block", "rationale": "confirmed exploitation"},
                    block_id="tu_b",
                ),
            ]
        )
        plan, performed = self._agent(client, database, settings).decide(
            attack_dossier, VERDICT_MALICIOUS, LLMBudget(5)
        )
        assert plan["final_action"] == "block"
        assert database.is_blocked(attack_dossier.ip)
        assert performed[0]["type"] == "block"

    def test_request_level_dry_run_does_not_block(self, attack_dossier, database, settings):
        client = FakeClient(
            [
                tool_use("block_ip", {"reason": "sqli"}, block_id="tu_a"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "block", "rationale": "x"},
                    block_id="tu_b",
                ),
            ]
        )
        _, performed = self._agent(client, database, settings).decide(
            attack_dossier, VERDICT_MALICIOUS, LLMBudget(5), dry_run=True
        )
        assert database.is_blocked(attack_dossier.ip) is False
        assert performed[0]["executed"] is False

    def test_check_blocklist_reflects_the_database(self, attack_dossier, database, settings):
        database.add_to_blocklist(attack_dossier.ip, "previous", "dry_run")
        client = FakeClient(
            [
                tool_use("check_blocklist", {}, block_id="tu_a"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "monitor", "rationale": "already contained"},
                    block_id="tu_b",
                ),
            ]
        )
        self._agent(client, database, settings).decide(
            attack_dossier, VERDICT_MALICIOUS, LLMBudget(5)
        )
        tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
        assert '"blocked": true' in tool_result["content"]

    def test_alert_without_credentials_reports_not_sent(
        self, attack_dossier, database, settings
    ):
        client = FakeClient(
            [
                tool_use("send_alert", {"message": "SQLi detected"}, block_id="tu_a"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "alert", "rationale": "notify on-call"},
                    block_id="tu_b",
                ),
            ]
        )
        _, performed = self._agent(client, database, settings).decide(
            attack_dossier, VERDICT_MALICIOUS, LLMBudget(5)
        )
        assert performed[0]["type"] == "alert"
        assert performed[0]["executed"] is False

    def test_verdict_reaches_the_prompt(self, attack_dossier, database, settings):
        client = FakeClient(
            [tool_use("submit_action_plan", {"final_action": "allow", "rationale": "ok"})]
        )
        self._agent(client, database, settings).decide(
            attack_dossier, VERDICT_BENIGN, LLMBudget(5)
        )
        prompt = client.messages.calls[0]["messages"][0]["content"]
        assert "TRIAGE VERDICT" in prompt
        assert "crawler" in prompt
