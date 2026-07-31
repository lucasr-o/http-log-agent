from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier


class TestIPBlocker:
    def test_dry_run_executes_nothing(self):
        result = IPBlocker(mode="dry_run").block("203.0.113.9", "confirmed sqli")
        assert result.executed is False
        assert result.mode == "dry_run"

    def test_invalid_ip_is_rejected(self):
        result = IPBlocker(mode="dry_run").block("not-an-ip", "test")
        assert result.executed is False
        assert result.invalid_target is True
        assert "invalid" in result.detail

    def test_command_injection_is_rejected(self):
        """The target comes from an LLM decision over third-party data; validation is mandatory."""
        blocker = IPBlocker(mode="enforce", command="/bin/echo {ip}")
        result = blocker.block("1.2.3.4; rm -rf /", "injection attempt")
        assert result.executed is False
        assert result.invalid_target is True

    def test_enforce_without_configured_command_does_nothing(self):
        result = IPBlocker(mode="enforce", command="").block("203.0.113.9", "x")
        assert result.executed is False
        assert "BLOCK_COMMAND" in result.detail

    def test_enforce_runs_the_configured_command(self):
        blocker = IPBlocker(mode="enforce", command="/bin/echo blocking {ip}")
        result = blocker.block("203.0.113.9", "sqli")
        assert result.executed is True
        assert "203.0.113.9" in result.detail

    def test_failing_command_reports_the_error(self):
        blocker = IPBlocker(mode="enforce", command="/bin/false {ip}")
        result = blocker.block("203.0.113.9", "sqli")
        assert result.executed is False

    def test_unknown_mode_falls_back_to_dry_run(self):
        assert IPBlocker(mode="anything-at-all").mode == "dry_run"


class TestNotifier:
    def test_without_credentials_it_only_logs(self):
        result = Notifier().send("test alert")
        assert result.sent is False
        assert result.channel == "log"

    def test_configured_requires_both_token_and_chat(self):
        assert Notifier(bot_token="t").configured is False
        assert Notifier(chat_id="c").configured is False
        assert Notifier(bot_token="t", chat_id="c").configured is True

    def test_incident_formatting(self):
        text = Notifier.format_incident(
            incident_id="inc_1", ip="203.0.113.9", severity="critical",
            verdict="malicious", summary="SQL injection.", action="block",
            attack_types=["sqli"],
        )
        assert "inc_1" in text
        assert "203.0.113.9" in text
        assert "CRITICAL" in text
        assert "sqli" in text

    def test_formatting_without_declared_types(self):
        text = Notifier.format_incident(
            "inc_2", "1.2.3.4", "low", "suspicious", "summary", "monitor", []
        )
        assert "unclassified" in text
