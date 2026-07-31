"""Alert notification.

Sends to Telegram when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured;
otherwise it logs the alert and reports that nothing was sent. A missing
credential is not an error — it is the default mode, so the project runs with no
configuration at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
REQUEST_TIMEOUT = 10.0
MAX_MESSAGE_CHARS = 3500


@dataclass(slots=True)
class NotifyResult:
    sent: bool
    channel: str
    detail: str


class Notifier:
    def __init__(self, bot_token: str = "", chat_id: str = "") -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, message: str) -> NotifyResult:
        text = message[:MAX_MESSAGE_CHARS]
        if not self.configured:
            logger.info("[notification not configured] %s", text.replace("\n", " | "))
            return NotifyResult(
                sent=False,
                channel="log",
                detail="TELEGRAM_BOT_TOKEN/CHAT_ID missing; alert written to the log",
            )
        try:
            response = httpx.post(
                TELEGRAM_API.format(token=self.bot_token),
                json={"chat_id": self.chat_id, "text": text},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("failed to send alert to Telegram: %s", exc)
            return NotifyResult(False, "telegram", f"delivery failed: {exc}")
        return NotifyResult(True, "telegram", "alert delivered")

    @staticmethod
    def format_incident(
        incident_id: str, ip: str, severity: str, verdict: str,
        summary: str, action: str, attack_types: list[str],
    ) -> str:
        types = ", ".join(attack_types) if attack_types else "unclassified"
        return (
            f"[BOT] {severity.upper()} incident\n"
            f"ID: {incident_id}\n"
            f"IP: {ip}\n"
            f"Verdict: {verdict}\n"
            f"Types: {types}\n"
            f"Action: {action}\n\n"
            f"{summary}"
        )
