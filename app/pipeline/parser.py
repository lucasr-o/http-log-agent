"""Log event normalization.

Two input shapes are accepted: an already structured dictionary (what the API
receives) or a raw line in Apache/Nginx Combined Log Format. Both produce the
same `LogEvent`, which is the only representation used from here on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus, urlsplit

COMBINED_LOG_RE = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<url>\S+) (?P<protocol>[^"]*)" '
    r'(?P<status>\d{3}) (?P<size>\S+)'
    r'(?: "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)")?'
)

APACHE_TIME_FMT = "%d/%b/%Y:%H:%M:%S %z"


class ParseError(ValueError):
    """An event that could not be normalized."""


@dataclass(slots=True)
class LogEvent:
    ip: str
    timestamp: datetime
    method: str
    url: str
    protocol: str
    status: int
    size: int
    referrer: str
    user_agent: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def path(self) -> str:
        return urlsplit(self.url).path

    @property
    def query(self) -> str:
        return urlsplit(self.url).query

    @property
    def decoded_url(self) -> str:
        """URL with percent-encoding resolved.

        Attacks routinely obfuscate payloads with URL encoding (`%27` for a
        single quote, for instance). Decoding before scoring keeps the same
        injection from reaching the model as two different token sequences.
        """
        try:
            return unquote_plus(self.url)
        except (UnicodeDecodeError, ValueError):
            return self.url


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", "-"}:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        raise ParseError("missing timestamp")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, APACHE_TIME_FMT)
        except ValueError as exc:
            raise ParseError(f"unrecognizable timestamp: {text!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_event(payload: dict[str, Any]) -> LogEvent:
    """Normalize a structured event. Raises ParseError when unusable."""
    ip = str(payload.get("ip") or "").strip()
    url = str(payload.get("url") or "").strip()
    if not ip:
        raise ParseError("field 'ip' is required")
    if not url:
        raise ParseError("field 'url' is required")

    return LogEvent(
        ip=ip,
        timestamp=_coerce_timestamp(payload.get("time") or payload.get("timestamp")),
        method=str(payload.get("method") or "GET").strip().upper()[:16],
        url=url,
        protocol=str(payload.get("protocol") or "HTTP/1.1").strip(),
        status=_coerce_int(payload.get("status"), 0),
        size=_coerce_int(payload.get("size"), 0),
        referrer=str(payload.get("referrer") or "-").strip(),
        user_agent=str(payload.get("user_agent") or "-").strip(),
        raw=payload,
    )


def parse_log_line(line: str) -> LogEvent:
    """Normalize a raw Combined Log Format line."""
    match = COMBINED_LOG_RE.match(line.strip())
    if not match:
        raise ParseError("line does not match Combined Log Format")
    groups = match.groupdict()
    return parse_event(
        {
            "ip": groups["ip"],
            "time": groups["time"],
            "method": groups["method"],
            "url": groups["url"],
            "protocol": groups["protocol"],
            "status": groups["status"],
            "size": groups["size"],
            "referrer": groups.get("referrer") or "-",
            "user_agent": groups.get("user_agent") or "-",
        }
    )


def _as_payload(item: Any) -> dict[str, Any]:
    """Accept either a dict or a Pydantic model — the API hands over the latter."""
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump()
    raise ParseError(f"unsupported event type: {type(item).__name__}")


def parse_batch(items: list[Any]) -> tuple[list[LogEvent], list[dict[str, Any]]]:
    """Normalize a heterogeneous batch.

    Returns the valid events plus the list of rejections. One malformed line does
    not sink the whole batch: recording the rejection and moving on is what a log
    collector is expected to do.
    """
    events: list[LogEvent] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        try:
            if isinstance(item, str):
                events.append(parse_log_line(item))
            else:
                events.append(parse_event(_as_payload(item)))
        except ParseError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return events, rejected
