from datetime import timezone

import pytest

from app.pipeline.parser import ParseError, parse_batch, parse_event, parse_log_line


def test_parse_event_normalizes_fields():
    event = parse_event(
        {
            "ip": "10.0.0.1",
            "time": "2019-01-22 00:26:16+00:00",
            "method": "get",
            "url": "/product/1?a=2",
            "status": "200",
            "size": "5667",
            "user_agent": "curl/8.0",
        }
    )
    assert event.ip == "10.0.0.1"
    assert event.method == "GET"
    assert event.status == 200
    assert event.size == 5667
    assert event.path == "/product/1"
    assert event.query == "a=2"
    assert event.timestamp.tzinfo is not None


def test_missing_ip_is_an_error():
    with pytest.raises(ParseError):
        parse_event({"url": "/x", "time": "2019-01-22 00:00:00+00:00"})


def test_missing_url_is_an_error():
    with pytest.raises(ParseError):
        parse_event({"ip": "1.2.3.4", "time": "2019-01-22 00:00:00+00:00"})


def test_invalid_timestamp_is_an_error():
    with pytest.raises(ParseError):
        parse_event({"ip": "1.2.3.4", "url": "/x", "time": "yesterday"})


def test_timestamp_without_timezone_assumes_utc():
    event = parse_event({"ip": "1.2.3.4", "url": "/x", "time": "2019-01-22 00:00:00"})
    assert event.timestamp.tzinfo == timezone.utc


def test_dash_size_becomes_zero():
    event = parse_event(
        {"ip": "1.2.3.4", "url": "/x", "time": "2019-01-22 00:00:00+00:00", "size": "-"}
    )
    assert event.size == 0


def test_decoded_url_resolves_percent_encoding():
    """Obfuscated payloads must reach the model decoded."""
    event = parse_event(
        {
            "ip": "1.2.3.4",
            "url": "/x?q=%27%20OR%201%3D1--",
            "time": "2019-01-22 00:00:00+00:00",
        }
    )
    assert "' OR 1=1--" in event.decoded_url


def test_parse_log_line_combined_format():
    line = (
        '203.0.113.5 - - [22/Jan/2019:00:26:16 +0000] "GET /wp-login.php HTTP/1.1" '
        '404 1234 "-" "python-requests/2.21.0"'
    )
    event = parse_log_line(line)
    assert event.ip == "203.0.113.5"
    assert event.url == "/wp-login.php"
    assert event.status == 404
    assert event.user_agent == "python-requests/2.21.0"


def test_invalid_log_line():
    with pytest.raises(ParseError):
        parse_log_line("this is not a log line")


def test_parse_batch_isolates_rejections():
    """One malformed line must not sink the whole batch."""
    events, rejected = parse_batch(
        [
            {"ip": "1.2.3.4", "url": "/a", "time": "2019-01-22 00:00:00+00:00"},
            {"url": "/b", "time": "2019-01-22 00:00:00+00:00"},
            "invalid line",
        ]
    )
    assert len(events) == 1
    assert len(rejected) == 2
    assert {item["index"] for item in rejected} == {1, 2}
