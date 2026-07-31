from datetime import timedelta

import pytest

from app.pipeline.correlator import correlate, group_by_ip_window
from tests.conftest import SQLI_URL, StubDetector, make_event


def test_batch_without_suspects_opens_no_incident():
    """The benign case must cost zero: no incident, no LLM call."""
    events = [make_event(url=f"/product/{i}", offset=i) for i in range(20)]
    scores = [0.01] * len(events)
    assert correlate(events, scores, threshold=0.5) == []


def test_incident_groups_by_ip():
    events = [
        make_event(url=SQLI_URL, ip="1.1.1.1"),
        make_event(url=SQLI_URL, ip="2.2.2.2"),
    ]
    dossiers = correlate(events, [0.9, 0.9], threshold=0.5)
    assert {dossier.ip for dossier in dossiers} == {"1.1.1.1", "2.2.2.2"}


def test_dossier_includes_the_clean_traffic_of_the_window():
    """Knowing 2 of 6 requests were malicious is different from 2 of 2."""
    events = [make_event(url=f"/product/{i}", offset=i) for i in range(4)]
    events.append(make_event(url=SQLI_URL, offset=5))
    scores = [0.01] * 4 + [0.95]
    dossier = correlate(events, scores, threshold=0.5)[0]
    assert dossier.event_count == 5
    assert dossier.suspicious_count == 1


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError):
        correlate([make_event()], [0.1, 0.2], threshold=0.5)


def test_dossiers_arrive_sorted_by_score():
    events = [
        make_event(url=SQLI_URL, ip="1.1.1.1"),
        make_event(url=SQLI_URL, ip="2.2.2.2"),
    ]
    dossiers = correlate(events, [0.6, 0.99], threshold=0.5)
    assert dossiers[0].ip == "2.2.2.2"
    assert dossiers[0].max_score >= dossiers[1].max_score


def test_top_samples_honors_limit_and_order():
    events = [make_event(url=f"{SQLI_URL}{i}", offset=i) for i in range(10)]
    scores = [i / 10 for i in range(10)]
    dossier = correlate(events, scores, threshold=0.0)[0]
    samples = dossier.top_samples(3)
    assert len(samples) == 3
    assert samples[0].score > samples[-1].score


def test_dossier_prompt_carries_the_essentials(attack_dossier):
    """The prompt is the only context the agent receives; it has to be complete."""
    prompt = attack_dossier.to_prompt()
    assert attack_dossier.incident_id in prompt
    assert attack_dossier.ip in prompt
    assert "WINDOW STATISTICS" in prompt
    assert "LEAST LIKELY EVENTS" in prompt
    assert "SELECT" in prompt.upper()


def test_prompt_explains_what_the_score_means(attack_dossier):
    """The agent needs to know that improbable is not a synonym for malicious."""
    prompt = attack_dossier.to_prompt()
    assert "unusual legitimate" in prompt
    assert "0.50 is the threshold" in prompt


def test_evidence_is_attached_to_the_top_events(attack_dossier):
    """The improbable span is what makes the score quotable by the agent."""
    samples = attack_dossier.top_samples()
    assert any(item.evidence for item in samples)
    assert any(item.evidence in attack_dossier.to_prompt() for item in samples)


def test_explain_is_called_only_for_the_dossier_samples():
    """Locating the substring costs more than scoring; only worth it for what is read."""
    calls = []

    def explain(event):
        calls.append(event)
        return "span"

    events = [make_event(url=f"{SQLI_URL}{i}", offset=i) for i in range(30)]
    correlate(events, [0.9] * 30, threshold=0.5, explain=explain)
    assert len(calls) == 5  # MAX_SAMPLE_EVENTS


def test_correlate_without_explain_does_not_break():
    events = [make_event(url=SQLI_URL)]
    dossier = correlate(events, [0.9], threshold=0.5)[0]
    assert dossier.top_samples()[0].evidence == ""


def test_prompt_stays_compact_next_to_the_raw_batch():
    """The correlator's reason to exist: the dossier must not grow with the batch."""
    # Every offset stays below 300s so they land in the same window.
    events = [make_event(url=f"/product/{i}", offset=i % 290) for i in range(300)]
    events.append(make_event(url=SQLI_URL, offset=295))
    detector = StubDetector()
    scores = detector.score_events(events)
    dossier = correlate(events, scores, detector.event_threshold)[0]
    assert dossier.event_count == 301
    assert len(dossier.to_prompt()) < 4000


def test_grouping_is_stable_when_the_batch_is_sliced():
    """The bucket comes from the absolute timestamp, not from the position in the batch.

    Without that, the same event would land in different groups depending on how the
    client sliced the request, and the analysis outcome would depend on batch size.
    """
    events = [make_event(offset=i * 60) for i in range(10)]
    everything = group_by_ip_window(events, timedelta(minutes=5))
    first = group_by_ip_window(events[:5], timedelta(minutes=5))
    second = group_by_ip_window(events[5:], timedelta(minutes=5))
    assert set(first) | set(second) == set(everything)


def test_grouping_separates_ips():
    events = [make_event(ip="1.1.1.1"), make_event(ip="2.2.2.2")]
    assert len(group_by_ip_window(events)) == 2
