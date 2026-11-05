import json

import pytest

from callgrade.recorded import call_metrics, caller_lines, load_call, to_scenario

CALL = {
    "id": "c1",
    "segments": [
        {"speaker": "caller", "start": 0.0, "end": 2.0, "text": "Hi."},
        {"speaker": "agent", "start": 3.0, "end": 5.0, "text": "Hello, how can I help?"},  # gap 1.0
        {"speaker": "caller", "start": 4.5, "end": 6.0, "text": "I need"},  # caller cuts in
        {"speaker": "caller", "start": 6.2, "end": 7.0, "text": "a cleaning."},
        {"speaker": "agent", "start": 6.8, "end": 8.0, "text": "Sure."},  # agent cuts in
        {"speaker": "caller", "start": 12.0, "end": 13.0, "text": "Tuesday."},  # 4 s silence
        {"speaker": "agent", "start": 16.0, "end": 17.0, "text": "Booked."},  # gap 3.0, silence
    ],
}


def test_call_metrics():
    m = call_metrics(CALL)
    assert m.duration_s == 17.0
    assert m.caller_interruptions == 1 and m.agent_interruptions == 1
    assert m.long_silences == 2
    assert m.response_gap_p50_s == 1.0 and m.response_gap_p95_s == 3.0
    assert m.agent_talk_share == round(4.2 / 17, 3)


def test_caller_lines_merge_consecutive_caller_segments():
    assert caller_lines(CALL) == ["Hi.", "I need a cleaning.", "Tuesday."]


def test_to_scenario_adds_voice_checks():
    s = to_scenario(dict(CALL, title="T"), source="c1.json")
    assert s.id == "replay-c1" and s.caller[1] == "I need a cleaning."
    assert {next(iter(e)) for e in s.expect} == {"speakable", "max_words_per_turn", "no_pii", "no_repeated_turn"}
    assert s.judge and s.source == "c1.json"


def test_load_call_sorts_and_validates(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"id": "x", "segments": list(reversed(CALL["segments"]))}))
    assert load_call(p)["segments"][0]["text"] == "Hi."
    p.write_text(json.dumps({"id": "x", "segments": [{"speaker": "bot", "start": 0, "end": 1, "text": "x"}]}))
    with pytest.raises(ValueError, match="speaker"):
        load_call(p)
    p.write_text(json.dumps({"id": "x", "segments": [{"speaker": "agent", "start": 2, "end": 1, "text": "x"}]}))
    with pytest.raises(ValueError, match="ends before"):
        load_call(p)


def test_shipped_synthetic_calls_load():
    m = call_metrics(load_call("calls/synthetic-reschedule-slow.json"))
    assert m.agent_interruptions == 1 and m.long_silences == 1
