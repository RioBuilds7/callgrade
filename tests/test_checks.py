import pytest

from callgrade.checks import percentile, run_checks

from .conftest import make_transcript

BOOKED = make_transcript(
    ("caller", "Book me Tuesday."),
    ("agent", "To confirm: Tuesday at 10:30 AM for Dana. Is that right?"),
    ("caller", "Yes."),
    ("agent", "You're booked.", [("book_appointment", {"day": "Tuesday", "time": "10:30"})]),
    ended_by="agent",
)


def one(t, name, spec):
    return run_checks(t, [{name: spec}])[0]


def test_tool_called_matches_args_case_insensitively():
    assert one(BOOKED, "tool_called", {"name": "book_appointment", "args": {"day": "tuesday"}}).passed


def test_tool_called_reports_wrong_args():
    r = one(BOOKED, "tool_called", {"name": "book_appointment", "args": {"day": "friday"}})
    assert not r.passed and "wanted" in r.detail


def test_tool_called_missing_tool():
    r = one(BOOKED, "tool_called", {"name": "transfer_call"})
    assert not r.passed and "never called" in r.detail


def test_tool_called_wrong_args_blames_that_call():
    r = one(BOOKED, "tool_called", {"name": "book_appointment", "args": {"day": "friday"}})
    assert r.turn == 3


def test_tool_called_within_turns():
    late = make_transcript(
        ("caller", "It's bleeding."),
        ("agent", "Our next opening is Thursday."),
        ("caller", "Please hurry."),
        ("agent", "Transferring you.", [("transfer_call", {})]),
    )
    assert one(late, "tool_called", {"name": "transfer_call"}).passed
    r = one(late, "tool_called", {"name": "transfer_call", "within_turns": 1})
    assert not r.passed and r.turn == 1 and "first 1 agent turn" in r.detail
    assert one(late, "tool_called", {"name": "transfer_call", "within_turns": 2}).passed


def test_tool_not_called_points_at_turn():
    r = one(BOOKED, "tool_not_called", {"name": "book_appointment"})
    assert not r.passed and r.turn == 3


def test_said_any_and_never_said():
    assert one(BOOKED, "said_any", ["to confirm"]).passed
    assert not one(BOOKED, "said_any", ["goodbye"]).passed
    r = one(BOOKED, "never_said", [r"\btuesday\b"])
    assert not r.passed and r.turn == 1


def test_said_before_tool_passes_when_readback_first():
    assert one(BOOKED, "said_before_tool", {"tool": "book_appointment", "pattern": "to confirm"}).passed


def test_said_before_tool_fails_when_tool_first():
    t = make_transcript(
        ("caller", "Book me."),
        ("agent", "Booked! To confirm, Tuesday.", [("book_appointment", {})]),
    )
    r = one(t, "said_before_tool", {"tool": "book_appointment", "pattern": "to confirm"})
    assert not r.passed and r.turn == 1


@pytest.mark.parametrize(
    "text",
    [
        "Our hours:\n- Monday 8 to 5",
        "We are **open** today",
        "1. Monday\n2. Tuesday",
        "See https://example.com/hours",
        "Great 😀",
        "Hi {{name}}, you're booked",
        "Hi [CUSTOMER_NAME]",
    ],
)
def test_speakable_flags_what_tts_reads_literally(text):
    t = make_transcript(("caller", "hi"), ("agent", text))
    assert not one(t, "speakable", {}).passed


def test_speakable_allows_normal_speech():
    t = make_transcript(("caller", "hi"), ("agent", "We're open 8 AM to 5 PM, Monday to Friday - see you soon."))
    assert one(t, "speakable", {}).passed


def test_max_words_per_turn_blames_longest_turn():
    t = make_transcript(("caller", "hi"), ("agent", "short"), ("caller", "and"), ("agent", "one two three four"))
    r = one(t, "max_words_per_turn", 3)
    assert not r.passed and r.turn == 3


def test_latency_budget():
    t = make_transcript(
        ("caller", "a"), ("agent", "b", [], 100.0), ("caller", "c"), ("agent", "d", [], 900.0)
    )
    assert not one(t, "latency", {"p95_ms": 800}).passed
    assert one(t, "latency", {"max_ms": 1000}).passed


def test_latency_without_measurements_fails():
    t = make_transcript(("caller", "a"), ("agent", "b", [], None))
    assert not one(t, "latency", {"p95_ms": 800}).passed


@pytest.mark.parametrize("text", ["Card 4111 1111 1111 1111", "SSN 123-45-6789", "mail jo@example.com"])
def test_no_pii(text):
    t = make_transcript(("caller", "hi"), ("agent", text))
    assert not one(t, "no_pii", {}).passed


def test_no_pii_extra_names():
    t = make_transcript(("caller", "hi"), ("agent", "John Parker is in on Thursday"))
    assert not one(t, "no_pii", {"names": ["john parker"]}).passed
    assert one(t, "no_pii", {}).passed


def test_no_repeated_turn():
    t = make_transcript(("caller", "a"), ("agent", "Which time?"), ("caller", "one"), ("agent", "Which time?"))
    r = one(t, "no_repeated_turn", {})
    assert not r.passed and r.turn == 3


def test_ended_by_and_max_agent_turns():
    assert one(BOOKED, "ended_by", "agent").passed
    assert not one(BOOKED, "max_agent_turns", 1).passed


def test_error_transcript_adds_failed_completion_check():
    t = make_transcript(("caller", "hi"))
    t.error = "agent failed: HTTP 500"
    assert run_checks(t, [])[0].name == "call_completed"


def test_unknown_check_is_an_error():
    with pytest.raises(ValueError, match="unknown check"):
        run_checks(BOOKED, [{"made_up": {}}])


def test_two_keys_in_one_expectation_is_an_error():
    with pytest.raises(ValueError, match="exactly one key"):
        run_checks(BOOKED, [{"speakable": {}, "said_any": ["x"]}])


def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([float(i) for i in range(1, 21)], 95) == 19.0
