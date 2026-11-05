"""The bench against the real sample agent (in-process, no network), v1 and v2."""

import asyncio

import httpx
import pytest

from callgrade.adapters import AgentError, AgentReply, HttpAgent, OpenAIChatAgent
from callgrade.models import Scenario
from callgrade.runner import load_scenarios, play, run_suite
from sample_agent.app import Receptionist, parse_time

from .conftest import ScriptedAgent, asgi_transport

SCENARIOS = load_scenarios("scenarios")


def run(variant, delay=0.0, scenarios=SCENARIOS):
    agent = HttpAgent("http://agent", transport=asgi_transport(variant, delay))

    async def go():
        try:
            return await run_suite(agent, scenarios)
        finally:
            await agent.aclose()

    return asyncio.run(go())


def test_v1_passes_every_scenario():
    result = run("v1")
    failures = {r.scenario.id: [c.detail for c in r.checks if not c.passed] for r in result.results if not r.passed}
    assert failures == {}
    assert len(result.results) == len(SCENARIOS) >= 12


def test_v2_regressions_are_each_caught_by_a_check():
    by_id = run("v2").by_id()
    failed = lambda sid: {c.name for c in by_id[sid].checks if not c.passed}  # noqa: E731
    assert "said_before_tool:book_appointment" in failed("book-cleaning")  # 1. no read-back
    assert "speakable" in failed("office-hours")  # 2. markdown list
    assert "tool_called:transfer_call" in failed("emergency-transfer")  # 3. no transfer
    assert "never_said" in failed("other-patient-privacy")  # 4. privacy leak
    assert by_id["insurance"].passed  # untouched behaviour still passes


def test_v2_latency_regression_is_caught():
    # 5. the slowdown: a smaller delay than the real 1.1 s keeps the test quick but still over budget
    by_id = run("v2", delay=0.85, scenarios=[s for s in SCENARIOS if s.id == "book-cleaning"]).by_id()
    assert any(c.name == "latency" and not c.passed for c in by_id["book-cleaning"].checks)


def test_load_scenarios_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a.yaml").write_text("- {id: x, title: a, caller: [hi]}\n")
    (tmp_path / "b.yaml").write_text("- {id: x, title: b, caller: [hi]}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_scenarios(tmp_path)


def test_load_scenarios_filters_by_tag():
    assert {s.id for s in load_scenarios("scenarios", ["safety"])} == {
        "emergency-transfer", "other-patient-privacy", "asks-for-human"
    }


def test_play_stops_when_agent_ends_call():
    s = Scenario(id="s", title="s", caller=["one", "two", "three"])
    t = asyncio.run(play(ScriptedAgent([AgentReply("ok"), AgentReply("bye", end_call=True)]), s, "x"))
    assert t.ended_by == "agent" and len(t.agent_turns()) == 2


def test_play_records_agent_failure():
    s = Scenario(id="s", title="s", caller=["one", "two"])
    t = asyncio.run(play(ScriptedAgent([AgentReply("ok"), AgentError("HTTP 500")]), s, "x"))
    assert t.ended_by == "error" and "HTTP 500" in t.error


def test_http_agent_retries_5xx_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"text": "hi", "tool_calls": [{"name": "t", "args": {}}]})

    async def go():
        a = HttpAgent("http://a", transport=httpx.MockTransport(handler))
        try:
            return await a.respond("s", "hello")
        finally:
            await a.aclose()

    reply = asyncio.run(go())
    assert reply.text == "hi" and reply.tool_calls[0].name == "t" and len(calls) == 3


def test_http_agent_does_not_retry_4xx():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(422, text="bad body")

    async def go():
        a = HttpAgent("http://a", transport=httpx.MockTransport(handler))
        try:
            await a.respond("s", "hello")
        finally:
            await a.aclose()

    with pytest.raises(AgentError, match="422"):
        asyncio.run(go())
    assert len(calls) == 1


def test_openai_agent_runs_a_tool_round(tmp_path):
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        "model: m\nbase_url: http://llm/v1\nsystem: be brief\n"
        "tools:\n  - name: transfer_call\n    mock_result: {status: transferred}\n    ends_call: true\n"
    )
    replies = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "transfer_call", "arguments": '{"reason": "urgent"}'}}]},
        {"role": "assistant", "content": "Connecting you now."},
    ]
    seen = []

    def handler(req):
        import json
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": replies.pop(0)}]})

    async def go():
        a = OpenAIChatAgent(str(cfg), transport=httpx.MockTransport(handler))
        try:
            return await a.respond("s", "my tooth is bleeding")
        finally:
            await a.aclose()

    reply = asyncio.run(go())
    assert reply.text == "Connecting you now." and reply.end_call
    assert reply.tool_calls[0].args == {"reason": "urgent"}
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": '{"status": "transferred"}'}


@pytest.mark.parametrize(
    "text,want",
    [("one PM", "13:00"), ("1 pm", "13:00"), ("at 3", "15:00"), ("ten thirty", "10:30"), ("morning", "10:30"),
     ("noonish", None)],
)
def test_sample_agent_parse_time(text, want):
    assert parse_time(text, ["10:30", "13:00", "15:00"]) == want


def test_sample_agent_rejects_unknown_variant():
    with pytest.raises(ValueError):
        Receptionist("v3")
