import asyncio
import json

import httpx
import pytest

from callgrade.judge import Judge, JudgeParseError, parse_scores
from callgrade.models import JudgeSpec, RubricItem, Scenario

from .conftest import make_transcript

SCENARIO = Scenario(
    id="s1", title="t", caller=["hi"],
    judge=JudgeSpec(rubric=[RubricItem(id="task", question="q1"), RubricItem(id="tone", question="q2")]),
)
TRANSCRIPT = make_transcript(("caller", "hi"), ("agent", "hello"))
GOOD = '{"scores": [{"id": "task", "reason": "fine", "score": 5}, {"id": "tone", "reason": "ok", "score": 4}]}'


def fake_llm(*contents):
    """Transport that answers each chat request with the next content string."""
    replies = list(contents)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    return httpx.MockTransport(handler), seen


def score(judge):
    async def go():
        try:
            return await judge.score(SCENARIO, TRANSCRIPT)
        finally:
            await judge.aclose()

    return asyncio.run(go())


def test_parse_scores_accepts_prose_around_json():
    scores = parse_scores("Sure! " + GOOD + " Hope that helps", ["task", "tone"])
    assert [(s.id, s.score) for s in scores] == [("task", 5), ("tone", 4)]


@pytest.mark.parametrize(
    "raw",
    [
        "no json here",
        "{not json}",
        '{"scores": "nope"}',
        '{"scores": [{"id": "task", "score": 5}]}',  # tone missing
        '{"scores": [{"id": "task", "score": 9}, {"id": "tone", "score": 4}]}',  # out of range
    ],
)
def test_parse_scores_rejects_bad_replies(raw):
    with pytest.raises(JudgeParseError):
        parse_scores(raw, ["task", "tone"])


def test_live_judge_writes_cache_then_cache_mode_reads_it(tmp_path):
    transport, seen = fake_llm(GOOD)
    r = score(Judge("auto", model="m", base_url="http://llm", cache_dir=tmp_path, transport=transport))
    assert r.status == "scored" and not r.from_cache and r.mean() == 4.5
    assert len(seen) == 1 and seen[0]["temperature"] == 0
    stored = list(tmp_path.glob("s1-*.json"))
    assert len(stored) == 1 and json.loads(stored[0].read_text())["raw"] == GOOD

    transport, seen = fake_llm()
    r = score(Judge("cache", model="m", cache_dir=tmp_path, transport=transport))
    assert r.status == "scored" and r.from_cache and seen == []


def test_cache_miss_is_skipped_not_passed(tmp_path):
    r = score(Judge("cache", model="m", cache_dir=tmp_path))
    assert r.status == "skipped" and r.scores == []


def test_auto_without_endpoint_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    r = score(Judge("auto", model="m", cache_dir=tmp_path))
    assert r.status == "skipped" and "JUDGE_BASE_URL" in r.detail


def test_cache_key_depends_on_model(tmp_path):
    transport, _ = fake_llm(GOOD)
    score(Judge("auto", model="model-a", base_url="http://llm", cache_dir=tmp_path, transport=transport))
    r = score(Judge("cache", model="model-b", cache_dir=tmp_path))
    assert r.status == "skipped"


def test_retries_once_after_unparseable_reply(tmp_path):
    transport, seen = fake_llm("I think it went well.", GOOD)
    r = score(Judge("live", model="m", base_url="http://llm", cache_dir=tmp_path, transport=transport))
    assert r.status == "scored" and len(seen) == 2
    assert "not valid" in seen[1]["messages"][-1]["content"]


def test_two_bad_replies_is_an_error_and_nothing_cached(tmp_path):
    transport, _ = fake_llm("nope", "still nope")
    r = score(Judge("live", model="m", base_url="http://llm", cache_dir=tmp_path, transport=transport))
    assert r.status == "error" and not list(tmp_path.iterdir())


def test_http_error_is_reported(tmp_path):
    transport = httpx.MockTransport(lambda req: httpx.Response(429, json={"error": "rate limited"}))
    r = score(Judge("live", model="m", base_url="http://llm", cache_dir=tmp_path, transport=transport))
    assert r.status == "error" and "429" in r.detail


def test_off_mode_and_no_rubric_return_none(tmp_path):
    assert score(Judge("off", cache_dir=tmp_path)) is None

    async def go():
        j = Judge("auto", base_url="http://llm", cache_dir=tmp_path)
        try:
            return await j.score(Scenario(id="x", title="x", caller=["hi"]), TRANSCRIPT)
        finally:
            await j.aclose()

    assert asyncio.run(go()) is None


def test_failed_call_is_not_judged(tmp_path):
    t = make_transcript(("caller", "hi"))
    t.error = "agent failed"

    async def go():
        j = Judge("auto", base_url="http://llm", cache_dir=tmp_path)
        try:
            return await j.score(SCENARIO, t)
        finally:
            await j.aclose()

    assert asyncio.run(go()).status == "skipped"
