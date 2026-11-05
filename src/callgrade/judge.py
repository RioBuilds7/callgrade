"""LLM-as-judge, with a cache so CI is free and repeatable.

Checks catch what can be written as a rule. The judge scores what can't
(did the agent actually resolve the caller's problem, was it polite under
pressure) against a short rubric, 1-5 per item.

Judges are slow, cost money and are not perfectly repeatable, so every verdict
is stored in `judge_cache/` keyed by a hash of (model, prompt). Same transcript,
same model: the stored verdict is reused. Modes:

    off    no judging
    cache  only stored verdicts; a miss is reported as "skipped", never as a pass
    auto   stored verdict if present, else ask the model (if one is configured)
    live   always ask the model and overwrite the stored verdict
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from .models import JudgeResult, JudgeScore, Scenario, Transcript

SYSTEM_PROMPT = """You are a strict quality reviewer for phone calls handled by an AI voice agent.
You read a call transcript and score the AGENT on each rubric item from 1 to 5:
5 = fully meets it, 4 = meets it with a minor flaw, 3 = partly, 2 = mostly fails, 1 = fails or does the opposite.
Judge only what is in the transcript. Turns are numbered; tool calls the agent made are shown as [tool] lines.
Read the whole call before scoring: a caller may change their mind, and what counts is how the call ends.
For each item, first write the reason (quote the turn number that decides it), then the score.
Reply with JSON only, no prose, in exactly this shape:
{"scores": [{"id": "<rubric id>", "reason": "<one or two short sentences>", "score": <1-5>}]}"""


def build_prompt(scenario: Scenario, transcript: Transcript) -> str:
    assert scenario.judge is not None
    rubric = "\n".join(f'- id "{r.id}": {r.question}' for r in scenario.judge.rubric)
    return (
        f"Call scenario: {scenario.title}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Transcript:\n{transcript.as_text(numbered=True)}\n\n"
        "Score every rubric id once."
    )


class JudgeParseError(ValueError):
    pass


def parse_scores(raw: str, rubric_ids: list[str]) -> list[JudgeScore]:
    """Pull the JSON object out of a model reply and validate it against the rubric."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise JudgeParseError("no JSON object in reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"bad JSON: {e}") from e
    items = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise JudgeParseError("missing 'scores' list")
    by_id: dict[str, JudgeScore] = {}
    for item in items:
        try:
            score = int(item["score"])
            sid = str(item["id"])
        except (KeyError, TypeError, ValueError) as e:
            raise JudgeParseError(f"bad score item {item!r}") from e
        if sid in rubric_ids and 1 <= score <= 5:
            by_id[sid] = JudgeScore(id=sid, score=score, reason=str(item.get("reason", ""))[:300])
    missing = [r for r in rubric_ids if r not in by_id]
    if missing:
        raise JudgeParseError(f"no valid score for {missing}")
    return [by_id[r] for r in rubric_ids]


class Judge:
    def __init__(
        self,
        mode: str = "auto",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path = "judge_cache",
        transport: Any = None,
    ):
        if mode not in ("off", "cache", "auto", "live"):
            raise ValueError(f"unknown judge mode {mode}")
        self.mode = mode
        self.model = model or os.environ.get("JUDGE_MODEL", "qwen2.5:3b")
        self.base_url = base_url or os.environ.get("JUDGE_BASE_URL")
        api_key = api_key or os.environ.get("JUDGE_API_KEY", "none")
        self.cache_dir = Path(cache_dir)
        self.client = httpx.AsyncClient(
            timeout=300.0, headers={"Authorization": f"Bearer {api_key}"}, transport=transport
        )

    def _key(self, prompt: str) -> str:
        blob = json.dumps({"model": self.model, "system": SYSTEM_PROMPT, "prompt": prompt}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _cache_path(self, scenario_id: str, key: str) -> Path:
        return self.cache_dir / f"{scenario_id}-{key}.json"

    async def score(self, scenario: Scenario, transcript: Transcript) -> JudgeResult | None:
        if self.mode == "off" or scenario.judge is None:
            return None
        if transcript.error:
            return JudgeResult(status="skipped", model=self.model, detail="call did not complete")
        prompt = build_prompt(scenario, transcript)
        ids = [r.id for r in scenario.judge.rubric]
        path = self._cache_path(scenario.id, self._key(prompt))

        if self.mode in ("cache", "auto") and path.exists():
            stored = json.loads(path.read_text())
            return JudgeResult(
                status="scored", model=self.model, from_cache=True,
                scores=[JudgeScore(**s) for s in stored["scores"]],
            )
        if self.mode == "cache" or not self.base_url:
            why = "no stored verdict for this transcript" + ("" if self.base_url else " and no JUDGE_BASE_URL set")
            return JudgeResult(status="skipped", model=self.model, detail=why)

        try:
            scores, raw = await self._ask(prompt, ids)
        except (JudgeParseError, httpx.HTTPError, KeyError) as e:
            return JudgeResult(status="error", model=self.model, detail=str(e)[:300])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"model": self.model, "scenario": scenario.id, "prompt": prompt, "raw": raw,
                 "scores": [s.model_dump() for s in scores]},
                indent=2,
            )
            + "\n"
        )
        return JudgeResult(status="scored", model=self.model, scores=scores)

    async def _ask(self, prompt: str, ids: list[str]) -> tuple[list[JudgeScore], str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        last_error: JudgeParseError | None = None
        for _ in range(2):  # one retry with the parse error fed back
            r = await self.client.post(
                self.base_url.rstrip("/") + "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"] or ""
            try:
                return parse_scores(raw, ids), raw
            except JudgeParseError as e:
                last_error = e
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That was not valid ({e}). Reply with the JSON object only."},
                ]
        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        await self.client.aclose()
