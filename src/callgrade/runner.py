"""Play scenarios against an agent and grade them."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .adapters import Agent, AgentError
from .checks import run_checks
from .judge import Judge
from .models import RunResult, Scenario, ScenarioResult, Transcript, Turn

MAX_AGENT_TURNS = 20


def load_scenarios(path: str | Path, tags: list[str] | None = None) -> list[Scenario]:
    """Load one YAML file, or every *.yaml under a folder (subfolders included), sorted by path."""
    p = Path(path)
    files = sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    scenarios = []
    for f in files:
        data = yaml.safe_load(f.read_text())
        for item in data if isinstance(data, list) else [data]:
            scenarios.append(Scenario(**item))
    ids = [s.id for s in scenarios]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate scenario ids: {sorted(dupes)}")
    if tags:
        scenarios = [s for s in scenarios if set(tags) & set(s.tags)]
    return scenarios


async def play(agent: Agent, scenario: Scenario, session_id: str) -> Transcript:
    """Send the caller lines one at a time and record what comes back."""
    t = Transcript(scenario_id=scenario.id)
    for line in scenario.caller:
        if len(t.agent_turns()) >= MAX_AGENT_TURNS:
            t.ended_by = "max_turns"
            return t
        t.turns.append(Turn(speaker="caller", text=line))
        start = time.perf_counter()
        try:
            reply = await agent.respond(session_id, line)
        except (AgentError, KeyError, ValueError) as e:
            t.ended_by, t.error = "error", f"agent failed: {e}"
            return t
        latency = (time.perf_counter() - start) * 1000
        t.turns.append(Turn(speaker="agent", text=reply.text, tool_calls=reply.tool_calls, latency_ms=round(latency, 1)))
        if reply.end_call:
            t.ended_by = "agent"
            return t
    t.ended_by = "caller"
    return t


async def grade(scenario: Scenario, transcript: Transcript, judge: Judge | None, duration_ms: float) -> ScenarioResult:
    checks = run_checks(transcript, scenario.expect)
    verdict = await judge.score(scenario, transcript) if judge else None
    passed = all(c.passed for c in checks)
    if verdict and verdict.status == "scored" and scenario.judge:
        passed &= all(s.score >= scenario.judge.min_score for s in verdict.scores)
    return ScenarioResult(
        scenario=scenario, transcript=transcript, checks=checks, judge=verdict,
        passed=passed, duration_ms=round(duration_ms, 1),
    )


async def run_suite(
    agent: Agent, scenarios: list[Scenario], judge: Judge | None = None, concurrency: int = 4
) -> RunResult:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    gate = asyncio.Semaphore(concurrency)

    async def one(s: Scenario) -> ScenarioResult:
        async with gate:
            t0 = time.perf_counter()
            transcript = await play(agent, s, session_id=f"{run_id}-{s.id}")
            return await grade(s, transcript, judge, (time.perf_counter() - t0) * 1000)

    results = await asyncio.gather(*(one(s) for s in scenarios))
    return RunResult(
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        agent=agent.name,
        judge_model=judge.model if judge and judge.mode != "off" else "",
        results=list(results),
    )
