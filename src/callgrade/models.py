"""Data shapes shared by every part of the bench.

A *scenario* is what we want to test (caller lines plus expectations).
A *transcript* is what actually happened when we played it against an agent.
A *result* is the transcript plus every check and judge score.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """One utterance in a call."""

    speaker: Literal["caller", "agent"]
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Wall-clock time the agent took to answer this caller turn (agent turns only).
    latency_ms: float | None = None
    # Audio timestamps in seconds, only present for recorded calls.
    start: float | None = None
    end: float | None = None


class Transcript(BaseModel):
    scenario_id: str
    turns: list[Turn] = Field(default_factory=list)
    ended_by: Literal["agent", "caller", "max_turns", "error"] = "caller"
    error: str | None = None

    def agent_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == "agent"]

    def tool_calls(self) -> list[ToolCall]:
        return [c for t in self.agent_turns() for c in t.tool_calls]

    def as_text(self, numbered: bool = False) -> str:
        lines = []
        for i, t in enumerate(self.turns):
            line = f"{t.speaker.upper()}: {t.text}"
            if numbered:
                line = f"[{i}] {line}"
            for c in t.tool_calls:
                line += f"\n  [tool] {c.name}({c.args})"
            lines.append(line)
        return "\n".join(lines)


class RubricItem(BaseModel):
    id: str
    question: str


class JudgeSpec(BaseModel):
    rubric: list[RubricItem]
    # Each rubric item is scored 1-5; the scenario fails if any item is below this.
    min_score: int = 4


class Scenario(BaseModel):
    id: str
    title: str
    tags: list[str] = Field(default_factory=list)
    # What the caller says, in order. The bench sends one line per agent reply.
    caller: list[str]
    # Deterministic checks, each a one-key mapping like {"tool_called": {...}}.
    expect: list[dict[str, Any]] = Field(default_factory=list)
    judge: JudgeSpec | None = None
    # Where the scenario came from: "synthetic" or the recorded call file it replays.
    source: str = "synthetic"


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    turn: int | None = None  # index into transcript.turns, when one turn is to blame


class JudgeScore(BaseModel):
    id: str
    score: int
    reason: str = ""


class JudgeResult(BaseModel):
    status: Literal["scored", "skipped", "error"]
    model: str = ""
    scores: list[JudgeScore] = Field(default_factory=list)
    from_cache: bool = False
    detail: str = ""

    def mean(self) -> float | None:
        if not self.scores:
            return None
        return sum(s.score for s in self.scores) / len(self.scores)


class ScenarioResult(BaseModel):
    scenario: Scenario
    transcript: Transcript
    checks: list[CheckResult]
    judge: JudgeResult | None = None
    passed: bool
    duration_ms: float


class RunResult(BaseModel):
    run_id: str
    started_at: str
    agent: str
    judge_model: str = ""
    results: list[ScenarioResult]

    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    def by_id(self) -> dict[str, ScenarioResult]:
        return {r.scenario.id: r for r in self.results}
