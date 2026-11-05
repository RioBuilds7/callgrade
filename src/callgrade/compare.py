"""Compare a run to a baseline and decide whether CI should fail."""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import percentile
from .models import RunResult, ScenarioResult


def p95_latency(r: ScenarioResult) -> float | None:
    values = [t.latency_ms for t in r.transcript.agent_turns() if t.latency_ms is not None]
    return percentile(values, 95) if values else None


def judge_mean(r: ScenarioResult) -> float | None:
    return r.judge.mean() if r.judge and r.judge.status == "scored" else None


@dataclass
class Row:
    scenario_id: str
    status: str  # regressed, fixed, still failing, ok, new, removed
    before_passed: bool | None
    after_passed: bool | None
    judge_before: float | None = None
    judge_after: float | None = None
    p95_before: float | None = None
    p95_after: float | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class Comparison:
    rows: list[Row]
    pass_rate_before: float
    pass_rate_after: float

    @property
    def regressions(self) -> list[Row]:
        return [r for r in self.rows if r.status == "regressed"]

    def failed(self, min_pass_rate: float = 0.0) -> bool:
        return bool(self.regressions) or self.pass_rate_after < min_pass_rate

    def markdown(self) -> str:
        fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
        lines = [
            f"**Pass rate:** {self.pass_rate_before:.0%} -> {self.pass_rate_after:.0%}  ",
            f"**Regressions:** {len(self.regressions)}",
            "",
            "| Scenario | Status | Judge | p95 latency (ms) | Why |",
            "|---|---|---|---|---|",
        ]
        for r in self.rows:
            judge = f"{fmt(r.judge_before, '.1f')} -> {fmt(r.judge_after, '.1f')}"
            lat = f"{fmt(r.p95_before, '.0f')} -> {fmt(r.p95_after, '.0f')}"
            lines.append(f"| {r.scenario_id} | {r.status} | {judge} | {lat} | {'; '.join(r.reasons)} |")
        return "\n".join(lines)


def compare(
    baseline: RunResult, current: RunResult, max_judge_drop: float = 0.5, max_latency_increase_ms: float = 300
) -> Comparison:
    """Regression = passed before and fails now, or the judge mean fell by more than
    `max_judge_drop`, or p95 latency rose by more than `max_latency_increase_ms`.
    The latency rule catches a slower agent even in scenarios with no latency budget."""
    before, after = baseline.by_id(), current.by_id()
    rows = []
    for sid in list(after) + [s for s in before if s not in after]:
        b, a = before.get(sid), after.get(sid)
        if a is None:
            rows.append(Row(sid, "removed", b.passed if b else None, None))
            continue
        row = Row(
            sid, "ok", b.passed if b else None, a.passed,
            judge_mean(b) if b else None, judge_mean(a),
            p95_latency(b) if b else None, p95_latency(a),
        )
        row.reasons = [f"{c.name}: {c.detail}" for c in a.checks if not c.passed]
        if a.judge and a.judge.status == "scored" and a.scenario.judge:
            row.reasons += [
                f"judge {s.id}={s.score}" for s in a.judge.scores if s.score < a.scenario.judge.min_score
            ]
        dropped = (
            row.judge_before is not None
            and row.judge_after is not None
            and row.judge_before - row.judge_after > max_judge_drop
        )
        if dropped:
            row.reasons.append(f"judge mean fell {row.judge_before:.1f} -> {row.judge_after:.1f}")
        slower = (
            row.p95_before is not None
            and row.p95_after is not None
            and row.p95_after - row.p95_before > max_latency_increase_ms
        )
        if slower:
            row.reasons.append(f"p95 latency rose {row.p95_before:.0f} -> {row.p95_after:.0f} ms")
        if b is None:
            row.status = "new" if a.passed else "regressed"
        elif b.passed and (not a.passed or dropped or slower):
            row.status = "regressed"
        elif not b.passed and a.passed:
            row.status = "fixed"
        elif not a.passed:
            row.status = "still failing"
        rows.append(row)
    return Comparison(rows, baseline.pass_rate(), current.pass_rate())
