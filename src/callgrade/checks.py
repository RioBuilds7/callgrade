"""Deterministic checks.

Each check is a plain function that takes the transcript and the check's
parameters from the scenario YAML and returns one CheckResult. They are cheap,
never flaky, and run before the LLM judge, so most regressions are caught here.

Voice agents have failure modes chat agents don't: anything the TTS engine
reads aloud literally (markdown, URLs, emoji), answers too long to listen to,
and slow replies that leave dead air. The `speakable`, `max_words_per_turn`
and `latency` checks target those.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from .models import CheckResult, Transcript

CheckFn = Callable[[Transcript, Any], CheckResult]
REGISTRY: dict[str, CheckFn] = {}


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    def register(fn: CheckFn) -> CheckFn:
        REGISTRY[name] = fn
        return fn

    return register


def _agent_turns(t: Transcript) -> list[tuple[int, str]]:
    return [(i, turn.text) for i, turn in enumerate(t.turns) if turn.speaker == "agent"]


def _arg_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


@check("tool_called")
def tool_called(t: Transcript, spec: dict) -> CheckResult:
    """The agent called tool `name`, with at least the given args (case-insensitive).

    `within_turns: N` also requires the call in one of the agent's first N replies
    (an emergency transfer that comes late is still wrong). On failure it blames
    the Nth agent reply, the last point where the call should have happened.
    """
    name, want, within = spec["name"], spec.get("args", {}), spec.get("within_turns")
    label = f"tool_called:{name}"
    agent_idx = [i for i, turn in enumerate(t.turns) if turn.speaker == "agent"]
    allowed = agent_idx[:within] if within else agent_idx
    calls = [(i, c) for i in allowed for c in t.turns[i].tool_calls if c.name == name]
    if not calls:
        where = f" in the first {within} agent turn(s)" if within else ""
        blame = allowed[-1] if within and allowed else None
        return CheckResult(name=label, passed=False, detail=f"{name} was never called{where}", turn=blame)
    for _, c in calls:
        if all(k in c.args and _arg_matches(v, c.args[k]) for k, v in want.items()):
            return CheckResult(name=label, passed=True, detail=f"{name}({c.args})")
    seen = "; ".join(str(c.args) for _, c in calls)
    return CheckResult(name=label, passed=False, detail=f"called with {seen}, wanted {want}", turn=calls[-1][0])


@check("tool_not_called")
def tool_not_called(t: Transcript, spec: dict) -> CheckResult:
    name = spec["name"]
    for i, turn in enumerate(t.turns):
        for c in turn.tool_calls:
            if c.name == name:
                return CheckResult(
                    name=f"tool_not_called:{name}", passed=False, detail=f"called {name}({c.args})", turn=i
                )
    return CheckResult(name=f"tool_not_called:{name}", passed=True)


@check("said_any")
def said_any(t: Transcript, patterns: list[str]) -> CheckResult:
    """At least one agent turn matches one of the regexes (case-insensitive)."""
    for i, text in _agent_turns(t):
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return CheckResult(name="said_any", passed=True, detail=f"'{p}' in turn {i}", turn=i)
    return CheckResult(name="said_any", passed=False, detail=f"none of {patterns} said")


@check("said_before_tool")
def said_before_tool(t: Transcript, spec: dict) -> CheckResult:
    """An agent turn matching `pattern` comes before the first call to `tool` (e.g. read-back before booking)."""
    tool, rx = spec["tool"], re.compile(spec["pattern"], re.IGNORECASE)
    label = f"said_before_tool:{tool}"
    for i, turn in enumerate(t.turns):
        if turn.speaker != "agent":
            continue
        if any(c.name == tool for c in turn.tool_calls):
            return CheckResult(name=label, passed=False, detail=f"{tool} called in turn {i} before any '{spec['pattern']}'", turn=i)
        if rx.search(turn.text):
            return CheckResult(name=label, passed=True, detail=f"said in turn {i}")
    return CheckResult(name=label, passed=True, detail=f"{tool} never called")


@check("never_said")
def never_said(t: Transcript, patterns: list[str]) -> CheckResult:
    for i, text in _agent_turns(t):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return CheckResult(name="never_said", passed=False, detail=f"said '{m.group(0)}'", turn=i)
    return CheckResult(name="never_said", passed=True)


_UNSPEAKABLE = [
    (re.compile(r"\*\*|__|`|^#{1,6}\s", re.M), "markdown formatting"),
    (re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M), "a bulleted or numbered list"),
    (re.compile(r"https?://|www\.", re.I), "a URL"),
    (re.compile(r"[\U0001F300-\U0001FAFF☀-➿]"), "an emoji"),
    (re.compile(r"\{\{?\s*\w+\s*\}?\}|\[[A-Z_]{3,}\]"), "a template placeholder"),
]


@check("speakable")
def speakable(t: Transcript, _spec: Any = None) -> CheckResult:
    """Nothing a text-to-speech engine would read out literally or garble."""
    for i, text in _agent_turns(t):
        for rx, what in _UNSPEAKABLE:
            if rx.search(text):
                return CheckResult(name="speakable", passed=False, detail=f"turn {i} contains {what}", turn=i)
    return CheckResult(name="speakable", passed=True)


@check("max_words_per_turn")
def max_words_per_turn(t: Transcript, limit: int) -> CheckResult:
    worst_i, worst_n = None, 0
    for i, text in _agent_turns(t):
        n = len(text.split())
        if n > worst_n:
            worst_i, worst_n = i, n
    passed = worst_n <= limit
    return CheckResult(
        name="max_words_per_turn",
        passed=passed,
        detail=f"longest turn {worst_n} words (limit {limit})",
        turn=None if passed else worst_i,
    )


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, so p95 of a short list is a value that occurred."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


@check("latency")
def latency(t: Transcript, spec: dict) -> CheckResult:
    """Agent response time budget, e.g. {p95_ms: 1200} or {max_ms: 2000}."""
    values = [turn.latency_ms for turn in t.agent_turns() if turn.latency_ms is not None]
    if not values:
        return CheckResult(name="latency", passed=False, detail="no latency measured")
    parts, passed = [], True
    if "p95_ms" in spec:
        p95 = percentile(values, 95)
        passed &= p95 <= spec["p95_ms"]
        parts.append(f"p95 {p95:.0f} ms (budget {spec['p95_ms']})")
    if "max_ms" in spec:
        mx = max(values)
        passed &= mx <= spec["max_ms"]
        parts.append(f"max {mx:.0f} ms (budget {spec['max_ms']})")
    return CheckResult(name="latency", passed=passed, detail=", ".join(parts))


_PII = [
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "a card-like number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "an SSN-like number"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "an email address"),
]


@check("no_pii")
def no_pii(t: Transcript, spec: dict | None = None) -> CheckResult:
    """The agent never reads back sensitive data, plus any extra names listed in `names`."""
    extra = [(re.compile(re.escape(n), re.I), f"'{n}'") for n in (spec or {}).get("names", [])]
    for i, text in _agent_turns(t):
        for rx, what in _PII + extra:
            if rx.search(text):
                return CheckResult(name="no_pii", passed=False, detail=f"turn {i} reveals {what}", turn=i)
    return CheckResult(name="no_pii", passed=True)


@check("no_repeated_turn")
def no_repeated_turn(t: Transcript, _spec: Any = None) -> CheckResult:
    """The agent never says the same thing twice in a row (the usual sign of a stuck dialog loop)."""
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    turns = _agent_turns(t)
    for (_, prev), (i, cur) in zip(turns, turns[1:]):
        if norm(prev) and norm(prev) == norm(cur):
            return CheckResult(name="no_repeated_turn", passed=False, detail=f"turn {i} repeats the previous agent turn", turn=i)
    return CheckResult(name="no_repeated_turn", passed=True)


@check("ended_by")
def ended_by(t: Transcript, who: str) -> CheckResult:
    return CheckResult(name="ended_by", passed=t.ended_by == who, detail=f"ended by {t.ended_by}, wanted {who}")


@check("max_agent_turns")
def max_agent_turns(t: Transcript, limit: int) -> CheckResult:
    n = len(t.agent_turns())
    return CheckResult(name="max_agent_turns", passed=n <= limit, detail=f"{n} agent turns (limit {limit})")


def run_checks(t: Transcript, expect: list[dict[str, Any]]) -> list[CheckResult]:
    results = []
    if t.error:
        results.append(CheckResult(name="call_completed", passed=False, detail=t.error))
    for item in expect:
        if len(item) != 1:
            raise ValueError(f"each expectation needs exactly one key, got {list(item)}")
        (name, spec), = item.items()
        if name not in REGISTRY:
            raise ValueError(f"unknown check '{name}'. Known: {sorted(REGISTRY)}")
        results.append(REGISTRY[name](t, spec))
    return results
