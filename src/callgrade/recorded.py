"""Recorded calls: timing metrics, and turning a call into a replayable scenario.

A recorded call is the diarized, timestamped transcript most speech-to-text
services return (one segment per stretch of speech):

    {"id": "...", "title": "...", "segments": [
        {"speaker": "caller", "start": 0.0, "end": 2.4, "text": "Hi, I need to move my appointment"},
        {"speaker": "agent",  "start": 3.1, "end": 6.0, "text": "Sure, can I get your name?"}]}

Timing is where voice agents feel broken even when the words are right:
slow first responses, talking over the caller, long dead air. Those come from
the timestamps, not the text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .checks import percentile
from .models import JudgeSpec, RubricItem, Scenario, Turn

SILENCE_S = 2.5


@dataclass
class CallMetrics:
    call_id: str
    duration_s: float
    agent_talk_share: float
    response_gap_p50_s: float
    response_gap_p95_s: float
    agent_interruptions: int  # agent starts talking while the caller is still talking
    caller_interruptions: int  # caller cuts in on the agent (barge-in)
    long_silences: int  # gaps longer than SILENCE_S between any two segments

    def as_dict(self) -> dict:
        return asdict(self)


def load_call(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    segs = data.get("segments", [])
    for s in segs:
        if s["speaker"] not in ("caller", "agent"):
            raise ValueError(f"{path}: speaker must be caller or agent, got {s['speaker']!r}")
        if s["end"] < s["start"]:
            raise ValueError(f"{path}: segment ends before it starts: {s}")
    data["segments"] = sorted(segs, key=lambda s: s["start"])
    return data


def call_metrics(call: dict) -> CallMetrics:
    segs = [Turn(**s) for s in call["segments"]]
    if not segs:
        return CallMetrics(call["id"], 0, 0, 0, 0, 0, 0, 0)
    duration = max(s.end for s in segs) - min(s.start for s in segs)
    agent_time = sum(s.end - s.start for s in segs if s.speaker == "agent")
    gaps, agent_cut, caller_cut, silences = [], 0, 0, 0
    for prev, cur in zip(segs, segs[1:]):
        gap = cur.start - prev.end
        if gap > SILENCE_S:
            silences += 1
        if prev.speaker == cur.speaker:
            continue
        if gap < 0:
            if cur.speaker == "agent":
                agent_cut += 1
            else:
                caller_cut += 1
        elif cur.speaker == "agent":
            gaps.append(gap)
    return CallMetrics(
        call_id=call["id"],
        duration_s=round(duration, 2),
        agent_talk_share=round(agent_time / duration, 3) if duration else 0.0,
        response_gap_p50_s=round(percentile(gaps, 50), 2),
        response_gap_p95_s=round(percentile(gaps, 95), 2),
        agent_interruptions=agent_cut,
        caller_interruptions=caller_cut,
        long_silences=silences,
    )


def caller_lines(call: dict) -> list[str]:
    """Caller speech between agent turns, merged into one line per caller turn."""
    lines: list[str] = []
    last = None
    for s in call["segments"]:
        if s["speaker"] == "caller":
            if last == "caller" and lines:
                lines[-1] += " " + s["text"].strip()
            else:
                lines.append(s["text"].strip())
        last = s["speaker"]
    return lines


def to_scenario(call: dict, source: str) -> Scenario:
    """Make a replay scenario: same caller lines, standard voice checks, a generic rubric.

    The recorded agent's own lines are not reused; the point is to see how the
    *current* agent handles the same caller. Edit the generated YAML to add
    task-specific checks (which tool should be called, with what).
    """
    return Scenario(
        id=f"replay-{call['id']}",
        # The recording's own title often describes how the *old* agent behaved ("agent talks over
        # the caller"). It is kept out of the scenario title because the judge reads that title.
        title=f"Replay of recorded call {call['id']}",
        tags=["replay"],
        caller=caller_lines(call),
        expect=[{"speakable": {}}, {"max_words_per_turn": 45}, {"no_pii": {}}, {"no_repeated_turn": {}}],
        judge=JudgeSpec(
            rubric=[
                RubricItem(id="resolved", question=call.get("goal", "Did the agent resolve what the caller called about, or hand off correctly?")),
                RubricItem(id="tone", question="Was the agent brief, polite and clear enough to follow by ear?"),
            ]
        ),
        source=source,
    )
