"""Measure how far to trust the judge before letting it gate CI.

A labels file lists stored transcripts and the verdict a person gave each
rubric item (pass or fail). We run the judge on the same transcripts and count
agreement. The number that matters most is false passes: the judge says fine,
the person says broken. A judge with false passes cannot be the only gate.

    items:
      - transcript: transcripts/v2-office-hours.json
        scenario: office-hours
        labels: {answer: fail}
        note: reads a markdown list and a URL aloud
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .judge import Judge
from .models import Scenario, Transcript


@dataclass
class Disagreement:
    transcript: str
    item: str
    human: str
    judge: str
    reason: str


@dataclass
class Calibration:
    model: str
    agree: int = 0
    false_pass: int = 0  # judge pass, human fail: the dangerous kind
    false_fail: int = 0  # judge fail, human pass: noisy, costs reruns
    unscored: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.agree + self.false_pass + self.false_fail

    def report(self) -> str:
        rate = self.agree / self.total if self.total else 0.0
        lines = [
            f"judge {self.model}: agrees with labels on {self.agree}/{self.total} items ({rate:.0%})",
            f"  false passes (judge ok, label fail): {self.false_pass}",
            f"  false fails  (judge fail, label ok): {self.false_fail}",
        ]
        if self.unscored:
            lines.append(f"  not scored: {self.unscored}")
        for d in self.disagreements:
            lines.append(f"  - {d.transcript} [{d.item}] label {d.human}, judge {d.judge}: {d.reason}")
        return "\n".join(lines)


async def calibrate(labels_path: str | Path, scenarios: dict[str, Scenario], judge: Judge) -> Calibration:
    labels_path = Path(labels_path)
    spec = yaml.safe_load(labels_path.read_text())
    result = Calibration(model=judge.model)
    for item in spec["items"]:
        scenario = scenarios[item["scenario"]]
        if scenario.judge is None:
            raise ValueError(f"scenario {scenario.id} has no judge rubric to calibrate")
        transcript = Transcript.model_validate_json((labels_path.parent / item["transcript"]).read_text())
        verdict = await judge.score(scenario, transcript)
        if verdict is None or verdict.status != "scored":
            result.unscored += len(item["labels"])
            continue
        scores = {s.id: s for s in verdict.scores}
        for rubric_id, human in item["labels"].items():
            s = scores[rubric_id]
            said = "pass" if s.score >= scenario.judge.min_score else "fail"
            if said == human:
                result.agree += 1
                continue
            if said == "pass":
                result.false_pass += 1
            else:
                result.false_fail += 1
            result.disagreements.append(
                Disagreement(item["transcript"], rubric_id, human, f"{said} ({s.score})", s.reason)
            )
    return result


def run_calibration(labels_path: str, scenarios: list[Scenario], judge: Judge) -> Calibration:
    async def go() -> Calibration:
        try:
            return await calibrate(labels_path, {s.id: s for s in scenarios}, judge)
        finally:
            await judge.aclose()

    return asyncio.run(go())
