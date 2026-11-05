"""Command line entry point.

Exit codes: 0 = quality gate passed, 1 = quality gate failed,
2 = could not test at all (bad input, agent unreachable on every call).
A broken agent endpoint is not reported as a quality drop, and vice versa.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

from . import __version__
from .adapters import make_agent
from .calibrate import run_calibration
from .checks import REGISTRY
from .compare import compare
from .judge import Judge
from .recorded import call_metrics, load_call, to_scenario
from .report import load_run, write_report
from .runner import load_scenarios, run_suite


def _run(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios, args.tag)
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 2
    agent = make_agent(args.agent)
    judge = Judge(mode=args.judge, cache_dir=args.judge_cache)

    async def go():
        try:
            return await run_suite(agent, scenarios, judge, concurrency=args.concurrency)
        finally:
            await agent.aclose()
            await judge.aclose()

    run = asyncio.run(go())
    comparison = (
        compare(load_run(args.baseline), run, args.max_judge_drop, args.max_latency_increase_ms)
        if args.baseline else None
    )
    out = write_report(run, Path(args.out) / run.run_id, comparison)

    for r in run.results:
        mark = "PASS" if r.passed else "FAIL"
        judge_note = ""
        if r.judge:
            judge_note = f"  judge {r.judge.mean():.1f}" if r.judge.status == "scored" else f"  judge {r.judge.status}"
        print(f"{mark}  {r.scenario.id}{judge_note}")
        for c in r.checks:
            if not c.passed:
                print(f"      {c.name}: {c.detail}")
    print(f"\n{sum(r.passed for r in run.results)}/{len(run.results)} passed. Report: {out}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and comparison:
        with open(summary, "a") as f:
            f.write("## callgrade\n\n" + comparison.markdown() + "\n")

    if all(r.transcript.error for r in run.results):
        print("every call failed to complete: check the agent is running", file=sys.stderr)
        return 2
    if comparison:
        print(f"\n{comparison.markdown()}")
        return 1 if comparison.failed(args.min_pass_rate) else 0
    return 0 if all(r.passed for r in run.results) else 1


def _compare(args: argparse.Namespace) -> int:
    c = compare(load_run(args.baseline), load_run(args.current), args.max_judge_drop, args.max_latency_increase_ms)
    print(c.markdown())
    return 1 if c.failed(args.min_pass_rate) else 0


def _import_call(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.calls:
        s = to_scenario(load_call(path), source=Path(path).name)
        dest = out / f"{s.id}.yaml"
        dest.write_text(yaml.safe_dump(s.model_dump(exclude_defaults=False), sort_keys=False, allow_unicode=True))
        print(f"wrote {dest} ({len(s.caller)} caller turns)")
    return 0


def _call_metrics(args: argparse.Namespace) -> int:
    rows = [call_metrics(load_call(p)).as_dict() for p in args.calls]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    cols = list(rows[0]) if rows else []
    print("  ".join(cols))
    for r in rows:
        print("  ".join(str(r[c]) for c in cols))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    judge = Judge(mode=args.judge, cache_dir=args.judge_cache)
    result = run_calibration(args.labels, load_scenarios(args.scenarios), judge)
    print(result.report())
    if result.total == 0:
        return 2
    return 1 if result.false_pass > args.max_false_pass else 0


def _checks(_args: argparse.Namespace) -> int:
    for name, fn in sorted(REGISTRY.items()):
        print(f"{name:20} {(fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="callgrade", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="play scenarios against an agent and grade them")
    r.add_argument("--agent", required=True, help="agent URL (HTTP protocol) or agent .yaml (OpenAI-compatible)")
    r.add_argument("--scenarios", default="scenarios", help="YAML file or folder")
    r.add_argument("--tag", action="append", help="only scenarios with this tag (repeatable)")
    r.add_argument("--judge", default="auto", choices=["off", "cache", "auto", "live"])
    r.add_argument("--judge-cache", default="judge_cache")
    r.add_argument("--out", default="runs")
    r.add_argument("--baseline", help="results.json of a known-good run to compare against")
    r.add_argument("--max-judge-drop", type=float, default=0.5)
    r.add_argument("--max-latency-increase-ms", type=float, default=300)
    r.add_argument("--min-pass-rate", type=float, default=0.0)
    r.add_argument("--concurrency", type=int, default=4)
    r.set_defaults(fn=_run)

    c = sub.add_parser("compare", help="compare two results.json files")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--max-judge-drop", type=float, default=0.5)
    c.add_argument("--max-latency-increase-ms", type=float, default=300)
    c.add_argument("--min-pass-rate", type=float, default=0.0)
    c.set_defaults(fn=_compare)

    i = sub.add_parser("import-call", help="turn recorded call JSON into replay scenarios")
    i.add_argument("calls", nargs="+")
    i.add_argument("--out", default="scenarios")
    i.set_defaults(fn=_import_call)

    m = sub.add_parser("call-metrics", help="timing metrics for recorded calls")
    m.add_argument("calls", nargs="+")
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=_call_metrics)

    cal = sub.add_parser("calibrate", help="measure judge agreement with hand labels")
    cal.add_argument("labels")
    cal.add_argument("--scenarios", default="scenarios")
    cal.add_argument("--judge", default="auto", choices=["cache", "auto", "live"])
    cal.add_argument("--judge-cache", default="judge_cache")
    cal.add_argument("--max-false-pass", type=int, default=0)
    cal.set_defaults(fn=_calibrate)

    k = sub.add_parser("checks", help="list the available checks")
    k.set_defaults(fn=_checks)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
