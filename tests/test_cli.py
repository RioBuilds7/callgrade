"""End to end through the CLI, with the sample agent on a real local port."""

import json

from callgrade.cli import main


def latest(out):
    return next(out.glob("*/results.json"))


def test_v1_passes_then_v2_fails_against_baseline(live_agent, tmp_path, capsys):
    v1 = live_agent("v1")
    assert main(["run", "--agent", v1, "--judge", "off", "--out", str(tmp_path / "v1")]) == 0
    baseline = latest(tmp_path / "v1")

    v2 = live_agent("v2", delay=0.0)
    code = main(["run", "--agent", v2, "--judge", "off", "--out", str(tmp_path / "v2"), "--baseline", str(baseline)])
    out = capsys.readouterr().out
    assert code == 1
    assert "| emergency-transfer | regressed |" in out
    report = next((tmp_path / "v2").glob("*/report.html")).read_text()
    assert "regressions against the baseline" in report

    assert main(["compare", str(baseline), str(latest(tmp_path / "v2"))]) == 1
    assert main(["compare", str(baseline), str(baseline)]) == 0


def test_unreachable_agent_is_exit_2_not_a_quality_failure(tmp_path, capsys):
    code = main(["run", "--agent", "http://127.0.0.1:9", "--judge", "off", "--tag", "smoke", "--out", str(tmp_path)])
    assert code == 2
    assert "check the agent is running" in capsys.readouterr().err


def test_bad_input_is_exit_2(tmp_path, capsys):
    assert main(["run", "--agent", "ftp://x", "--out", str(tmp_path)]) == 2
    assert main(["run", "--agent", "http://x", "--scenarios", str(tmp_path / "missing.yaml")]) == 2


def test_import_call_and_metrics(tmp_path, capsys):
    assert main(["import-call", "calls/synthetic-hours-then-book.json", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "replay-synthetic-hours-then-book.yaml").exists()
    assert main(["call-metrics", "--json", "calls/synthetic-hours-then-book.json"]) == 0
    rows = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert rows[0]["caller_interruptions"] == 1


def test_calibrate_from_cache_without_a_model(tmp_path, capsys):
    # with no stored verdicts and no model, nothing can be scored: exit 2, not a fake 100%
    code = main(["calibrate", "calibration/labels.yaml", "--judge", "cache", "--judge-cache", str(tmp_path)])
    assert code == 2 and "not scored: 12" in capsys.readouterr().out


def test_checks_lists_registry(capsys):
    assert main(["checks"]) == 0
    assert "said_before_tool" in capsys.readouterr().out


def test_require_judge_fails_on_cache_miss(live_agent, tmp_path, capsys):
    v1 = live_agent("v1")
    args = ["run", "--agent", v1, "--tag", "smoke", "--judge", "cache", "--judge-cache", str(tmp_path / "empty"),
            "--out", str(tmp_path / "runs")]
    assert main(args) == 0  # skipped verdicts are reported, not counted as failures...
    assert main(args + ["--require-judge"]) == 2  # ...unless CI asks for them
    assert "re-record" in capsys.readouterr().err


def test_shipped_baseline_matches_v1_with_stored_verdicts(live_agent, tmp_path):
    """What CI does: v1 against the committed baseline, judge from the committed cache only."""
    v1 = live_agent("v1")
    code = main(["run", "--agent", v1, "--judge", "cache", "--require-judge", "--baseline", "baseline/results.json",
                 "--out", str(tmp_path)])
    assert code == 0
