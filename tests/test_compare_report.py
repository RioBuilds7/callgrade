import json
from xml.etree import ElementTree as ET

from callgrade.compare import compare
from callgrade.models import CheckResult, JudgeResult, JudgeScore, JudgeSpec, RubricItem, RunResult, Scenario, ScenarioResult
from callgrade.report import load_run, render_html, write_report

from .conftest import make_transcript

SPEC = JudgeSpec(rubric=[RubricItem(id="task", question="q")])


def result(sid, passed, judge=None, latency=10.0, text="hello"):
    scenario = Scenario(id=sid, title=f"title {sid}", caller=["hi"], judge=SPEC)
    j = JudgeResult(status="scored", model="m", scores=[JudgeScore(id="task", score=judge)]) if judge else None
    return ScenarioResult(
        scenario=scenario,
        transcript=make_transcript(("caller", "hi"), ("agent", text, [], latency)),
        checks=[CheckResult(name="speakable", passed=passed, detail="" if passed else "markdown", turn=None if passed else 1)],
        judge=j, passed=passed, duration_ms=5,
    )


def run(*results):
    return RunResult(run_id="r", started_at="2026-01-01T00:00:00+00:00", agent="a", results=list(results))


def statuses(c):
    return {r.scenario_id: r.status for r in c.rows}


def test_status_for_each_kind_of_change():
    before = run(result("a", True), result("b", False), result("c", True), result("d", False), result("gone", True))
    after = run(result("a", False), result("b", True), result("c", True), result("d", False),
                result("new-ok", True), result("new-bad", False))
    c = compare(before, after)
    assert statuses(c) == {"a": "regressed", "b": "fixed", "c": "ok", "d": "still failing",
                           "new-ok": "new", "new-bad": "regressed", "gone": "removed"}
    assert c.failed()


def test_judge_drop_counts_as_regression_even_if_still_passing():
    c = compare(run(result("a", True, judge=5)), run(result("a", True, judge=4)), max_judge_drop=0.5)
    assert statuses(c)["a"] == "regressed" and "judge mean fell" in c.rows[0].reasons[0]
    assert not compare(run(result("a", True, judge=5)), run(result("a", True, judge=5))).failed()


def test_latency_rise_counts_as_regression():
    c = compare(run(result("a", True, latency=50)), run(result("a", True, latency=600)), max_latency_increase_ms=300)
    assert statuses(c)["a"] == "regressed"
    c = compare(run(result("a", True, latency=50)), run(result("a", True, latency=200)), max_latency_increase_ms=300)
    assert statuses(c)["a"] == "ok"


def test_min_pass_rate_gate():
    c = compare(run(result("a", False)), run(result("a", False)))
    assert not c.failed() and c.failed(min_pass_rate=0.5)


def test_markdown_summary_lists_regressions():
    md = compare(run(result("a", True)), run(result("a", False))).markdown()
    assert "**Regressions:** 1" in md and "| a | regressed |" in md


def test_report_files_and_escaping(tmp_path):
    evil = "<script>alert(1)</script>"
    current = run(result("a", False, text=evil), result("b", True, judge=5))
    comparison = compare(run(result("a", True), result("b", True)), current)
    html_path = write_report(current, tmp_path, comparison)

    html = html_path.read_text()
    assert evil not in html and "&lt;script&gt;" in html
    assert "1 regression against the baseline" in html
    assert 'class="turn agent blame"' in html  # failing turn is highlighted

    suite = ET.parse(tmp_path / "junit.xml").getroot()
    assert suite.get("tests") == "2" and suite.get("failures") == "1"
    assert load_run(tmp_path / "results.json").results[0].scenario.id == "a"
    assert "regressed" in (tmp_path / "comparison.md").read_text()


def test_html_without_comparison_has_no_banner():
    assert "banner" not in render_html(run(result("a", True))).split("<main>")[1]
