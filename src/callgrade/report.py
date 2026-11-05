"""Outputs: results.json, a JUnit file for CI, and a static HTML dashboard.

The dashboard is plain HTML with a few lines of CSS and no JavaScript build,
so it opens from disk, from a CI artifact, or from GitHub Pages.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

from .compare import Comparison, judge_mean, p95_latency
from .models import RunResult, ScenarioResult

CSS = """
:root{--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700;--mute:#59636e;--line:#d1d9e0;--bg:#f6f8fa}
*{box-sizing:border-box}body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;color:#1f2328;background:#fff}
header{padding:24px 32px;border-bottom:1px solid var(--line);background:var(--bg)}
h1{font-size:20px;margin:0 0 4px}main{padding:24px 32px;max-width:1200px}
.meta{color:var(--mute);font-size:13px}.tiles{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
.tile{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:130px}
.tile b{display:block;font-size:22px}.tile span{color:var(--mute);font-size:12px}
.banner{padding:10px 14px;border-radius:8px;margin-bottom:16px;font-weight:600}
.banner.bad{background:#ffebe9;color:var(--bad)}.banner.ok{background:#dafbe1;color:var(--ok)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--mute);font-weight:600;text-transform:uppercase}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.pass{background:var(--ok)}.fail{background:var(--bad)}.skip{background:var(--mute)}
.regressed{background:var(--bad)}.fixed{background:var(--ok)}.new{background:var(--warn)}
details{margin:0 0 4px}summary{cursor:pointer;color:var(--mute);font-size:13px}
tr.main td{border-bottom:none;padding-bottom:2px}td.num{white-space:nowrap}.why{font-size:13px;color:var(--mute)}
.detail-body{display:grid;grid-template-columns:minmax(260px,1fr) 2fr;gap:24px;padding:8px 0 12px}
@media (max-width:800px){.detail-body{grid-template-columns:1fr}}
.turn{margin:6px 0;padding:8px 12px;border-radius:8px;max-width:760px}
.caller{background:var(--bg)}.agent{background:#ddf4ff;margin-left:40px}
.turn.blame{outline:2px solid var(--bad)}.who{font-size:11px;font-weight:700;color:var(--mute);text-transform:uppercase}
.tool{font:12px ui-monospace,monospace;color:#6639ba;margin-top:4px}.lat{float:right;font-size:11px;color:var(--mute)}
.bar{height:6px;background:var(--line);border-radius:3px;width:120px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:6px;border-radius:3px;background:#0969da}
ul.checks{margin:6px 0;padding-left:18px}.checks .no{color:var(--bad)}.checks .yes{color:var(--ok)}
"""


def write_json(run: RunResult, path: Path) -> None:
    path.write_text(run.model_dump_json(indent=2) + "\n")


def load_run(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text())


def write_junit(run: RunResult, path: Path) -> None:
    suite = ET.Element(
        "testsuite", name="callgrade", tests=str(len(run.results)),
        failures=str(sum(not r.passed for r in run.results)),
    )
    for r in run.results:
        case = ET.SubElement(suite, "testcase", classname="callgrade", name=r.scenario.id,
                             time=f"{r.duration_ms / 1000:.3f}")
        if not r.passed:
            why = [f"{c.name}: {c.detail}" for c in r.checks if not c.passed]
            if r.judge and r.judge.status == "scored":
                why += [f"judge {s.id}={s.score}: {s.reason}" for s in r.judge.scores]
            fail = ET.SubElement(case, "failure", message="; ".join(why)[:500])
            fail.text = r.transcript.as_text()
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _judge_cell(r: ScenarioResult) -> str:
    if r.judge is None:
        return "-"
    if r.judge.status != "scored":
        return f'<span class="pill skip" title="{escape(r.judge.detail)}">{r.judge.status}</span>'
    return f"{r.judge.mean():.1f} / 5"


def _transcript(r: ScenarioResult) -> str:
    blamed = {c.turn for c in r.checks if not c.passed and c.turn is not None}
    out = []
    for i, t in enumerate(r.transcript.turns):
        lat = f'<span class="lat">{t.latency_ms:.0f} ms</span>' if t.latency_ms is not None else ""
        tools = "".join(f'<div class="tool">&rarr; {escape(c.name)}({escape(json.dumps(c.args))})</div>'
                        for c in t.tool_calls)
        cls = f"turn {t.speaker}" + (" blame" if i in blamed else "")
        out.append(f'<div class="{cls}">{lat}<div class="who">{t.speaker}</div>{escape(t.text)}{tools}</div>')
    if r.transcript.error:
        out.append(f'<div class="turn agent blame"><div class="who">error</div>{escape(r.transcript.error)}</div>')
    return "".join(out)


def _detail(r: ScenarioResult) -> str:
    checks = "".join(
        f'<li class="{"yes" if c.passed else "no"}">{"pass" if c.passed else "FAIL"} '
        f"<b>{escape(c.name)}</b> {escape(c.detail)}</li>" for c in r.checks
    )
    judge = ""
    if r.judge and r.judge.scores:
        src = "stored verdict" if r.judge.from_cache else "live"
        judge = f"<p><b>Judge</b> ({escape(r.judge.model)}, {src})</p><ul class='checks'>" + "".join(
            f"<li><b>{escape(s.id)}</b> {s.score}/5: {escape(s.reason)}</li>" for s in r.judge.scores
        ) + "</ul>"
    elif r.judge:
        judge = f"<p><b>Judge</b> {r.judge.status}: {escape(r.judge.detail)}</p>"
    return f"<div class='detail-body'><div><ul class='checks'>{checks}</ul>{judge}</div><div>{_transcript(r)}</div></div>"


def render_html(run: RunResult, comparison: Comparison | None = None) -> str:
    rows_by_id = {row.scenario_id: row for row in comparison.rows} if comparison else {}
    lat_values = [v for r in run.results if (v := p95_latency(r)) is not None]
    lat_max = max(lat_values, default=1) or 1
    judged = [m for r in run.results if (m := judge_mean(r)) is not None]
    tiles = [
        (f"{run.pass_rate():.0%}", "scenarios passing"),
        (f"{sum(r.passed for r in run.results)} / {len(run.results)}", "passed"),
        (f"{sum(judged) / len(judged):.2f}" if judged else "-", f"judge mean ({len(judged)} judged)"),
        (f"{max(lat_values):.0f} ms" if lat_values else "-", "worst p95 latency"),
    ]
    banner = ""
    if comparison:
        n = len(comparison.regressions)
        banner = (f'<div class="banner bad">{n} regression{"s" if n != 1 else ""} against the baseline</div>' if n
                  else '<div class="banner ok">No regressions against the baseline</div>')
    rows = []
    for r in run.results:
        sid = r.scenario.id
        pill = '<span class="pill pass">pass</span>' if r.passed else '<span class="pill fail">fail</span>'
        row = rows_by_id.get(sid)
        change = row.status if row else ""
        change_pill = f' <span class="pill {change}">{change}</span>' if change in ("regressed", "fixed", "new") else ""
        # With a baseline, the comparison's reasons also cover judge drops and latency rises.
        why = row.reasons if row else [f"{c.name}: {c.detail}" for c in r.checks if not c.passed]
        p95 = p95_latency(r)
        bar = (f'<span class="bar"><i style="width:{100 * p95 / lat_max:.0f}%"></i></span> {p95:.0f} ms'
               if p95 is not None else "-")
        rows.append(
            f"<tr class='main'><td>{pill}{change_pill}</td><td><b>{escape(sid)}</b><br>"
            f"<span class='meta'>{escape(r.scenario.title)}</span></td>"
            f"<td class='num'>{_judge_cell(r)}</td><td class='num'>{bar}</td>"
            f"<td class='why'>{'<br>'.join(escape(w) for w in why) or '-'}</td></tr>"
            f"<tr><td></td><td colspan='4'><details><summary>transcript and checks</summary>{_detail(r)}</details></td></tr>"
        )
    tile_html = "".join(f"<div class='tile'><b>{v}</b><span>{label}</span></div>" for v, label in tiles)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>callgrade run {escape(run.run_id)}</title>
<style>{CSS}</style></head><body><header><h1>callgrade run {escape(run.run_id)}</h1>
<div class="meta">agent {escape(run.agent)} &middot; started {escape(run.started_at)} &middot; judge {escape(run.judge_model or "off")}</div>
<div class="tiles">{tile_html}</div></header><main>{banner}
<table><thead><tr><th>Result</th><th>Scenario</th><th>Judge</th><th>p95 latency</th><th>Why it failed</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></main></body></html>
"""


def write_report(run: RunResult, out_dir: Path, comparison: Comparison | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(run, out_dir / "results.json")
    write_junit(run, out_dir / "junit.xml")
    html_path = out_dir / "report.html"
    html_path.write_text(render_html(run, comparison))
    if comparison:
        (out_dir / "comparison.md").write_text(comparison.markdown() + "\n")
    return html_path
