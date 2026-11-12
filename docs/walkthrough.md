# How callgrade works: a walkthrough

This is for whoever has to explain the project in an interview. Read it with the code open. Each
part names the file and function to look at. The last section has three small exercises. Once you've
done them, you've changed every layer yourself.

## The pitch in three sentences

Voice agents break in ways that unit tests don't see: they say the right thing in a form the voice
can't speak, book before confirming, or get slower after a model swap. callgrade replays call scripts
against the agent, grades each call with cheap deterministic checks first and an LLM judge second,
and fails CI when a call that used to pass now fails. I didn't trust the judge blindly. I measured it
against labelled calls, and that changed the design.

## One run, start to finish

Command: `callgrade run --agent http://localhost:8000 --baseline baseline/results.json`

1. **Load scenarios.** `runner.py: load_scenarios` reads every YAML file under `scenarios/`, checks
   that the ids are unique and validates each one into a `Scenario` (`models.py`). Pydantic catches a
   typo in a scenario file before anything runs.
2. **Pick an adapter.** `adapters.py: make_agent` returns an `HttpAgent` for a URL or an
   `OpenAIChatAgent` for a `.yaml` config. Both have one method, `respond(session_id, text)`, so the
   runner doesn't care which kind of agent it's talking to.
3. **Play the call.** `runner.py: play` sends caller lines one at a time, times each reply
   (`latency_ms`) and records tool calls. It stops when the agent ends the call. If the agent errors,
   it records the error on the transcript and stops, and the run carries on with other scenarios.
   Scenarios run in parallel, capped by a semaphore (`--concurrency`).
4. **Checks.** `checks.py: run_checks` runs each expectation. Every check is a small function in a
   registry (`@check("name")`), so adding one takes about five lines. A failed check records which
   turn was to blame, and the report highlights that turn.
5. **Judge.** `judge.py: Judge.score` builds a prompt from the rubric and numbered transcript, looks
   up the hash of (model, prompt) in `judge_cache/`, and only calls the model on a miss. The reply is
   parsed strictly: every rubric id needs a 1-5 score. On a bad reply it retries once with the error
   fed back, then gives up with status `error`. A cache miss without a model is `skipped`, never a
   silent pass.
6. **Pass or fail.** A scenario passes if every check passes and, when the judge scored, every rubric
   item is at least `min_score` (default 4).
7. **Compare.** `compare.py: compare` matches scenarios to the baseline by id. It's a regression if
   the scenario passed before and fails now, the judge mean fell by more than 0.5, or p95 latency rose
   by more than 300 ms. The latency rule matters: most scenarios have no latency budget, but a whole
   second slower should still stop a release.
8. **Outputs.** `report.py` writes `results.json` (the next baseline), `junit.xml` (CI test tab),
   `report.html` (the dashboard) and `comparison.md` (also posted to the GitHub job summary).
9. **Exit code.** `cli.py`: 0 pass, 1 quality failure, 2 couldn't test. An unreachable agent is a 2,
   so on-call can tell "staging is down" from "the new prompt is worse".

## Design decisions you'll be asked about

**Why checks before an LLM judge?** Checks are free, instant and give the same answer every time. Most
real regressions are rule-shaped: wrong tool, wrong argument, missing confirmation, markdown in speech.
The judge is for outcome questions a rule can't answer well.

**Why cache judge verdicts?** Three reasons. CI shouldn't need an API key or cost money per push. The
same transcript should get the same grade, so a flaky judge can't fail a build. And the cache file
keeps the raw model reply, so you can audit why something passed.

**What happens when the agent changes and the cache misses?** In `auto` mode the judge is asked and the
new verdict is stored. CI uses `cache` mode with `--require-judge`, so a miss exits 2 and tells you to
re-record. Without that flag a miss shows as "skipped" in the report, never as a pass.

**How do you know the judge is any good?** `callgrade calibrate`. Read `docs/judge-calibration.md`
before an interview, because it's the best story in the repo. The short version: the 3B judge had two
false passes. Rubric rewording fixed one. The other was about turn order, which a small model can't
judge reliably, so it became a deterministic check. Say plainly that the set is small and the prompt
was tuned on it.

**Why a rule-based sample agent and not an LLM?** The bench is the product, not the agent. A
deterministic agent makes the demo free and repeatable, and the planted v2 bugs prove each check
works. `examples/llm-receptionist.yaml` shows the same scenarios against a real model.

**Why an HTTP protocol instead of integrating with Vapi or Retell directly?** Every customer's stack is
different. One small endpoint can wrap any of them in an afternoon, and the bench stays
vendor-neutral. A Vapi or Retell adapter would be one more class in `adapters.py`.

**Why is p95 "nearest rank"?** With five turns, an interpolated p95 is a number that never happened.
Nearest rank always returns a real observed latency (`checks.py: percentile`).

**What are recorded calls for?** Timing problems (dead air, talking over the caller) live in the audio
timestamps, not the words. `recorded.py: call_metrics` computes them from any diarized STT output.
`import-call` turns the caller's half into a replay, so today's agent faces yesterday's real caller.
Replaying the synthetic reschedule call found a real bug: the agent couldn't parse "one PM" and looped.
See commit 4c76259.

## A three-minute demo

1. `scripts/demo.sh`. Point out that v1 passes, v2 is stopped, and the exit codes are 0 and 1.
2. Open the v2 `report.html`. Show the red banner, then expand `emergency-transfer`. The check says
   `transfer_call` wasn't called in the agent's first reply, the judge gives 1/5, and that first
   reply (a Thursday slot offered to a bleeding child) has a red outline. The outline goes on the turn
   a check names. Checks like `ended_by` describe the whole call, so they don't outline anything.
3. Expand `office-hours`. Explain that the `speakable` check exists because the voice would read the
   asterisks out loud.
4. Open `docs/judge-calibration.md` and walk through the table.
5. Close with `.github/workflows/ci.yml`: this runs on every pull request with no secrets.

## Likely questions and short answers

- *How would you use this with a real customer?* Write twenty scenarios from their top call reasons,
  wrap their agent in the `/turn` endpoint on staging, and record a baseline. From then on, every
  prompt or model change runs the gate. Their real calls (redacted) become replay scenarios.
- *What about audio?* This works on transcripts. Next steps are TTS for the caller lines, the agent's
  real audio pipeline, and measuring time-to-first-audio. The checks and judge stay the same.
- *What doesn't it do?* The caller is scripted, so it can't improvise. Mocked tools return fixed results.
  The calibration set is small. All of this is listed in the README under Limits.
- *How long did it take?* Answer honestly. The README says it was built with help from AI coding
  agents. The point to make is that you can explain and change every part, which the exercises below prove.

## Three exercises (about an hour)

1. **Add a check.** Add `no_empty_turn` in `checks.py`: fail when the agent gives an empty reply. Use the `@check` decorator, add a test in `tests/test_checks.py` and use it in a
   scenario. Run `uv run pytest`.
2. **Break the agent and watch the gate.** In `sample_agent/app.py`, make v1 stop transferring on the
   word "swollen". Add a scenario where the caller says their face is swollen, run `scripts/demo.sh`,
   then fix it. Leave the `judge:` block out of the new scenario. The demo only uses stored judge
   verdicts, so a new scenario with a rubric has none and the gate exits 2 ("no judge verdict") before
   you see the regression. To judge it as well, start Ollama and run once with `--judge auto`. That
   stores the verdict.
3. **Re-label one calibration item.** Open `calibration/transcripts/v2-other-patient-privacy.json`,
   decide your own verdict, and run `callgrade calibrate calibration/labels.yaml --judge cache`. Then
   look at all twelve labels and change any you disagree with. They should be your labels before you
   talk about the numbers.

## Before an interview

- Redo the twelve labels in `calibration/labels.yaml` yourself (exercise 3), run
  `callgrade calibrate calibration/labels.yaml --judge cache`, and update `docs/judge-calibration.md` if
  the numbers change.
