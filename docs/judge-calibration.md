# Judge calibration notes

The judge is `qwen2.5:3b` running on CPU through Ollama. It was picked because it's free and fits in
a few GB of RAM. `qwen2.5:7b` was tried too, but it ran out of memory on the 8 GB build machine.

The labelled set is `calibration/labels.yaml`. It has twelve transcripts: six from the good agent (v1)
and six from the regressed one (v2). Each rubric item has a pass or fail label. To re-run it:

```bash
callgrade calibrate calibration/labels.yaml --judge cache   # from stored verdicts, free
callgrade calibrate calibration/labels.yaml --judge live    # ask the model again
```

## What happened, in order

| Step | Change | Agreement | False passes | False fails |
|---|---|---|---|---|
| 1 | First prompt, score before reason | not measured; one clear miss seen by eye | | |
| 2 | Reason before score, numbered turns, "what counts is how the call ends" | 12/14 | 2 | 0 |
| 3 | Rewrote the two missed rubric questions to be concrete | 13/14 | 0 | 1 |
| 4 | Rewrote the read-back question again | 12/14 | 0 | 2 |
| 5 | Moved turn-order rules to a check and dropped a vague "voice" item | 12/12 | 0 | 0 |

**Step 1.** In `change-at-confirmation`, the caller says no to Monday and books Friday. The judge
scored the good agent 2/5 and said it "did not change the day". It had read the first booking
attempt and stopped there.

**Step 2.** Asking for the reason before the score, numbering the turns and telling the judge to
read to the end fixed that. Two false passes were left:

- `v2-book-cleaning`: the agent booked without reading the details back, and the judge gave 5.
- `v2-office-hours`: the agent answered with markdown bullets and a URL, and the judge called it
  "easy to understand".

**Steps 3 and 4.** Concrete rubric questions fixed the markdown case. The judge needed to be told the
reply is spoken by a TTS voice and that symbols get read out. The read-back question kept flipping.
Depending on the wording, the 3B model either missed a read-back that was there or invented one
that wasn't. It can't reliably tell which of two turns came first.

**Step 5.** Turn order is a fact about the transcript, not a judgement. It became a deterministic
check, `said_before_tool`, and the rubric question went back to what a judge is good at: did the
caller get the booking they asked for? The subjective "voice" item was also dropped. It scored 3 and 5
on near-identical turns, and `max_words_per_turn` covers the same ground without the noise.

## A prompt leak found along the way

Replay scenarios first used the recorded call's title as the scenario title, for example
"agent is slow and talks over them once". The judge reads the scenario title. It then marked the
*current* agent down for talking over the caller, which never happened in the replayed transcript.
Replay titles are now neutral ("Replay of recorded call ..."), and a test pins that.

## What this means

- Use checks for anything that can be stated as a rule: tool calls, arguments, order, forbidden
  phrases, format, timing. Use the judge for outcome questions.
- Count false passes separately. A judge that's too strict costs a re-run. A judge that's too lenient
  ships a broken agent.
- Twelve items from one labeller is enough to find problems like these, but not enough to prove the
  judge is good. A real deployment would label fifty or more calls, preferably by two people, and
  report agreement between the labellers too.
- The prompt was tuned on the same twelve items it's scored on, so 12/12 is an optimistic number.
  New labelled calls are the honest test.
