"""A small phone receptionist for a made-up dental office, used as the agent under test.

It is deliberately rule-based: no model, no keys, the same answer every time,
so the demo and CI cost nothing and fail only when the code changes.

AGENT_VARIANT=v1 is the good version. AGENT_VARIANT=v2 is the same agent after
a "harmless refactor" that ships five realistic regressions:
  1. books without reading the details back for confirmation
  2. answers the hours question with a markdown list (the TTS reads the asterisks)
  3. offers an appointment instead of transferring a dental emergency
  4. tells a caller about another patient's appointment
  5. is about a second slower on every turn
The demo shows the bench catching all five.

Run:  AGENT_VARIANT=v1 uvicorn sample_agent.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel

OFFICE = "Maple Street Dental"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Open slots per day, 24-hour clock. Wednesday is the staff training day.
SCHEDULE = {
    "monday": ["09:00", "14:00"],
    "tuesday": ["10:30", "15:00"],
    "wednesday": [],
    "thursday": ["09:00", "11:00"],
    "friday": ["08:30", "13:00"],
    "saturday": ["10:00"],
    "sunday": [],
}
# Existing bookings, used only to test that the agent never reveals them.
BOOKINGS = {"john parker": ("thursday", "14:00"), "dana whitfield": ("monday", "09:00")}
REASONS = {
    "cleaning": "cleaning", "clean": "cleaning", "checkup": "checkup", "check-up": "checkup",
    "check up": "checkup", "filling": "filling", "cavity": "filling", "crown": "crown",
    "whitening": "whitening", "toothache": "exam", "tooth hurts": "exam", "consult": "consultation",
}

YES = re.compile(r"\b(yes|yeah|yep|correct|that'?s right|sounds good|perfect|sure|that works|ok(ay)?)\b", re.I)
NO = re.compile(r"\b(no|nope|not right|wrong|actually)\b", re.I)
EMERGENCY = re.compile(r"\b(bleeding|swollen|swelling|severe|knocked out|emergency|can'?t stop)\b", re.I)
HUMAN = re.compile(r"\b(human|real person|someone real|front desk|receptionist|operator)\b", re.I)
GOODBYE = re.compile(r"\b(bye|goodbye|that'?s all|that is all|nothing else|no thanks)\b", re.I)

log = logging.getLogger("sample_agent")


def spoken_time(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def parse_day(text: str) -> str | None:
    for d in DAYS:
        if d in text.lower():
            return d
    return None


NUMBER_WORDS = {
    w: str(i) for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}


def parse_time(text: str, slots: list[str]) -> str | None:
    # Speech-to-text often writes times as words ("one PM"), so turn those into digits first.
    t = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group(1)], text.lower())
    t = re.sub(r"\b(\d+) thirty\b", r"\1:30", t)
    if "morning" in t:
        return next((s for s in slots if int(s[:2]) < 12), None)
    if "afternoon" in t:
        return next((s for s in slots if int(s[:2]) >= 12), None)
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
    if not m:
        return None
    h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and h < 12:
        h += 12
    if not ampm and h < 8:  # "at 3" on a phone call means 3 PM
        h += 12
    want = f"{h:02d}:{mins:02d}"
    return want if want in slots else None


def parse_name(text: str) -> str | None:
    m = re.search(r"\b(?i:my name is|this is|it'?s|name'?s|i'?m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z'-]+)+)", text)
    return m.group(1) if m else None


def parse_reason(text: str) -> str | None:
    t = text.lower()
    return next((v for k, v in REASONS.items() if k in t), None)


@dataclass
class Session:
    intent: str | None = None
    reason: str | None = None
    day: str | None = None
    offered_day: str | None = None
    time: str | None = None
    name: str | None = None
    confirming: bool = False
    done: bool = False
    slots: dict[str, list[str]] = field(default_factory=dict)
    refused_privacy: bool = False
    tool_calls: list[dict] = field(default_factory=list)

    def call(self, tool: str, **args) -> dict:
        self.tool_calls.append({"name": tool, "args": args})
        if tool == "check_availability":
            return {"slots": SCHEDULE[args["day"]]}
        return {"status": "ok"}


class Reply(BaseModel):
    text: str
    tool_calls: list[dict] = []
    end_call: bool = False


class Receptionist:
    def __init__(self, variant: str = "v1"):
        if variant not in ("v1", "v2"):
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.sessions: dict[str, Session] = {}

    def turn(self, session_id: str, text: str) -> Reply:
        s = self.sessions.setdefault(session_id, Session())
        s.tool_calls = []
        reply_text, end = self._respond(s, text)
        return Reply(text=reply_text, tool_calls=s.tool_calls, end_call=end)

    def _respond(self, s: Session, text: str) -> tuple[str, bool]:
        low = text.lower()
        v2 = self.variant == "v2"

        if EMERGENCY.search(text):
            if v2:
                s.intent, s.reason = "book", "exam"
                return "I'm sorry to hear that. Our next opening is Thursday at 9 AM. Would you like it?", False
            s.call("transfer_call", reason="dental emergency")
            return "That sounds urgent. I'm connecting you to our on-call dentist right now. Please stay on the line.", True

        if HUMAN.search(text):
            s.call("transfer_call", reason="caller asked for a person")
            return "Of course. I'm transferring you to the front desk now.", True

        other = next((n for n in BOOKINGS if n in low), None)
        if other and parse_name(text) is None and s.intent != "book":
            if v2:
                day, hhmm = BOOKINGS[other]
                return f"{other.title()} is booked on {day.title()} at {spoken_time(hhmm)}.", False
            s.refused_privacy = True
            return ("I'm sorry, I can't share details about another patient's appointment. "
                    "They're welcome to call us directly. Is there anything I can help you with?"), False

        if re.search(r"\b(hours|open|close)\b", low) and s.intent is None:
            if v2:
                return ("Here are our hours:\n- **Monday to Friday:** 8 AM to 5 PM\n- **Saturday:** 9 AM to 1 PM\n"
                        "- **Sunday:** closed\nYou can also see them at https://example.com/hours."), False
            return "We're open Monday to Friday, 8 AM to 5 PM, and Saturday 9 AM to 1 PM. Can I help you book a visit?", False

        if re.search(r"\binsurance\b|\baccept\b", low) and s.intent is None:
            return "We accept most major plans, including Delta Dental, Cigna, Aetna and MetLife. Would you like to book?", False

        if s.intent is None:
            if re.search(r"\bcancel\b", low):
                s.intent = "cancel"
            elif re.search(r"\b(reschedule|move|change)\b", low):
                s.intent = "reschedule"
            elif re.search(r"\b(book|appointment|schedule|come in|cleaning|checkup|check-up)\b", low) or parse_reason(text):
                s.intent = "book"
            elif GOODBYE.search(text):
                return f"Thanks for calling {OFFICE}. Goodbye!", True
            elif s.refused_privacy:
                return ("I understand. I can only discuss an appointment with the patient, "
                        "so please have him call us. Anything else I can do?"), False
            else:
                return f"Thanks for calling {OFFICE}. I can book, move or cancel an appointment. What can I do for you?", False
        elif s.done and (GOODBYE.search(text) or NO.search(text)):
            return f"Thanks for calling {OFFICE}. Have a great day!", True

        if s.intent == "cancel":
            return self._cancel(s, text)
        return self._book(s, text, v2)

    def _cancel(self, s: Session, text: str) -> tuple[str, bool]:
        s.name = s.name or parse_name(text)
        if not s.name:
            return "I can help with that. What's the name on the appointment?", False
        if not s.done:
            s.call("cancel_appointment", name=s.name)
            s.done = True
            return f"Done, {s.name.split()[0]}. Your appointment is cancelled. Anything else?", False
        return "Is there anything else I can help with?", False

    def _book(self, s: Session, text: str, v2: bool) -> tuple[str, bool]:
        if s.done:
            return "Is there anything else I can help with?", False
        s.reason = s.reason or parse_reason(text) or ("appointment" if s.intent == "reschedule" else None)
        s.name = s.name or parse_name(text)

        if s.confirming:
            if YES.search(text) and not NO.search(text):
                return self._finish(s)
            s.confirming, s.day, s.time = False, None, None
            return "No problem. What day would you like instead?", False

        if s.offered_day and YES.search(text):
            s.day, s.offered_day = s.offered_day, None
        new_day = parse_day(text)
        if new_day:
            s.day, s.time, s.offered_day = new_day, None, None

        if not s.reason:
            return "Sure. Is this for a cleaning, a checkup, or something that's bothering you?", False
        if not s.day:
            return "What day works best for you?", False

        if s.day not in s.slots:  # ask the calendar once per day mentioned
            s.slots[s.day] = s.call("check_availability", day=s.day)["slots"]
        slots = s.slots[s.day]
        if not slots:
            nxt = next(d for d in DAYS[DAYS.index(s.day) + 1:] + DAYS if SCHEDULE[d])
            closed, s.day, s.offered_day = s.day, None, nxt
            return f"We don't have openings on {closed.title()}. The next day with openings is {nxt.title()}. Would that work?", False
        s.time = s.time or parse_time(text, slots)
        if not s.time and len(slots) == 1 and YES.search(text) and not new_day:
            s.time = slots[0]
        if not s.time:
            if len(slots) == 1:
                return f"On {s.day.title()} I have {spoken_time(slots[0])} open. Does that work?", False
            options = " or ".join(spoken_time(x) for x in slots)
            return f"On {s.day.title()} I have {options}. Which would you prefer?", False
        if not s.name:
            return "Great. Can I get your full name, please?", False
        if v2:
            return self._finish(s)
        s.confirming = True
        what = "moving your appointment to" if s.intent == "reschedule" else f"a {s.reason} on"
        return f"To confirm: {what} {s.day.title()} at {spoken_time(s.time)} for {s.name}. Is that right?", False

    def _finish(self, s: Session) -> tuple[str, bool]:
        tool = "reschedule_appointment" if s.intent == "reschedule" else "book_appointment"
        s.call(tool, name=s.name, day=s.day, time=s.time, reason=s.reason)
        s.confirming, s.done = False, True
        return (f"You're all set, {s.name.split()[0]}. We'll see you {s.day.title()} at {spoken_time(s.time)}. "
                "Is there anything else?"), False


class TurnIn(BaseModel):
    session_id: str
    text: str


def create_app(variant: str | None = None, delay: float | None = None) -> FastAPI:
    variant = variant or os.environ.get("AGENT_VARIANT", "v1")
    agent = Receptionist(variant)
    app = FastAPI(title=f"{OFFICE} receptionist ({variant})")
    if delay is None:
        delay = 1.1 if variant == "v2" else 0.0

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "variant": variant}

    @app.post("/turn", response_model=Reply)
    async def turn(body: TurnIn) -> Reply:
        if delay:
            await asyncio.sleep(delay)
        reply = agent.turn(body.session_id, body.text)
        log.info(json.dumps({"session": body.session_id, "caller": body.text, "agent": reply.text,
                             "tools": reply.tool_calls, "end": reply.end_call}))
        return reply

    return app


app = create_app()
