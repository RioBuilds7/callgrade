import socket
import threading
import time

import httpx
import pytest
import uvicorn

from callgrade.adapters import AgentReply
from callgrade.models import ToolCall, Transcript, Turn
from sample_agent.app import create_app


def make_transcript(*turns, ended_by="caller") -> Transcript:
    """Build a transcript from ("caller"|"agent", text[, tool_calls[, latency_ms]]) tuples."""
    out = []
    for t in turns:
        speaker, text = t[0], t[1]
        calls = [ToolCall(name=n, args=a) for n, a in (t[2] if len(t) > 2 else [])]
        latency = t[3] if len(t) > 3 else (10.0 if speaker == "agent" else None)
        out.append(Turn(speaker=speaker, text=text, tool_calls=calls, latency_ms=latency))
    return Transcript(scenario_id="t", turns=out, ended_by=ended_by)


class ScriptedAgent:
    """Fake agent that returns canned replies in order."""

    name = "scripted"

    def __init__(self, replies: list[AgentReply | Exception]):
        self.replies = list(replies)

    async def respond(self, session_id, caller_text):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def aclose(self):
        pass


def asgi_transport(variant: str, delay: float = 0.0) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=create_app(variant, delay=delay))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_agent():
    """Start the sample agent on a real port in a thread; yields a function variant -> URL."""
    servers = []

    def start(variant: str, delay: float = 0.0) -> str:
        port = _free_port()
        config = uvicorn.Config(create_app(variant, delay=delay), port=port, log_level="error")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        servers.append(server)
        return f"http://127.0.0.1:{port}"

    yield start
    for s in servers:
        s.should_exit = True
