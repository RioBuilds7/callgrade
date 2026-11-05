"""Ways to talk to the agent under test.

The bench only needs one thing from an agent: given a session and what the
caller just said, return what the agent says back (and any tools it called).

* HttpAgent speaks a tiny JSON protocol, so any agent stack (Vapi/Retell
  webhook, LangGraph, a homegrown state machine) can be wrapped in a few lines.
* OpenAIChatAgent drives any OpenAI-compatible chat endpoint directly (Ollama,
  vLLM, Groq, OpenAI), with tool results mocked from the agent config file.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import yaml

from .models import ToolCall


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    end_call: bool = False


class Agent(Protocol):
    name: str

    async def respond(self, session_id: str, caller_text: str) -> AgentReply: ...

    async def aclose(self) -> None: ...


class AgentError(Exception):
    pass


async def _post_with_retry(client: httpx.AsyncClient, url: str, payload: dict, retries: int) -> dict:
    """POST JSON, retrying connection errors and 5xx with a short backoff.

    4xx is the caller's fault, so it fails at once instead of retrying.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = await client.post(url, json=payload)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            last = AgentError(f"HTTP {r.status_code} from {url}")
        except httpx.HTTPStatusError as e:
            raise AgentError(f"HTTP {e.response.status_code} from {url}: {e.response.text[:200]}") from e
        except httpx.TransportError as e:
            last = AgentError(f"{type(e).__name__} talking to {url}")
        if attempt < retries:
            await asyncio.sleep(0.2 * 2**attempt)
    raise last or AgentError("request failed")


class HttpAgent:
    """POST {base_url}/turn  {"session_id", "text"} -> {"text", "tool_calls", "end_call"}."""

    def __init__(self, base_url: str, timeout: float = 15.0, retries: int = 2, transport: Any = None):
        self.name = base_url
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def respond(self, session_id: str, caller_text: str) -> AgentReply:
        data = await _post_with_retry(
            self.client, f"{self.base_url}/turn", {"session_id": session_id, "text": caller_text}, self.retries
        )
        return AgentReply(
            text=data.get("text", ""),
            tool_calls=[ToolCall(**c) for c in data.get("tool_calls", [])],
            end_call=bool(data.get("end_call", False)),
        )

    async def aclose(self) -> None:
        await self.client.aclose()


class OpenAIChatAgent:
    """Test a prompt + tools directly against an OpenAI-compatible model.

    Config file (YAML):
        model: qwen2.5:3b
        base_url: http://localhost:11434/v1   # or $AGENT_BASE_URL
        api_key_env: AGENT_API_KEY            # optional
        system: "You are the receptionist..."
        tools:
          - name: book_appointment
            description: ...
            parameters: {type: object, properties: {...}}
            mock_result: {"status": "booked"}
            ends_call: false
    """

    MAX_TOOL_ROUNDS = 4

    def __init__(self, config_path: str, transport: Any = None):
        cfg = yaml.safe_load(open(config_path))
        self.model = cfg["model"]
        self.name = f"openai:{self.model}"
        self.system = cfg["system"]
        self.tools = {t["name"]: t for t in cfg.get("tools", [])}
        base = os.environ.get("AGENT_BASE_URL", cfg.get("base_url", "http://localhost:11434/v1"))
        key = os.environ.get(cfg.get("api_key_env", "AGENT_API_KEY"), "none")
        self.url = base.rstrip("/") + "/chat/completions"
        self.client = httpx.AsyncClient(
            timeout=120.0, headers={"Authorization": f"Bearer {key}"}, transport=transport
        )
        self.sessions: dict[str, list[dict]] = {}

    def _tool_specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in self.tools.values()
        ]

    async def respond(self, session_id: str, caller_text: str) -> AgentReply:
        messages = self.sessions.setdefault(session_id, [{"role": "system", "content": self.system}])
        messages.append({"role": "user", "content": caller_text})
        calls: list[ToolCall] = []
        end_call = False
        for _ in range(self.MAX_TOOL_ROUNDS):
            payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
            if self.tools:
                payload["tools"] = self._tool_specs()
            data = await _post_with_retry(self.client, self.url, payload, retries=2)
            msg = data["choices"][0]["message"]
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not msg.get("tool_calls"):
                return AgentReply(text=(msg.get("content") or "").strip(), tool_calls=calls, end_call=end_call)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                args = fn.get("arguments") or "{}"
                args = json.loads(args) if isinstance(args, str) else args
                calls.append(ToolCall(name=fn["name"], args=args))
                tool = self.tools.get(fn["name"], {})
                end_call |= bool(tool.get("ends_call"))
                result = tool.get("mock_result", {"error": f"unknown tool {fn['name']}"})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(result)})
        raise AgentError(f"model kept calling tools for {self.MAX_TOOL_ROUNDS} rounds")

    async def aclose(self) -> None:
        await self.client.aclose()


def make_agent(target: str) -> Agent:
    """`http://...` -> HttpAgent; a .yaml path -> OpenAIChatAgent."""
    if target.startswith(("http://", "https://")):
        return HttpAgent(target)
    if target.endswith((".yaml", ".yml")):
        return OpenAIChatAgent(target)
    raise ValueError(f"don't know how to reach agent '{target}' (give a URL or an agent .yaml)")
