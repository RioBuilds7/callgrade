#!/usr/bin/env bash
# Start one sample agent variant, grade it against the baseline, stop it.
# Usage: scripts/gate.sh v1|v2 [expected exit code]
# CI runs "gate.sh v1 0" (must pass) and "gate.sh v2 1" (the planted regressions must fail the gate).
set -uo pipefail
variant="${1:?v1 or v2}"
expected="${2:-0}"
port="${PORT:-8000}"

AGENT_VARIANT="$variant" uv run --frozen uvicorn sample_agent.app:app --port "$port" --log-level warning &
agent_pid=$!
trap 'kill $agent_pid 2>/dev/null' EXIT
for _ in $(seq 50); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 0.2; done

uv run --frozen callgrade run --agent "http://localhost:$port" --judge cache --require-judge \
  --baseline baseline/results.json --out "runs/$variant"
code=$?
echo "callgrade exit code $code (expected $expected)"
[ "$code" -eq "$expected" ]
