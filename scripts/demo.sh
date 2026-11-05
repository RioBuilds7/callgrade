#!/usr/bin/env bash
# The whole demo in one command: grade the good agent (v1), then the regressed one (v2),
# and print where the two HTML reports are. No API keys: the judge uses stored verdicts.
set -uo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen --quiet
PORT=8000 scripts/gate.sh v1 0 && v1_ok=yes || v1_ok=no
PORT=8001 scripts/gate.sh v2 1 && v2_ok=yes || v2_ok=no
echo
echo "v1 passes the gate:          $v1_ok"
echo "v2 is stopped by the gate:   $v2_ok"
echo "Reports:"
ls -1 runs/v1/*/report.html runs/v2/*/report.html | tail -2
[ "$v1_ok" = yes ] && [ "$v2_ok" = yes ]
