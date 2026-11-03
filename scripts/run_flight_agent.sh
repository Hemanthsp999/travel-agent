#!/usr/bin/env bash
# Chat with the Flight Agent (start the Flight API first: scripts/run_mocks.sh --only flight).
# Usage: scripts/run_flight_agent.sh [--thread ID] [--no-llm] [--verbose] [--debug]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m travel_agent.flight_agent "$@"
