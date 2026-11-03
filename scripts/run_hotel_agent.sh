#!/usr/bin/env bash
# Chat with the Hotel Agent (start the Hotel API first: scripts/run_mocks.sh --only hotel).
# Usage: scripts/run_hotel_agent.sh [--thread ID] [--no-llm] [--verbose] [--debug]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m travel_agent.hotel_agent "$@"
