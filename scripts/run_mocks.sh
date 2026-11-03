#!/usr/bin/env bash
# Start the Hotel API (8001) and Flight API (8002). Usage: scripts/run_mocks.sh [--only hotel|flight]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m mock_apis.run_mocks "$@"
