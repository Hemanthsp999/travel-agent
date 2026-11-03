"""Start the Hotel API (8001) and Flight API (8002) as independent processes.

Usage: python -m mock_apis.run_mocks [--only hotel|flight]
Ctrl+C stops both. Each service can also be started alone, see the README.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

SERVICES = {"hotel": "mock_apis.hotel_api", "flight": "mock_apis.flight_api"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the mock hotel and flight APIs.")
    parser.add_argument("--only", choices=sorted(SERVICES), help="Start just one service")
    args = parser.parse_args()
    names = [args.only] if args.only else list(SERVICES)

    procs = {name: subprocess.Popen([sys.executable, "-m", SERVICES[name]]) for name in names}
    try:
        while all(p.poll() is None for p in procs.values()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            proc.wait()
    return max((p.returncode or 0) for p in procs.values())


if __name__ == "__main__":
    raise SystemExit(main())
