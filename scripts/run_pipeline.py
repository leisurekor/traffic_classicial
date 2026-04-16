"""Repository CLI entrypoint for the traffic graph pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

