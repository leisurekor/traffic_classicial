"""Check whether the project can use the Graph AutoEncoder backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.pipeline.graph_binary_detection import _is_torch_available
from traffic_graph.pipeline.pcap_graph_smoke import _has_torch


def build_torch_env_report() -> dict[str, object]:
    """Build a stable JSON-friendly report for torch and backend availability."""

    torch_available = _has_torch() and _is_torch_available()
    report: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "torch_available": torch_available,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "pcap_graph_smoke_backend_expectation": "deterministic_fallback",
        "pcap_graph_experiment_backend_expectation": "deterministic_fallback",
        "merged_csv_graph_mode_backend_expectation": "requires_torch",
        "install_hint": (
            "python3 -m pip install -e '.[gae]' "
            "(inside a virtual environment when the system Python is externally managed)"
        ),
        "notes": [
            "Without torch, the real-PCAP smoke and PCAP experiment paths use the deterministic fallback scorer.",
            "With torch, those paths use the existing Graph AutoEncoder training and scoring stack.",
            "Merged-CSV graph mode requires torch and does not have a fallback backend.",
            "On Debian or Ubuntu system Python installs you may need python3-venv or python3.12-venv before creating a virtual environment.",
        ],
    }
    if not torch_available:
        return report

    import torch

    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_version"] = torch.version.cuda
    report["cuda_device_count"] = (
        int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    )
    report["pcap_graph_smoke_backend_expectation"] = "gae"
    report["pcap_graph_experiment_backend_expectation"] = "gae"
    report["merged_csv_graph_mode_backend_expectation"] = "gae"
    return report


def main() -> int:
    """Render the torch environment report as JSON."""

    print(json.dumps(build_torch_env_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
