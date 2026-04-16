"""Smoke tests for torch dependency declaration and environment checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import unittest
from importlib.util import find_spec
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class TorchEnvironmentSetupTests(unittest.TestCase):
    """Validate the torch dependency contract and raw-data layout guidance."""

    def test_pyproject_declares_optional_gae_dependency(self) -> None:
        """The project should expose torch through a dedicated optional extra."""

        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            payload = tomllib.load(handle)
        extras = payload["project"]["optional-dependencies"]
        self.assertIn("gae", extras)
        self.assertTrue(any(str(item).startswith("torch>=") for item in extras["gae"]))

    def test_check_torch_env_script_reports_backend_expectations(self) -> None:
        """The torch-check script should emit stable JSON with backend hints."""

        script_path = PROJECT_ROOT / "scripts" / "check_torch_env.py"
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertIn("torch_available", payload)
        self.assertIn("torch_version", payload)
        self.assertIn("cuda_available", payload)
        self.assertIn("pcap_graph_smoke_backend_expectation", payload)
        self.assertIn("pcap_graph_experiment_backend_expectation", payload)
        self.assertIn("merged_csv_graph_mode_backend_expectation", payload)
        self.assertIn("install_hint", payload)

        torch_available = find_spec("torch") is not None
        self.assertEqual(bool(payload["torch_available"]), torch_available)
        if torch_available:
            self.assertEqual(payload["pcap_graph_smoke_backend_expectation"], "gae")
            self.assertEqual(payload["pcap_graph_experiment_backend_expectation"], "gae")
            self.assertEqual(payload["merged_csv_graph_mode_backend_expectation"], "gae")
        else:
            self.assertEqual(
                payload["pcap_graph_smoke_backend_expectation"],
                "deterministic_fallback",
            )
            self.assertEqual(
                payload["pcap_graph_experiment_backend_expectation"],
                "deterministic_fallback",
            )
            self.assertEqual(
                payload["merged_csv_graph_mode_backend_expectation"],
                "requires_torch",
            )

    def test_ciciot2023_readme_describes_layout_and_protocol(self) -> None:
        """The raw-data guide should document the PCAP layout and protocol rules."""

        readme_path = PROJECT_ROOT / "data" / "ciciot2023" / "README.md"
        self.assertTrue(readme_path.exists())
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("data/ciciot2023/pcap/benign/", content)
        self.assertIn("data/ciciot2023/pcap/malicious/recon/", content)
        self.assertIn("benign-only", content)
        self.assertIn("malicious", content)
        self.assertIn("--run-pcap-graph-experiment", content)
        self.assertIn("scripts/check_torch_env.py", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
