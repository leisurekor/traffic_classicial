"""Tests for the tabular-versus-graph binary detection comparison report."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.cli import build_parser, main
from traffic_graph.pipeline.binary_detection import BINARY_ATTACK_METRIC_FIELDS, BinaryAttackMetricRecord
from traffic_graph.pipeline.compare_binary_detection_runs import (
    load_binary_detection_run_bundle,
    compare_and_export_binary_detection_runs,
    compare_binary_detection_runs,
    load_binary_detection_run_summary,
)
from traffic_graph.pipeline.comparison_report import (
    COMPARISON_OVERALL_FIELDS,
    COMPARISON_PER_ATTACK_FIELDS,
    export_comparison_report,
    summarize_comparison,
)


def _metric_record(
    *,
    task_name: str,
    requested_attack_type: str,
    attack_labels: tuple[str, ...],
    sample_count: int,
    benign_count: int,
    attack_count: int,
    roc_auc: float,
    pr_auc: float,
    precision: float,
    recall: float,
    f1: float,
    false_positive_rate: float,
    threshold: float,
    score_median: float,
    score_q95: float,
) -> BinaryAttackMetricRecord:
    """Build a deterministic per-attack metric record for synthetic run files."""

    return BinaryAttackMetricRecord(
        task_name=task_name,
        requested_attack_type=requested_attack_type,
        attack_labels=attack_labels,
        sample_count=sample_count,
        benign_count=benign_count,
        attack_count=attack_count,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        threshold=threshold,
        score_min=score_median - 0.2,
        score_q25=score_median - 0.1,
        score_median=score_median,
        score_q75=score_median + 0.1,
        score_q95=score_q95,
        score_max=score_q95 + 0.2,
        score_mean=score_median,
        score_std=0.1,
        benign_score_mean=score_median - 0.05,
        benign_score_median=score_median - 0.02,
        attack_score_mean=score_median + 0.05,
        attack_score_median=score_median + 0.02,
        notes=(f"Synthetic task {task_name}.",),
    )


def _write_run_bundle(
    root: Path,
    *,
    run_id: str,
    dataset_name: str,
    backend_name: str,
    overall_metrics: dict[str, float | None],
    per_attack_metrics: tuple[BinaryAttackMetricRecord, ...],
) -> Path:
    """Write a minimal binary-detection run bundle for comparison tests."""

    run_directory = root / run_id / "20260410T010203Z"
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics_summary.json"
    per_attack_path = run_directory / "per_attack_metrics.csv"
    manifest_path = run_directory / "manifest.json"

    attack_score_summaries = {
        record.task_name: {
            "task_name": record.task_name,
            "requested_attack_type": record.requested_attack_type,
            "attack_labels": list(record.attack_labels),
            "score_summary": {
                "count": record.sample_count,
                "mean": record.score_mean,
                "std": record.score_std,
                "min": record.score_min,
                "q25": record.score_q25,
                "median": record.score_median,
                "q75": record.score_q75,
                "q95": record.score_q95,
                "max": record.score_max,
            },
            "benign_score_summary": {
                "count": record.benign_count,
                "mean": record.benign_score_mean,
                "median": record.benign_score_median,
            },
            "attack_score_summary": {
                "count": record.attack_count,
                "mean": record.attack_score_mean,
                "median": record.attack_score_median,
            },
            "per_attack_label_breakdown": [],
        }
        for record in per_attack_metrics
    }
    report_payload = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "source_path": "/tmp/Merged01.csv",
        "created_at": "20260410T010203Z",
        "threshold_percentile": 95.0,
        "threshold": 0.5,
        "feature_columns": ["FeatureA", "FeatureB"],
        "model_n_components": 10,
        "train_sample_count": 100,
        "train_benign_count": 100,
        "overall_metrics": overall_metrics,
        "train_score_summary": {"count": 100, "mean": 0.2, "median": 0.2, "q95": 0.4},
        "overall_score_summary": {"count": 50, "mean": 0.5, "median": 0.5, "q95": 0.8},
        "per_attack_metrics": [record.to_dict() for record in per_attack_metrics],
        "attack_score_summaries": attack_score_summaries,
        "input_artifacts": {
            "model_mode": backend_name,
            "graph_score_reduction": "flow_p90" if backend_name == "graph" else "pca_reconstruction",
            "scorer_role": "fallback" if backend_name == "graph" else "tabular_control",
        },
        "artifact_paths": {},
        "notes": [f"{backend_name} synthetic run"],
    }
    metrics_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with per_attack_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BINARY_ATTACK_METRIC_FIELDS))
        writer.writeheader()
        for record in per_attack_metrics:
            writer.writerow(record.to_csv_row())

    manifest_payload = {
        "run_id": run_id,
        "timestamp": "20260410T010203Z",
        "split": "eval",
        "base_directory": root.as_posix(),
        "run_directory": run_directory.as_posix(),
        "layout_directory": run_directory.as_posix(),
        "artifact_paths": {
            "metrics_summary_json": metrics_path.as_posix(),
            "per_attack_metrics_csv": per_attack_path.as_posix(),
            "overall_scores_csv": (run_directory / "overall_scores.csv").as_posix(),
            "attack_scores_csv": (run_directory / "attack_scores.csv").as_posix(),
        },
        "row_counts": {"per_attack_metrics_csv": len(per_attack_metrics)},
        "notes": [f"{backend_name} synthetic run"],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory


class BinaryDetectionComparisonTests(unittest.TestCase):
    """Smoke tests for the binary detection comparison report pipeline."""

    def setUp(self) -> None:
        """Prepare synthetic tabular and graph run bundles."""

        tabular_metrics = (
            _metric_record(
                task_name="recon",
                requested_attack_type="Recon",
                attack_labels=("RECON-HOSTDISCOVERY",),
                sample_count=20,
                benign_count=10,
                attack_count=10,
                roc_auc=0.80,
                pr_auc=0.72,
                precision=0.75,
                recall=0.30,
                f1=0.43,
                false_positive_rate=0.18,
                threshold=0.50,
                score_median=0.42,
                score_q95=0.68,
            ),
            _metric_record(
                task_name="web-based",
                requested_attack_type="Web-based",
                attack_labels=("XSS", "SQLINJECTION"),
                sample_count=16,
                benign_count=8,
                attack_count=8,
                roc_auc=0.76,
                pr_auc=0.61,
                precision=0.62,
                recall=0.20,
                f1=0.30,
                false_positive_rate=0.15,
                threshold=0.50,
                score_median=0.40,
                score_q95=0.65,
            ),
            _metric_record(
                task_name="all_malicious",
                requested_attack_type="All-Malicious",
                attack_labels=("RECON-HOSTDISCOVERY", "XSS", "DDOS-ICMP_FLOOD"),
                sample_count=40,
                benign_count=20,
                attack_count=20,
                roc_auc=0.89,
                pr_auc=0.84,
                precision=0.83,
                recall=0.70,
                f1=0.76,
                false_positive_rate=0.11,
                threshold=0.50,
                score_median=0.55,
                score_q95=0.82,
            ),
        )
        graph_metrics = (
            _metric_record(
                task_name="recon",
                requested_attack_type="Recon",
                attack_labels=("RECON-HOSTDISCOVERY",),
                sample_count=20,
                benign_count=10,
                attack_count=10,
                roc_auc=0.84,
                pr_auc=0.77,
                precision=0.79,
                recall=0.45,
                f1=0.57,
                false_positive_rate=0.16,
                threshold=0.50,
                score_median=0.47,
                score_q95=0.74,
            ),
            _metric_record(
                task_name="web-based",
                requested_attack_type="Web-based",
                attack_labels=("XSS", "SQLINJECTION"),
                sample_count=16,
                benign_count=8,
                attack_count=8,
                roc_auc=0.81,
                pr_auc=0.70,
                precision=0.68,
                recall=0.33,
                f1=0.44,
                false_positive_rate=0.12,
                threshold=0.50,
                score_median=0.45,
                score_q95=0.70,
            ),
            _metric_record(
                task_name="all_malicious",
                requested_attack_type="All-Malicious",
                attack_labels=("RECON-HOSTDISCOVERY", "XSS", "DDOS-ICMP_FLOOD"),
                sample_count=40,
                benign_count=20,
                attack_count=20,
                roc_auc=0.91,
                pr_auc=0.87,
                precision=0.86,
                recall=0.77,
                f1=0.81,
                false_positive_rate=0.09,
                threshold=0.50,
                score_median=0.58,
                score_q95=0.86,
            ),
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.tabular_run = _write_run_bundle(
            root,
            run_id="tabular-run",
            dataset_name="Merged01",
            backend_name="tabular",
            overall_metrics={
                "roc_auc": 0.90,
                "pr_auc": 0.88,
                "precision": 0.84,
                "recall": 0.68,
                "f1": 0.75,
                "false_positive_rate": 0.13,
            },
            per_attack_metrics=tabular_metrics,
        )
        self.graph_run = _write_run_bundle(
            root,
            run_id="graph-run",
            dataset_name="Merged01",
            backend_name="graph",
            overall_metrics={
                "roc_auc": 0.93,
                "pr_auc": 0.91,
                "precision": 0.87,
                "recall": 0.73,
                "f1": 0.79,
                "false_positive_rate": 0.10,
            },
            per_attack_metrics=graph_metrics,
        )

    def tearDown(self) -> None:
        """Release temporary directories."""

        self.temp_dir.cleanup()

    def test_load_binary_detection_run_summary_accepts_wrapped_pcap_metrics(self) -> None:
        """PCAP experiment metrics wrappers should normalize into comparison summaries."""

        metrics_path = self.graph_run / "metrics_summary.json"
        original_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        wrapped_payload = {
            "pcap_experiment": {
                "run_id": original_payload["run_id"],
                "experiment_label": "wrapped-pcap-graph",
                "timestamp": original_payload["created_at"],
                "mode": "binary_evaluation",
                "backend": "gae",
                "benign_inputs": ["/tmp/benign-a.pcap"],
                "malicious_inputs": ["/tmp/recon-a.pcap"],
                "packet_limit": 20000,
                "window_size": 10,
                "graph_score_reduction": "flow_p90",
                "threshold_percentile": original_payload["threshold_percentile"],
                "anomaly_threshold": original_payload["threshold"],
                "overall_metrics": original_payload["overall_metrics"],
                "train_graph_score_summary": original_payload["train_score_summary"],
                "graph_score_summary": original_payload["overall_score_summary"],
                "per_attack_metrics": original_payload["per_attack_metrics"],
                "notes": ["wrapped pcap comparison payload"],
            }
        }
        metrics_path.write_text(
            json.dumps(wrapped_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary = load_binary_detection_run_summary(self.graph_run, backend_name="graph")

        self.assertEqual(summary.run_id, "graph-run")
        self.assertEqual(summary.dataset_name, "pcap_experiment")
        self.assertEqual(summary.threshold, 0.5)
        self.assertAlmostEqual(float(summary.overall_metrics["f1"] or 0.0), 0.79)
        self.assertEqual(int(summary.train_score_summary["count"]), 100)
        self.assertEqual(summary.input_artifacts.get("graph_score_reduction"), "flow_p90")
        self.assertIn("/tmp/benign-a.pcap", summary.source_path)

    def test_load_binary_detection_run_bundle_normalizes_shared_metadata(self) -> None:
        """The bundle helper should expose one stable metadata view for scripts."""

        comparison_summary_path = self.graph_run / "comparison_summary.json"
        comparison_summary_path.write_text(
            json.dumps(
                {
                    "experiment_label": "wrapped-pcap-graph",
                    "graph_score_reduction": "hybrid_max_rank_flow_node_max",
                    "threshold": 0.5,
                    "benign_train_graph_count": 12,
                    "benign_test_graph_count": 4,
                    "malicious_test_graph_count": 9,
                    "worst_malicious_source_name": "BrowserHijacking",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        pcap_summary_path = self.graph_run / "pcap_experiment_summary.json"
        pcap_summary_path.write_text(
            json.dumps(
                {
                    "experiment_label": "wrapped-pcap-graph",
                    "benign_inputs": ["/tmp/benign-a.pcap"],
                    "malicious_inputs": ["/tmp/recon-a.pcap"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        bundle = load_binary_detection_run_bundle(self.graph_run, backend_name="graph")

        self.assertEqual(bundle.metadata.experiment_label, "wrapped-pcap-graph")
        self.assertEqual(bundle.metadata.reduction_method, "hybrid_max_rank_flow_node_max")
        self.assertEqual(bundle.metadata.scorer_role, "default_candidate")
        self.assertEqual(bundle.metadata.benign_train_graph_count, 12)
        self.assertEqual(bundle.metadata.worst_malicious_source_name, "BrowserHijacking")
        self.assertEqual(bundle.metadata.benign_inputs, ("/tmp/benign-a.pcap",))
        self.assertIn("recon", bundle.per_attack_metrics_by_task)

    def test_run_summaries_load_and_join(self) -> None:
        """Run directories should load into comparable typed summaries."""

        tabular_summary = load_binary_detection_run_summary(self.tabular_run, backend_name="tabular")
        graph_summary = load_binary_detection_run_summary(self.graph_run, backend_name="graph")
        self.assertEqual(tabular_summary.run_id, "tabular-run")
        self.assertEqual(graph_summary.run_id, "graph-run")
        self.assertEqual(tabular_summary.dataset_name, "Merged01")
        self.assertEqual(len(tabular_summary.per_attack_metrics), 3)

    def test_comparison_metrics_are_joined_and_deltas_are_correct(self) -> None:
        """The comparison report should compute stable deltas for overall and attack metrics."""

        report = compare_binary_detection_runs(self.tabular_run, self.graph_run)
        self.assertEqual(len(report.overall_metrics), 6)
        self.assertEqual(len(report.per_attack_metrics), 3)
        overall_lookup = {row.metric_name: row for row in report.overall_metrics}
        self.assertAlmostEqual(overall_lookup["roc_auc"].delta or 0.0, 0.03)
        self.assertAlmostEqual(overall_lookup["false_positive_rate"].delta or 0.0, -0.03)

        attack_lookup = {row.task_name: row for row in report.per_attack_metrics}
        self.assertTrue(attack_lookup["recon"].highlighted)
        self.assertTrue(attack_lookup["web-based"].highlighted)
        self.assertTrue(attack_lookup["all_malicious"].highlighted)
        self.assertAlmostEqual(attack_lookup["recon"].delta_recall or 0.0, 0.15)
        self.assertAlmostEqual(attack_lookup["web-based"].delta_f1 or 0.0, 0.14)
        self.assertAlmostEqual(attack_lookup["all_malicious"].delta_pr_auc or 0.0, 0.03)

        rendered = summarize_comparison(report)
        self.assertIn("Binary detection comparison", rendered)
        self.assertIn("recon", rendered)
        self.assertIn("web-based", rendered)

    def test_export_files_have_stable_fields(self) -> None:
        """Exported comparison files should use stable column order and manifest layout."""

        report = compare_binary_detection_runs(self.tabular_run, self.graph_run)
        with tempfile.TemporaryDirectory() as export_dir:
            export_result = export_comparison_report(
                report,
                export_dir,
                export_markdown=True,
                timestamp="20260410T020304Z",
            )
            self.assertTrue(Path(export_result.manifest_path).exists())
            self.assertTrue(Path(export_result.summary_path).exists())
            self.assertTrue(Path(export_result.overall_metrics_path).exists())
            self.assertTrue(Path(export_result.per_attack_metrics_path).exists())
            self.assertTrue(Path(export_result.markdown_path or "").exists())
            overall_frame = pd.read_csv(export_result.overall_metrics_path)
            per_attack_frame = pd.read_csv(export_result.per_attack_metrics_path)
            self.assertEqual(list(overall_frame.columns), list(COMPARISON_OVERALL_FIELDS))
            self.assertEqual(list(per_attack_frame.columns), list(COMPARISON_PER_ATTACK_FIELDS))
            summary_payload = json.loads(Path(export_result.summary_path).read_text(encoding="utf-8"))
            self.assertIn("highlighted_attacks", summary_payload)
            self.assertIn("recon", summary_payload["highlighted_attacks"])
            self.assertEqual(summary_payload["tabular_run"]["scorer_role"], "tabular_control")
            self.assertEqual(summary_payload["graph_run"]["scorer_role"], "fallback")
            markdown_text = Path(export_result.markdown_path or "").read_text(encoding="utf-8")
            self.assertIn("`tabular_control`", markdown_text)
            self.assertIn("`fallback`", markdown_text)

    def test_cli_can_compare_two_exported_runs(self) -> None:
        """The CLI should compare two exported binary detection runs end to end."""

        parser = build_parser()
        parsed = parser.parse_args(
            [
                "--compare-binary-detection-runs",
                "--tabular-run-dir",
                str(self.tabular_run),
                "--graph-run-dir",
                str(self.graph_run),
                "--comparison-output-dir",
                str(Path(self.temp_dir.name) / "comparison-out"),
                "--comparison-markdown",
            ]
        )
        self.assertTrue(parsed.compare_binary_detection_runs)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--compare-binary-detection-runs",
                    "--tabular-run-dir",
                    str(self.tabular_run),
                    "--graph-run-dir",
                    str(self.graph_run),
                    "--comparison-output-dir",
                    str(Path(self.temp_dir.name) / "comparison-out"),
                    "--comparison-markdown",
                ]
            )
        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Binary detection comparison summary", rendered)
        self.assertIn("Saved comparison report", rendered)

    def test_unified_graph_scorer_summary_exports_graph_scorer_roles(self) -> None:
        """The unified scorer summary entrypoint should emit the current role table."""

        def _clone_graph_run(name: str, reduction: str) -> Path:
            target = Path(self.temp_dir.name) / name / "20260410T010203Z"
            shutil.copytree(self.graph_run, target)
            (target / "comparison_summary.json").write_text(
                json.dumps(
                    {
                        "experiment_label": f"{name}-label",
                        "graph_score_reduction": reduction,
                        "threshold": 0.5,
                        "benign_train_graph_count": 12,
                        "benign_test_graph_count": 4,
                        "malicious_test_graph_count": 9,
                        "worst_malicious_source_name": "BrowserHijacking",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return target

        flow_run = _clone_graph_run("flow-run", "flow_p90")
        hybrid_run = _clone_graph_run(
            "hybrid-run",
            "hybrid_max_rank_flow_node_max",
        )
        decision_run = _clone_graph_run(
            "decision-run",
            "decision_topk_flow_node",
        )
        relation_run = _clone_graph_run(
            "relation-run",
            "relation_max_flow_server_count",
        )
        structural_run = _clone_graph_run(
            "structural-run",
            "structural_fig_max",
        )
        output_dir = Path(self.temp_dir.name) / "paper-summary"

        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR)
        subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "summarize_graph_scorers.py"),
                "--flow-run-dir",
                str(flow_run),
                "--hybrid-run-dir",
                str(hybrid_run),
                "--decision-run-dir",
                str(decision_run),
                "--relation-run-dir",
                str(relation_run),
                "--structural-run-dir",
                str(structural_run),
                "--tabular-run-dir",
                str(self.tabular_run),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            env=env,
        )

        summary_path = output_dir / "graph_scorer_family_summary.csv"
        self.assertTrue(summary_path.exists())
        with summary_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["scorer_name"] for row in rows],
            [
                "hybrid_max_rank_flow_node_max",
                "flow_p90",
                "decision_topk_flow_node",
            ],
        )
        role_lookup = {row["scorer_name"]: row["scorer_role"] for row in rows}
        note_lookup = {row["scorer_name"]: row["notes"] for row in rows}
        self.assertEqual(role_lookup["hybrid_max_rank_flow_node_max"], "default_candidate")
        self.assertEqual(role_lookup["flow_p90"], "fallback")
        self.assertEqual(role_lookup["decision_topk_flow_node"], "experimental")
        self.assertIn("Recon", note_lookup["decision_topk_flow_node"])

    def test_graph_mainline_summary_exports_scorer_roles(self) -> None:
        """The mainline graph-vs-tabular summary should carry scorer-role fields."""

        def _clone_run(name: str, reduction: str, *, backend_name: str) -> Path:
            source = self.graph_run if backend_name == "graph" else self.tabular_run
            target = Path(self.temp_dir.name) / name / "20260410T010203Z"
            shutil.copytree(source, target)
            (target / "comparison_summary.json").write_text(
                json.dumps(
                    {
                        "experiment_label": f"{name}-label",
                        "graph_score_reduction": reduction,
                        "threshold": 0.5,
                        "benign_train_graph_count": 12,
                        "benign_test_graph_count": 4,
                        "malicious_test_graph_count": 9,
                        "worst_malicious_source_name": "BrowserHijacking",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return target

        flow_run = _clone_run("flow-run-mainline", "flow_p90", backend_name="graph")
        hybrid_run = _clone_run(
            "hybrid-run-mainline",
            "hybrid_max_rank_flow_node_max",
            backend_name="graph",
        )
        tabular_run = _clone_run("tabular-run-mainline", "tabular_graphsummary", backend_name="tabular")
        output_dir = Path(self.temp_dir.name) / "mainline-summary"

        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR)
        subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "summarize_graph_mainline_hybrid.py"),
                "--flow-run-dir",
                str(flow_run),
                "--hybrid-run-dir",
                str(hybrid_run),
                "--tabular-run-dir",
                str(tabular_run),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            env=env,
        )

        followup_path = output_dir / "graph_vs_tabular_hybrid_followup.csv"
        self.assertTrue(followup_path.exists())
        with followup_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("scorer_role", rows[0])
        role_lookup = {row["run_label"]: row["scorer_role"] for row in rows}
        self.assertEqual(role_lookup["graph_flow_p90"], "fallback")
        self.assertEqual(role_lookup["graph_hybrid_max_rank_flow_node_max"], "default_candidate")
        self.assertEqual(role_lookup["tabular_graphsummary"], "tabular_control")

        markdown_path = output_dir / "graph_reduction_mainline_hybrid_ab.md"
        markdown_text = markdown_path.read_text(encoding="utf-8")
        self.assertIn("`graph_flow_p90` (`fallback`)", markdown_text)
        self.assertIn(
            "`graph_hybrid_max_rank_flow_node_max` (`default_candidate`)",
            markdown_text,
        )
        self.assertIn("`tabular_graphsummary` (`tabular_control`)", markdown_text)

    def test_experiments_doc_links_analysis_notes(self) -> None:
        """The experiments doc should point readers at the short analysis notes."""

        docs_text = (ROOT_DIR / "docs" / "experiments.md").read_text(encoding="utf-8")
        self.assertIn("analysis_notes.md", docs_text)


if __name__ == "__main__":
    unittest.main()
