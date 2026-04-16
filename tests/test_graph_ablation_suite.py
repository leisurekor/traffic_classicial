"""Smoke tests for the graph ablation suite."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pandas as pd
import sys

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_graph.cli import build_parser
from traffic_graph.config import AssociationEdgeConfig, GraphConfig
from traffic_graph.data import BinaryExperimentConfig, HeldOutAttackProtocolConfig
from traffic_graph.features.feature_pack import build_model_feature_view
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EndpointNode,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.graph.nx_compat import MultiDiGraph
from traffic_graph.pipeline.binary_detection import (
    BinaryAttackMetricRecord,
    BinaryDetectionExportResult,
    BinaryDetectionReport,
)
from traffic_graph.pipeline.graph_ablation import (
    GRAPH_ABLATION_SUMMARY_FIELDS,
    run_graph_ablation_suite,
)
from traffic_graph.pipeline.graph_binary_detection import GraphBinaryDetectionConfig


def _make_toy_graph() -> InteractionGraph:
    """Create a tiny interaction graph for feature-view smoke tests."""

    graph = MultiDiGraph()
    node_a = EndpointNode(
        node_id="client:10.0.0.1:1234:tcp",
        endpoint_type="client",
        ip="10.0.0.1",
        port=1234,
        proto="tcp",
    )
    node_b = EndpointNode(
        node_id="server:10.0.0.2:80:tcp",
        endpoint_type="server",
        ip="10.0.0.2",
        port=80,
        proto="tcp",
    )
    graph.add_node(node_a.node_id, endpoint_type=node_a.endpoint_type, ip=node_a.ip, port=node_a.port, proto=node_a.proto)
    graph.add_node(node_b.node_id, endpoint_type=node_b.endpoint_type, ip=node_b.ip, port=node_b.port, proto=node_b.proto)
    edge = CommunicationEdge(
        edge_id="flow-1",
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        edge_type="communication",
        logical_flow_id="flow-1",
        pkt_count=10,
        byte_count=1000,
        duration=1.5,
        flow_count=1,
        is_aggregated=False,
        source_flow_ids=("flow-1",),
    )
    graph.add_edge(
        edge.source_node_id,
        edge.target_node_id,
        key=edge.edge_id,
        edge_type=edge.edge_type,
        logical_flow_id=edge.logical_flow_id,
        pkt_count=edge.pkt_count,
        byte_count=edge.byte_count,
        duration=edge.duration,
        flow_count=edge.flow_count,
        is_aggregated=edge.is_aggregated,
        source_flow_ids=edge.source_flow_ids,
    )
    nodes = (node_a, node_b)
    edges = (edge,)
    return InteractionGraph(
        window_index=0,
        window_start=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
        window_end=pd.Timestamp("2026-04-10T00:01:00Z").to_pydatetime(),
        graph=graph,
        nodes=nodes,
        edges=edges,
        stats=build_interaction_graph_stats(nodes, edges),
    )


def _make_metric(task_name: str, recall: float, pr_auc: float, f1: float) -> BinaryAttackMetricRecord:
    """Create a stable per-attack metric record for synthetic reports."""

    return BinaryAttackMetricRecord(
        task_name=task_name,
        requested_attack_type=task_name,
        attack_labels=(task_name.upper(),),
        sample_count=10,
        benign_count=5,
        attack_count=5,
        roc_auc=0.5,
        pr_auc=pr_auc,
        precision=0.75,
        recall=recall,
        f1=f1,
        false_positive_rate=0.25,
        threshold=0.5,
        score_min=0.1,
        score_q25=0.2,
        score_median=0.3,
        score_q75=0.4,
        score_q95=0.5,
        score_max=0.6,
        score_mean=0.3,
        score_std=0.1,
        benign_score_mean=0.2,
        benign_score_median=0.2,
        attack_score_mean=0.4,
        attack_score_median=0.4,
        notes=(),
    )


def _fake_run_graph_binary_detection_experiment(source, output_dir, **kwargs):
    """Return a synthetic binary-detection report and export bundle."""

    graph_config = kwargs["graph_config"]
    config_id = (
        f"assoc-{int(bool(graph_config.use_association_edges))}"
        f"-struct-{int(bool(graph_config.use_graph_structural_features))}"
        f"-win-{int(graph_config.window_size)}"
    )
    run_dir = Path(output_dir) / config_id
    report = BinaryDetectionReport(
        run_id=f"run-{config_id}",
        dataset_name="Merged01",
        source_path=Path(source).as_posix() if isinstance(source, (str, Path)) else "frame",
        created_at=str(kwargs["timestamp"]),
        threshold_percentile=float(kwargs["threshold_percentile"]),
        threshold=0.5,
        feature_columns=("FeatureA", "FeatureB"),
        model_n_components=4,
        train_sample_count=8,
        train_benign_count=8,
        overall_metrics={
            "roc_auc": 0.9 if graph_config.use_graph_structural_features else 0.8,
            "pr_auc": 0.95,
            "precision": 0.9,
            "recall": 0.85,
            "f1": 0.87,
            "false_positive_rate": 0.1,
        },
        train_score_summary={"mean": 0.2, "median": 0.2, "q95": 0.3},
        overall_score_summary={"mean": 0.4, "median": 0.4, "q95": 0.5},
        per_attack_metrics=(
            _make_metric("recon", 0.2 if not graph_config.use_graph_structural_features else 0.7, 0.6, 0.3),
            _make_metric("web-based", 0.15 if not graph_config.use_graph_structural_features else 0.55, 0.4, 0.25),
            _make_metric("all_malicious", 0.8, 0.9, 0.85),
        ),
        attack_score_summaries={},
        input_artifacts={
            "graph_model_config": graph_config.to_dict(),
            "source": "synthetic",
        },
        artifact_paths={},
        notes=(
            f"synthetic run for {config_id}",
            f"window_size={graph_config.window_size}",
        ),
    )
    export_result = BinaryDetectionExportResult(
        run_id=report.run_id,
        dataset_name=report.dataset_name,
        created_at=report.created_at,
        output_directory=run_dir.as_posix(),
        manifest_path=(run_dir / "manifest.json").as_posix(),
        metrics_summary_path=(run_dir / "metrics_summary.json").as_posix(),
        per_attack_metrics_path=(run_dir / "per_attack_metrics.csv").as_posix(),
        overall_scores_path=(run_dir / "overall_scores.csv").as_posix(),
        attack_scores_path=(run_dir / "attack_scores.csv").as_posix(),
        artifact_paths={
            "manifest_json": (run_dir / "manifest.json").as_posix(),
            "metrics_summary_json": (run_dir / "metrics_summary.json").as_posix(),
        },
        row_counts={},
        notes=[],
    )
    return report, export_result


class GraphAblationSuiteTests(unittest.TestCase):
    """Smoke tests for the graph ablation sweep helpers."""

    def test_feature_view_toggle_controls_structural_features(self) -> None:
        """Disabling structural features should shorten the node feature schema."""

        graph = _make_toy_graph()
        with_structural = build_model_feature_view(
            graph,
            include_graph_structural_features=True,
        )
        without_structural = build_model_feature_view(
            graph,
            include_graph_structural_features=False,
        )
        self.assertGreater(
            len(with_structural.node_features.field_names),
            len(without_structural.node_features.field_names),
        )
        self.assertNotIn("total_degree", without_structural.node_features.field_names)

    def test_graph_ablation_suite_exports_summary(self) -> None:
        """The suite should run a full Cartesian sweep and write stable summaries."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = run_graph_ablation_suite(
                "Merged01.csv",
                Path(tmp_dir) / "ablation",
                binary_experiment_config=BinaryExperimentConfig(),
                heldout_protocol_config=HeldOutAttackProtocolConfig(),
                window_sizes=(30, 60),
                use_association_edges_options=(False, True),
                use_graph_structural_features_options=(False, True),
                graph_model_config=GraphBinaryDetectionConfig(),
                run_experiment_fn=_fake_run_graph_binary_detection_experiment,
                timestamp="20260410T010203Z",
            )

            self.assertEqual(len(suite.summary_records), 8)
            self.assertEqual(len(suite.run_results), 8)
            self.assertTrue(Path(suite.summary_csv_path).exists())
            self.assertTrue(Path(suite.summary_json_path).exists())
            self.assertTrue(Path(suite.manifest_path).exists())
            exported = pd.read_csv(suite.summary_csv_path)
            self.assertEqual(list(exported.columns), list(GRAPH_ABLATION_SUMMARY_FIELDS))
            self.assertEqual(len(exported), 8)
            self.assertIn("recon_recall", exported.columns)
            self.assertIn("web_based_recall", exported.columns)
            self.assertTrue(
                all(result.export_result.output_directory.startswith(str(Path(tmp_dir) / "ablation")) for result in suite.run_results)
            )
            self.assertGreater(
                max(record.recon_recall or 0.0 for record in suite.summary_records),
                min(record.recon_recall or 0.0 for record in suite.summary_records),
            )

    def test_cli_parser_exposes_graph_ablation_flags(self) -> None:
        """The CLI should expose the graph ablation sweep entry point."""

        parser = build_parser()
        self.assertTrue(
            any(action.dest == "run_graph_ablation_suite" for action in parser._actions)
        )
        ablation_action = next(
            action
            for action in parser._actions  # type: ignore[attr-defined]
            if action.dest == "graph_ablation_window_sizes"
        )
        self.assertIsInstance(ablation_action, argparse._StoreAction)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
