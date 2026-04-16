"""Tests for anomaly scoring and evaluation metric helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.graph.graph_types import (  # noqa: E402
    CommunicationEdge,
    EndpointNode,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.pipeline.scoring import (  # noqa: E402
    build_edge_score_rows,
    build_flow_score_rows,
    build_graph_score_row,
    build_node_score_rows,
    compute_edge_anomaly_scores,
    compute_graph_anomaly_scores,
    compute_node_anomaly_scores,
)
from traffic_graph.pipeline.metrics import evaluate_scores  # noqa: E402
from traffic_graph.pipeline.runner import PipelineReport, PipelineStage  # noqa: E402


class ScoringAndMetricsTest(unittest.TestCase):
    """Validate score construction, reduction, and evaluation metrics."""

    def _build_graph(self) -> InteractionGraph:
        """Create a small interaction graph with communication and association edges."""

        nodes = (
            EndpointNode(
                node_id="client:10.0.0.1:1234:tcp",
                endpoint_type="client",
                ip="10.0.0.1",
                port=1234,
                proto="tcp",
            ),
            EndpointNode(
                node_id="server:10.0.0.2:80:tcp",
                endpoint_type="server",
                ip="10.0.0.2",
                port=80,
                proto="tcp",
            ),
            EndpointNode(
                node_id="client:10.0.0.3:2222:tcp",
                endpoint_type="client",
                ip="10.0.0.3",
                port=2222,
                proto="tcp",
            ),
        )
        edges = (
            CommunicationEdge(
                edge_id="flow-1",
                source_node_id=nodes[0].node_id,
                target_node_id=nodes[1].node_id,
                edge_type="communication",
                logical_flow_id="flow-1",
                pkt_count=10,
                byte_count=1000,
                duration=1.0,
                flow_count=1,
                is_aggregated=False,
                source_flow_ids=("flow-1",),
            ),
            CommunicationEdge(
                edge_id="assoc-1",
                source_node_id=nodes[0].node_id,
                target_node_id=nodes[2].node_id,
                edge_type="association_same_src_ip",
                logical_flow_id=None,
                pkt_count=0,
                byte_count=0,
                duration=0.0,
                flow_count=0,
                is_aggregated=False,
                source_flow_ids=(),
            ),
            CommunicationEdge(
                edge_id="flow-2",
                source_node_id=nodes[2].node_id,
                target_node_id=nodes[1].node_id,
                edge_type="communication",
                logical_flow_id="flow-2",
                pkt_count=8,
                byte_count=800,
                duration=2.0,
                flow_count=2,
                is_aggregated=True,
                source_flow_ids=("flow-2a", "flow-2b"),
            ),
        )
        stats = build_interaction_graph_stats(nodes, edges)
        return InteractionGraph(
            window_index=0,
            window_start=datetime(2026, 4, 8, 9, 0, 0),
            window_end=datetime(2026, 4, 8, 9, 1, 0),
            graph=None,
            nodes=nodes,
            edges=edges,
            stats=stats,
        )

    def test_node_edge_and_flow_score_rows_are_stable(self) -> None:
        """Score row helpers should preserve graph ordering and flow filtering."""

        graph = self._build_graph()
        node_scores = np.asarray([0.5, 1.5, 2.0], dtype=float)
        edge_scores = np.asarray([0.2, 0.7, 1.2], dtype=float)

        node_rows = build_node_score_rows(0, graph, node_scores)
        edge_rows = build_edge_score_rows(0, graph, edge_scores)
        flow_rows = build_flow_score_rows(0, graph, edge_scores)
        graph_row = build_graph_score_row(0, graph, 1.25)

        self.assertEqual(node_rows[0]["node_id"], "client:10.0.0.1:1234:tcp")
        self.assertEqual(node_rows[1]["endpoint_type"], "server")
        self.assertEqual(edge_rows[1]["edge_type"], "association_same_src_ip")
        self.assertEqual(len(flow_rows), 2)
        self.assertEqual(flow_rows[0]["logical_flow_id"], "flow-1")
        self.assertEqual(flow_rows[1]["flow_anomaly_score"], 1.2)
        self.assertEqual(graph_row["graph_anomaly_score"], 1.25)

    def test_score_computation_and_graph_reduction_are_correct(self) -> None:
        """Row-wise MSE and graph pooling should be deterministic."""

        node_features = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=float)
        reconstructed_nodes = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        node_scores = compute_node_anomaly_scores(node_features, reconstructed_nodes)
        edge_scores = compute_edge_anomaly_scores(
            np.asarray([[1.0, 0.0], [2.0, 2.0]], dtype=float),
            None,
        )

        self.assertTrue(np.allclose(node_scores, np.asarray([1.0, 0.5], dtype=float)))
        self.assertTrue(np.allclose(edge_scores, np.zeros(2, dtype=float)))
        self.assertAlmostEqual(
            float(compute_graph_anomaly_scores(node_scores, reduction="mean")),
            0.75,
        )
        graph_scores = compute_graph_anomaly_scores(
            np.asarray([0.5, 1.5, 2.0], dtype=float),
            graph_ptr=[0, 2, 3],
            reduction="max",
        )
        self.assertTrue(np.allclose(graph_scores, np.asarray([1.5, 2.0], dtype=float)))

    def test_discrete_feature_masks_are_ignored_in_rowwise_scoring(self) -> None:
        """Categorical columns should not inflate MSE-based anomaly scores."""

        edge_features = np.asarray(
            [
                [9.0, 1.0],
                [5.0, 2.0],
            ],
            dtype=float,
        )
        reconstructed = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        unmasked_scores = compute_edge_anomaly_scores(edge_features, reconstructed)
        masked_scores = compute_edge_anomaly_scores(
            edge_features,
            reconstructed,
            discrete_mask=[True, False],
        )

        self.assertTrue(np.allclose(unmasked_scores, np.asarray([41.0, 13.0], dtype=float)))
        self.assertTrue(np.allclose(masked_scores, np.asarray([1.0, 1.0], dtype=float)))

    def test_metrics_and_report_render_are_stable(self) -> None:
        """Metric summaries and report rendering should keep a stable shape."""

        metrics = evaluate_scores(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            threshold=0.5,
        )
        report = PipelineReport(
            run_name="demo",
            input_path="data/flows.csv",
            output_directory="artifacts",
            stages=[PipelineStage(name="eval", status="completed", detail="ok")],
            dry_run=False,
            evaluation_metrics={"graph": metrics.to_dict()},
            evaluation_artifacts={"metrics_json": "artifacts/evaluation/metrics.json"},
            evaluation_label_field="label",
            evaluation_score_reduction="mean",
            evaluation_threshold=0.5,
        )
        rendered = report.render()

        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.pr_auc, 1.0)
        self.assertIn("Evaluation metrics:", rendered)
        self.assertIn("metrics_json", rendered)
        self.assertIn("Evaluation label field: label", rendered)


if __name__ == "__main__":
    unittest.main()
