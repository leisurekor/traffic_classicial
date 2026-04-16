"""Unit tests for the minimal unsupervised graph autoencoder."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("PyTorch is not installed in this environment.") from exc

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import AssociationEdgeConfig, GraphConfig
from traffic_graph.data import LogicalFlowBatch, LogicalFlowRecord, LogicalFlowWindowStats
from traffic_graph.features import fit_feature_preprocessor, transform_graphs
from traffic_graph.models import (
    GraphAutoEncoder,
    ReconstructionLossWeights,
    packed_graph_to_tensors,
    reconstruction_loss,
)
from traffic_graph.graph import build_endpoint_graph


def _graph_config() -> GraphConfig:
    """Return a graph config with both association edge types enabled."""

    return GraphConfig(
        time_window_seconds=60,
        directed=True,
        association_edges=AssociationEdgeConfig(
            enable_same_src_ip=True,
            enable_same_dst_subnet=True,
            dst_subnet_prefix=24,
        ),
    )


def _build_batch(
    *,
    pkt_scale: int = 1,
    byte_scale: int = 1,
    start_offset_seconds: int = 0,
) -> LogicalFlowBatch:
    """Create a reusable logical-flow batch with configurable magnitudes."""

    window_start = datetime(2026, 4, 8, 12, 0, 0) + timedelta(
        seconds=start_offset_seconds
    )
    window_end = window_start + timedelta(seconds=60)
    return LogicalFlowBatch(
        index=start_offset_seconds // 60,
        window_start=window_start,
        window_end=window_end,
        logical_flows=(
            LogicalFlowRecord(
                logical_flow_id=f"logical-a-{start_offset_seconds}",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                dst_port=80,
                protocol="tcp",
                start_time=window_start + timedelta(seconds=5),
                end_time=window_start + timedelta(seconds=10),
                flow_count=1,
                total_pkt_count=11 * pkt_scale,
                total_byte_count=4096 * byte_scale,
                avg_duration=5.0,
                avg_pkt_count=float(11 * pkt_scale),
                avg_byte_count=float(4096 * byte_scale),
                source_flow_ids=("flow-a",),
                src_ports=(11111,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id=f"logical-b-{start_offset_seconds}",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.3",
                dst_port=443,
                protocol="tcp",
                start_time=window_start + timedelta(seconds=15),
                end_time=window_start + timedelta(seconds=21),
                flow_count=1,
                total_pkt_count=9 * pkt_scale,
                total_byte_count=3500 * byte_scale,
                avg_duration=6.0,
                avg_pkt_count=float(9 * pkt_scale),
                avg_byte_count=float(3500 * byte_scale),
                source_flow_ids=("flow-b",),
                src_ports=(22222,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id=f"logical-c-{start_offset_seconds}",
                src_ip="172.16.0.8",
                dst_ip="192.168.1.10",
                dst_port=53,
                protocol="udp",
                start_time=window_start + timedelta(seconds=30),
                end_time=window_start + timedelta(seconds=33),
                flow_count=2,
                total_pkt_count=6 * pkt_scale,
                total_byte_count=480 * byte_scale,
                avg_duration=2.5,
                avg_pkt_count=float(3 * pkt_scale),
                avg_byte_count=float(240 * byte_scale),
                source_flow_ids=("flow-c", "flow-d"),
                src_ports=(33333, 33334),
                directions=("outbound",),
                tcp_flags=(),
                is_aggregated_short_flow=True,
            ),
        ),
        stats=LogicalFlowWindowStats(
            index=start_offset_seconds // 60,
            window_start=window_start,
            window_end=window_end,
            raw_flow_count=4,
            short_flow_count=2,
            long_flow_count=2,
            logical_flow_count=3,
        ),
    )


def _build_packed_graphs():
    """Build two packed graphs for forward and batch tests."""

    graphs = [
        build_endpoint_graph(_build_batch(), graph_config=_graph_config()),
        build_endpoint_graph(
            _build_batch(pkt_scale=2, byte_scale=3, start_offset_seconds=60),
            graph_config=_graph_config(),
        ),
    ]
    preprocessor = fit_feature_preprocessor(graphs)
    return transform_graphs(graphs, preprocessor)


class GraphAutoEncoderTest(unittest.TestCase):
    """Validate forward shapes, batching, and loss backpropagation."""

    def setUp(self) -> None:
        """Create reusable packed graphs and a model for the tests."""

        torch.manual_seed(7)
        self.packed_graphs = _build_packed_graphs()
        self.single_graph = self.packed_graphs[0]
        self.model = GraphAutoEncoder(
            node_input_dim=self.single_graph.node_feature_dim,
            edge_input_dim=self.single_graph.edge_feature_dim,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            dropout=0.0,
            use_edge_features=True,
            reconstruct_edge_features=True,
            loss_weights=ReconstructionLossWeights(node_weight=1.0, edge_weight=0.5),
        )

    def test_forward_shapes_on_single_graph_are_correct(self) -> None:
        """A single packed graph should round-trip through the model shapes."""

        output = self.model(self.single_graph)

        self.assertEqual(
            output.node_embeddings.shape,
            (self.single_graph.node_features.shape[0], 16),
        )
        self.assertEqual(output.graph_embeddings.shape, (1, 16))
        self.assertEqual(
            output.reconstructed_node_features.shape,
            self.single_graph.node_features.shape,
        )
        self.assertEqual(
            output.reconstructed_edge_features.shape,
            self.single_graph.edge_features.shape,
        )
        self.assertIsNone(output.loss_components)

    def test_list_input_is_collated_into_a_batched_forward_pass(self) -> None:
        """A graph list should be collated into a batch with stable shapes."""

        output = self.model(self.packed_graphs)
        packed_batch = packed_graph_to_tensors(self.packed_graphs)

        self.assertEqual(output.graph_embeddings.shape, (2, 16))
        self.assertEqual(output.node_embeddings.shape[0], packed_batch.num_nodes)
        self.assertEqual(output.reconstructed_node_features.shape[0], packed_batch.num_nodes)
        self.assertEqual(output.reconstructed_edge_features.shape[0], packed_batch.num_edges)
        self.assertEqual(packed_batch.graph_ptr.tolist(), [0, 6, 12])

    def test_forward_targets_attach_loss_components(self) -> None:
        """Passing targets into forward should populate named loss components."""

        output = self.model(self.single_graph, targets=self.single_graph)

        self.assertIsNotNone(output.loss_components)
        assert output.loss_components is not None
        self.assertIn("total_loss", output.loss_components)
        self.assertIn("node_loss", output.loss_components)
        self.assertIn("edge_loss", output.loss_components)

    def test_reconstruction_loss_supports_backward(self) -> None:
        """The reconstruction objective should backpropagate through the model."""

        output = self.model(self.single_graph)
        loss_output = reconstruction_loss(
            output,
            weights=ReconstructionLossWeights(node_weight=1.5, edge_weight=0.25),
        )
        loss_output.total_loss.backward()

        first_parameter = next(self.model.parameters())
        self.assertIsNotNone(first_parameter.grad)
        self.assertEqual(first_parameter.grad.shape, first_parameter.shape)

    def test_loss_weighting_remains_linear(self) -> None:
        """Changing the loss weights should scale the total loss predictably."""

        output = self.model(self.single_graph)
        unit_loss = reconstruction_loss(
            output,
            weights=ReconstructionLossWeights(node_weight=1.0, edge_weight=0.0),
        )
        doubled_loss = reconstruction_loss(
            output,
            weights=ReconstructionLossWeights(node_weight=2.0, edge_weight=0.0),
        )

        self.assertAlmostEqual(
            float(doubled_loss.total_loss),
            2.0 * float(unit_loss.total_loss),
            places=6,
        )

    def test_reconstruction_loss_ignores_discrete_feature_columns(self) -> None:
        """Discrete packed feature columns should be masked out of MSE loss terms."""

        output = self.model(self.single_graph)
        mutated_node = output.reconstructed_node_features.detach().clone()
        mutated_edge = output.reconstructed_edge_features.detach().clone()

        node_discrete_indices = torch.nonzero(
            packed_graph_to_tensors(self.single_graph).node_discrete_mask,
            as_tuple=False,
        ).flatten()
        edge_discrete_indices = torch.nonzero(
            packed_graph_to_tensors(self.single_graph).edge_discrete_mask,
            as_tuple=False,
        ).flatten()
        if node_discrete_indices.numel() > 0:
            mutated_node[:, node_discrete_indices] = mutated_node[:, node_discrete_indices] + 10.0
        if edge_discrete_indices.numel() > 0:
            mutated_edge[:, edge_discrete_indices] = mutated_edge[:, edge_discrete_indices] + 10.0

        masked_output = type(output)(
            node_embeddings=output.node_embeddings,
            graph_embeddings=output.graph_embeddings,
            reconstructed_node_features=mutated_node,
            reconstructed_edge_features=mutated_edge,
            tensor_batch=output.tensor_batch,
            loss_components=None,
        )

        baseline_loss = reconstruction_loss(output)
        masked_loss = reconstruction_loss(masked_output)

        self.assertAlmostEqual(
            float(masked_loss.node_loss),
            float(baseline_loss.node_loss),
            places=6,
        )
        self.assertAlmostEqual(
            float(masked_loss.edge_loss),
            float(baseline_loss.edge_loss),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
