"""Small regression tests for the CTU-13 adapter and edge-centric additions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.data import FlowDataset, FlowRecord
from traffic_graph.data.packet_prefix_graph import PacketPrefixNode, encode_packet_prefix_graph
from traffic_graph.datasets import (
    CTU13LabeledFlow,
    align_flow_dataset_to_ctu13_labels,
    load_ctu13_manifest,
)
from traffic_graph.features import fit_feature_preprocessor, transform_graph
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EndpointNode,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.graph.nx_compat import MultiDiGraph

def _load_script_module(name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CTU13PipelineTests(unittest.TestCase):
    def test_ctu13_manifest_parse_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "ctu13_manifest.json"
            manifest_path.write_text(
                '[{"scenario_id":"52","scenario_name":"CTU-13 Scenario 52","scenario_url":"https://example/sc52/","pcap_path":"data/ctu13/raw/sc52/capture.pcap","label_file_path":"data/ctu13/raw/sc52/capture.binetflow","download_status":"downloaded"}]',
                encoding="utf-8",
            )
            entries = load_ctu13_manifest(manifest_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].scenario_id, "52")
            self.assertEqual(entries[0].download_status, "downloaded")

    def test_ctu13_label_alignment_smoke(self) -> None:
        start = datetime(2011, 8, 18, 10, 0, 0, tzinfo=timezone.utc)
        dataset = FlowDataset.from_mappings(
            [
                {
                    "flow_id": "flow-1",
                    "src_ip": "10.0.0.1",
                    "src_port": 12345,
                    "dst_ip": "8.8.8.8",
                    "dst_port": 53,
                    "protocol": "udp",
                    "start_time": start.isoformat(),
                    "end_time": (start + timedelta(seconds=1)).isoformat(),
                    "packet_count": 4,
                    "byte_count": 240,
                }
            ]
        )
        labeled = [
            CTU13LabeledFlow(
                scenario_id="52",
                start_time=start,
                end_time=start + timedelta(seconds=1),
                protocol="udp",
                src_ip="10.0.0.1",
                src_port=12345,
                dst_ip="8.8.8.8",
                dst_port=53,
                label_text="flow=From-Normal-UDP",
                binary_label="benign",
                raw_row={},
            )
        ]
        aligned_rows, summary = align_flow_dataset_to_ctu13_labels(
            dataset,
            labeled,
            scenario_id="52",
        )
        self.assertEqual(len(aligned_rows), 1)
        self.assertEqual(aligned_rows[0].aligned_label, "benign")
        self.assertEqual(summary.benign_count, 1)
        self.assertEqual(summary.unaligned_count, 0)

    def test_packet_prefix_graph_builder_smoke(self) -> None:
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        graph = encode_packet_prefix_graph(
            [
                PacketPrefixNode(0, base_time, 60, 1, 0.0, ("SYN",), 8192, 64, 1, None, "tcp"),
                PacketPrefixNode(1, base_time + timedelta(milliseconds=1), 60, 2, 0.001, ("SYN", "ACK"), 8192, 64, 1, 2, "tcp"),
                PacketPrefixNode(2, base_time + timedelta(milliseconds=2), 52, 1, 0.001, ("ACK",), 8192, 64, 2, 2, "tcp"),
            ]
        )
        self.assertEqual(len(graph.embedding), 16)
        self.assertGreaterEqual(graph.sequential_edge_count, 2)
        self.assertGreaterEqual(graph.acknowledgment_edge_count, 1)
        self.assertGreater(graph.prefix_behavior_signature, 0)

    def test_download_script_smoke_with_mocks(self) -> None:
        module = _load_script_module("download_ctu13", "scripts/download_ctu13.py")
        entry = module.CTU13ScenarioManifestEntry(
            scenario_id="52",
            scenario_name="CTU-13 Scenario 52",
            scenario_url="https://example/sc52/",
            pcap_source_url="https://example/sc52/capture.truncated.pcap.bz2",
            label_source_url="https://example/sc52/capture.binetflow.2format",
            readme_source_url="https://example/sc52/README.md",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(module, "RAW_ROOT", Path(temp_dir) / "raw"), \
                mock.patch.object(module, "_download_with_resume", return_value="downloaded"), \
                mock.patch.object(module, "_decompress_bz2", return_value=None):
                materialized = module._materialize_entry(entry)
        self.assertEqual(materialized.download_status, "downloaded")
        self.assertTrue(materialized.pcap_path.endswith(".pcap"))
        self.assertTrue(materialized.label_file_path.endswith(".2format"))

    def test_categorical_pattern_encoding_and_edge_centric_forward_path(self) -> None:
        node_a = EndpointNode("client:10.0.0.1:1111:tcp", "client", "10.0.0.1", 1111, "tcp")
        node_b = EndpointNode("server:8.8.8.8:443:tcp", "server", "8.8.8.8", 443, "tcp")
        edge = CommunicationEdge(
            edge_id="flow-1",
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type="communication",
            logical_flow_id="flow-1",
            pkt_count=10,
            byte_count=1200,
            duration=1.0,
            flow_count=1,
            is_aggregated=False,
            source_flow_ids=("flow-1",),
            flag_pattern_code=2,
            first_packet_size_pattern=11,
            first_packet_dir_size_pattern=19,
            first_4_packet_pattern_code=1234,
            prefix_behavior_signature=4321,
            flow_length_type="short",
            flow_internal_embedding=tuple(float(index) for index in range(16)),
            flow_internal_packet_count=8,
            flow_internal_sequential_edge_count=7,
            flow_internal_window_edge_count=4,
            flow_internal_ack_edge_count=2,
            flow_internal_opposite_direction_edge_count=3,
        )
        graph_backend = MultiDiGraph()
        graph_backend.add_node(node_a.node_id)
        graph_backend.add_node(node_b.node_id)
        graph_backend.add_edge(node_a.node_id, node_b.node_id, key=edge.edge_id)
        graph_sample = InteractionGraph(
            window_index=0,
            window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            graph=graph_backend,
            nodes=(node_a, node_b),
            edges=(edge,),
            stats=build_interaction_graph_stats((node_a, node_b), (edge,)),
        )
        preprocessor = fit_feature_preprocessor([graph_sample])
        packed_graph = transform_graph(graph_sample, preprocessor)
        self.assertIn("prefix_behavior_signature", packed_graph.edge_feature_fields)
        self.assertIn("flow_length_type_code", packed_graph.edge_feature_fields)
        self.assertIn(True, packed_graph.edge_discrete_mask)

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable in the system python test environment.")

        from traffic_graph.models import GraphAutoEncoder
        from traffic_graph.models.model_types import GraphAutoEncoderConfig

        model = GraphAutoEncoder(
            node_input_dim=packed_graph.node_feature_dim,
            edge_input_dim=packed_graph.edge_feature_dim,
            config=GraphAutoEncoderConfig(
                hidden_dim=16,
                latent_dim=8,
                num_layers=2,
                use_edge_features=True,
                reconstruct_edge_features=True,
                use_temporal_edge_projector=True,
                use_edge_categorical_embeddings=True,
                temporal_edge_field_names=("flow_internal_emb_0", "flow_internal_emb_1"),
            ),
        )
        output = model(packed_graph)
        self.assertEqual(output.reconstructed_edge_features.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
