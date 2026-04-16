"""Smoke tests for the real PCAP -> graph smoke experiment path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser
from traffic_graph.config import AssociationEdgeConfig, GraphConfig
from traffic_graph.data import (
    ShortFlowThresholds,
    load_pcap_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.features import fit_feature_preprocessor, transform_graphs
from traffic_graph.graph import FlowInteractionGraphBuilder
from traffic_graph.pipeline.pcap_graph_smoke import (
    PcapGraphSmokeConfig,
    run_pcap_graph_smoke_experiment,
)
from traffic_graph.pipeline.replay_io import load_export_bundle, summarize_replay_bundle
from traffic_graph.pipeline.replay_io import list_available_tables


class PcapGraphSmokeTest(unittest.TestCase):
    """Validate the real-PCAP smoke path and its exported artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate the local Recon-HostDiscovery capture or skip the suite."""

        cls.pcap_path = (
            PROJECT_ROOT / "artifacts" / "cic_iot2023" / "Recon-HostDiscovery.pcap"
        )
        if not cls.pcap_path.exists():
            raise unittest.SkipTest(
                "Recon-HostDiscovery.pcap is not available in artifacts/cic_iot2023."
            )

    def _load_sample(self, packet_limit: int = 1000):
        """Load a small real-PCAP flow dataset for repeated assertions."""

        return load_pcap_flow_dataset(
            self.pcap_path,
            max_packets=packet_limit,
            idle_timeout_seconds=60.0,
        )

    def test_real_pcap_parses_into_bidirectional_flow_records(self) -> None:
        """The parser should build normalized flow records from a real PCAP."""

        result = self._load_sample(packet_limit=1000)
        self.assertEqual(result.summary.total_packets, 1000)
        self.assertGreater(result.summary.parsed_packets, 0)
        self.assertGreater(result.summary.total_flows, 0)
        self.assertEqual(
            result.summary.total_flows,
            result.summary.flow_dataset_summary.flow_count,
        )

        first_record = result.dataset.records[0]
        self.assertGreater(first_record.packet_count, 0)
        self.assertEqual(
            first_record.packet_count,
            first_record.fwd_pkt_count + first_record.bwd_pkt_count,
        )
        self.assertEqual(
            first_record.byte_count,
            first_record.fwd_bytes + first_record.bwd_bytes,
        )
        self.assertGreaterEqual(first_record.duration, 0.0)
        self.assertEqual(len(first_record.pkt_len_seq), first_record.packet_count)
        self.assertEqual(len(first_record.iat_seq), max(0, first_record.packet_count - 1))
        self.assertIn(first_record.protocol, {"tcp", "udp"})

    def test_real_pcap_graph_knobs_change_windows_edges_and_feature_schema(self) -> None:
        """Window size, association edges, and structural features should matter."""

        result = self._load_sample(packet_limit=1000)
        thresholds = ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024)
        windows_30 = preprocess_flow_dataset(result.dataset, window_size=30, rules=thresholds)
        windows_60 = preprocess_flow_dataset(result.dataset, window_size=60, rules=thresholds)
        self.assertGreater(len(windows_30), len(windows_60))

        base_builder = FlowInteractionGraphBuilder(
            GraphConfig(
                time_window_seconds=60,
                directed=True,
                association_edges=AssociationEdgeConfig(
                    enable_same_src_ip=False,
                    enable_same_dst_subnet=False,
                    dst_subnet_prefix=24,
                ),
            )
        )
        assoc_builder = FlowInteractionGraphBuilder(
            GraphConfig(
                time_window_seconds=60,
                directed=True,
                association_edges=AssociationEdgeConfig(
                    enable_same_src_ip=True,
                    enable_same_dst_subnet=True,
                    dst_subnet_prefix=24,
                ),
            )
        )
        base_graphs = base_builder.build_many(windows_60)
        assoc_graphs = assoc_builder.build_many(windows_60)
        self.assertGreater(
            sum(graph.stats.association_edge_count for graph in assoc_graphs),
            sum(graph.stats.association_edge_count for graph in base_graphs),
        )
        self.assertTrue(
            any(graph.stats.association_edge_count > 0 for graph in assoc_graphs)
        )

        packed_with_structure = transform_graphs(
            assoc_graphs,
            fit_feature_preprocessor(
                assoc_graphs,
                include_graph_structural_features=True,
            ),
            include_graph_structural_features=True,
        )
        packed_without_structure = transform_graphs(
            assoc_graphs,
            fit_feature_preprocessor(
                assoc_graphs,
                include_graph_structural_features=False,
            ),
            include_graph_structural_features=False,
        )
        self.assertGreater(
            packed_with_structure[0].node_feature_dim,
            packed_without_structure[0].node_feature_dim,
        )
        self.assertNotEqual(
            packed_with_structure[0].node_feature_fields,
            packed_without_structure[0].node_feature_fields,
        )

    def test_pcap_smoke_experiment_exports_manifest_and_score_tables(self) -> None:
        """A smoke run should export a manifest-managed bundle and score tables."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_smoke_experiment(
                self.pcap_path,
                temp_dir,
                config=PcapGraphSmokeConfig(
                    packet_limit=1000,
                    smoke_graph_limit=4,
                    epochs=1,
                    batch_size=2,
                    window_size=60,
                    use_association_edges=True,
                    use_graph_structural_features=True,
                ),
            )

            self.assertIn(result.backend, {"gae", "fallback"})
            self.assertIsNotNone(result.export_result)
            self.assertGreater(len(result.graph_summaries), 0)
            self.assertGreater(result.alert_summary.get("total_count", 0), 0)
            export_result = result.export_result
            assert export_result is not None
            manifest_path = Path(export_result.manifest_path)
            self.assertTrue(manifest_path.exists())
            self.assertIn("pcap_config_json", export_result.artifact_paths)
            self.assertIn("pcap_smoke_summary_json", export_result.artifact_paths)
            self.assertIn("graph_scores_jsonl", export_result.artifact_paths)
            self.assertIn("node_scores_jsonl", export_result.artifact_paths)
            self.assertIn("edge_scores_jsonl", export_result.artifact_paths)
            self.assertIn("flow_scores_jsonl", export_result.artifact_paths)

            bundle = load_export_bundle(manifest_path)
            self.assertEqual(bundle.run_id, result.run_id)
            self.assertIn("graph", list_available_tables(bundle))
            self.assertIn("node", list_available_tables(bundle))
            self.assertIn("edge", list_available_tables(bundle))
            self.assertIn("flow", list_available_tables(bundle))
            self.assertIn("smoke", summarize_replay_bundle(bundle))

    def test_cli_parser_exposes_pcap_smoke_flags(self) -> None:
        """The CLI parser should expose the smoke-run PCAP entrypoint."""

        parser = build_parser()
        action_names = {action.dest for action in parser._actions}  # type: ignore[attr-defined]
        self.assertIn("run_pcap_graph_smoke_experiment", action_names)
        self.assertIn("pcap_input", action_names)
        self.assertIn("pcap_window_size", action_names)

    def test_cli_can_run_real_pcap_smoke_experiment(self) -> None:
        """The CLI should execute the real-PCAP smoke path end to end."""

        from traffic_graph.cli import main

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pcap-smoke"
            exit_code = main(
                [
                    "--run-pcap-graph-smoke-experiment",
                    "--pcap-input",
                    str(self.pcap_path),
                    "--pcap-output-dir",
                    str(output_dir),
                    "--pcap-packet-limit",
                    "200",
                    "--pcap-window-size",
                    "60",
                    "--pcap-smoke-graph-limit",
                    "2",
                    "--pcap-train-epochs",
                    "1",
                    "--pcap-batch-size",
                    "1",
                    "--pcap-learning-rate",
                    "0.001",
                    "--pcap-threshold-percentile",
                    "90",
                    "--pcap-random-seed",
                    "13",
                ]
            )
            self.assertEqual(exit_code, 0)
            manifest_paths = list(output_dir.rglob("manifest.json"))
            self.assertTrue(manifest_paths)
            config_paths = list(output_dir.rglob("pcap_config.json"))
            smoke_summary_paths = list(output_dir.rglob("pcap_smoke_summary.json"))
            self.assertTrue(config_paths)
            self.assertTrue(smoke_summary_paths)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
