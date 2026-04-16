"""Smoke tests for the reproducible real-PCAP experiment runner."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from dataclasses import replace
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import GraphConfig
from traffic_graph.data import ShortFlowThresholds, load_pcap_flow_dataset, preprocess_flow_dataset
from traffic_graph.features import (
    build_base_feature_views,
    fit_feature_preprocessor,
    transform_graphs,
)
from traffic_graph.graph import FlowInteractionGraphBuilder
from traffic_graph.cli import build_parser, main
from traffic_graph.pipeline.pcap_graph_experiment import (
    PcapGraphExperimentConfig,
    run_pcap_graph_experiment,
)
from traffic_graph.pipeline.pcap_graph_smoke import _apply_graph_score_reduction_to_rows


class PcapGraphExperimentTests(unittest.TestCase):
    """Validate the reproducible mini experiment entrypoint for real PCAPs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate the verified PCAP fixture or skip the suite."""

        cls.pcap_path = (
            PROJECT_ROOT / "artifacts" / "cic_iot2023" / "Recon-HostDiscovery.pcap"
        )
        if not cls.pcap_path.exists():
            raise unittest.SkipTest(
                "Recon-HostDiscovery.pcap is not available in artifacts/cic_iot2023."
            )

    def _config(
        self,
        *,
        packet_limit: int = 1000,
        window_size: int = 30,
        graph_score_reduction: str = "hybrid_max_rank_flow_node_max",
    ) -> PcapGraphExperimentConfig:
        """Build a tiny deterministic config for fast experiment tests."""

        return PcapGraphExperimentConfig(
            packet_limit=packet_limit,
            window_size=window_size,
            smoke_graph_limit=2,
            benign_train_ratio=0.5,
            graph_score_reduction=graph_score_reduction,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            threshold_percentile=90.0,
            random_seed=13,
        )

    def _link_fixture(self, directory: Path, name: str) -> Path:
        """Create a cheap alias path for the shared PCAP fixture."""

        alias_path = directory / name
        if alias_path.exists():
            alias_path.unlink()
        try:
            alias_path.symlink_to(self.pcap_path.resolve())
        except OSError:
            alias_path.hardlink_to(self.pcap_path.resolve())
        return alias_path

    def test_config_to_dict_keeps_required_fields(self) -> None:
        """The experiment config should serialize the reproducibility knobs."""

        config = self._config()
        payload = config.to_dict()
        self.assertEqual(payload["packet_limit"], 1000)
        self.assertEqual(payload["packet_sampling_mode"], "random_window")
        self.assertEqual(payload["window_size"], 30)
        self.assertEqual(payload["smoke_graph_limit"], 2)
        self.assertEqual(payload["benign_train_ratio"], 0.5)
        self.assertEqual(payload["graph_score_reduction"], "hybrid_max_rank_flow_node_max")
        self.assertTrue(payload["use_association_edges"])
        self.assertTrue(payload["use_graph_structural_features"])

    def test_random_window_sampling_exports_packet_start_offsets(self) -> None:
        """Random-window mode should export reproducible packet start offsets."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                malicious_inputs=[self.pcap_path],
                config=replace(
                    self._config(packet_limit=200, window_size=60),
                    packet_sampling_mode="random_window",
                ),
            )

            source_summaries = result.summary.get("source_summaries", [])
            self.assertTrue(source_summaries)
            parse_summary = source_summaries[0]["parse_summary"]
            self.assertIn("packet_start_offset", parse_summary)
            self.assertGreaterEqual(int(parse_summary["packet_start_offset"]), 0)

    def test_flow_level_probe_features_reach_edge_feature_rows(self) -> None:
        """New flow-level probe fields should survive into communication edges and edge features."""

        load_result = load_pcap_flow_dataset(
            self.pcap_path,
            max_packets=400,
            idle_timeout_seconds=60.0,
        )
        self.assertTrue(load_result.dataset.records)
        first_record = load_result.dataset.records[0]
        self.assertEqual(len(first_record.iat_hist), 6)
        self.assertEqual(len(first_record.pkt_len_hist), 6)
        self.assertGreaterEqual(first_record.retry_like_count, 0)
        self.assertGreaterEqual(first_record.retry_like_ratio, 0.0)
        self.assertGreaterEqual(first_record.coarse_ack_delay_mean, 0.0)
        self.assertGreaterEqual(first_record.seq_ack_match_ratio, 0.0)
        self.assertGreaterEqual(first_record.retry_burst_count, 0)
        self.assertGreaterEqual(first_record.small_pkt_burst_count, 0)

        batches = preprocess_flow_dataset(
            load_result.dataset,
            window_size=60,
            rules=ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024),
        )
        graph_sample = FlowInteractionGraphBuilder(GraphConfig(time_window_seconds=60)).build(
            batches[0]
        )
        communication_edges = [
            edge for edge in graph_sample.edges if edge.edge_type == "communication"
        ]
        self.assertTrue(communication_edges)
        first_edge = communication_edges[0]
        self.assertEqual(len(first_edge.iat_hist), 6)
        self.assertEqual(len(first_edge.pkt_len_hist), 6)
        self.assertGreaterEqual(first_edge.coarse_ack_delay_mean, 0.0)
        self.assertGreaterEqual(first_edge.seq_ack_match_ratio, 0.0)
        self.assertGreaterEqual(first_edge.retry_burst_count, 0)
        self.assertGreaterEqual(first_edge.small_pkt_burst_count, 0)

        feature_view = build_base_feature_views(graph_sample)
        self.assertTrue(feature_view.edge_features.feature_rows)
        first_edge_row = feature_view.edge_features.feature_rows[0]
        self.assertIn("retry_like_count", first_edge_row)
        self.assertIn("retry_like_ratio", first_edge_row)
        self.assertIn("flag_syn_ratio", first_edge_row)
        self.assertIn("flag_pattern_code", first_edge_row)
        self.assertIn("first_packet_size_pattern", first_edge_row)
        self.assertIn("coarse_ack_delay_mean", first_edge_row)
        self.assertIn("coarse_ack_delay_p75", first_edge_row)
        self.assertIn("ack_delay_large_gap_ratio", first_edge_row)
        self.assertIn("seq_ack_match_ratio", first_edge_row)
        self.assertIn("unmatched_seq_ratio", first_edge_row)
        self.assertIn("unmatched_ack_ratio", first_edge_row)
        self.assertIn("retry_burst_count", first_edge_row)
        self.assertIn("retry_burst_max_len", first_edge_row)
        self.assertIn("retry_like_dense_ratio", first_edge_row)
        self.assertIn("first_packet_dir_size_pattern", first_edge_row)
        self.assertIn("first_4_packet_pattern_code", first_edge_row)
        self.assertIn("small_pkt_burst_count", first_edge_row)
        self.assertIn("small_pkt_burst_ratio", first_edge_row)
        self.assertIn("rst_after_small_burst_indicator", first_edge_row)
        self.assertIn("iat_hist_bin_5", first_edge_row)
        self.assertIn("pkt_len_hist_bin_5", first_edge_row)

        preprocessor = fit_feature_preprocessor(
            [graph_sample],
            include_graph_structural_features=True,
        )
        packed_graph = transform_graphs(
            [graph_sample],
            preprocessor,
            include_graph_structural_features=True,
        )[0]
        edge_field_names = packed_graph.edge_feature_fields
        edge_discrete_mask = packed_graph.edge_discrete_mask
        self.assertTrue(
            edge_discrete_mask[edge_field_names.index("flag_pattern_code")]
        )
        self.assertTrue(
            edge_discrete_mask[edge_field_names.index("first_packet_size_pattern")]
        )
        self.assertTrue(
            edge_discrete_mask[edge_field_names.index("first_packet_dir_size_pattern")]
        )
        self.assertTrue(
            edge_discrete_mask[edge_field_names.index("first_4_packet_pattern_code")]
        )

    def test_temporal_edge_projector_forward_path_runs(self) -> None:
        """The optional temporal edge branch should run without changing default tests."""

        try:
            from traffic_graph.models import GraphAutoEncoder, GraphAutoEncoderConfig
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("torch is unavailable in the system python test environment.")
            raise

        load_result = load_pcap_flow_dataset(
            self.pcap_path,
            max_packets=400,
            idle_timeout_seconds=60.0,
        )
        batches = preprocess_flow_dataset(
            load_result.dataset,
            window_size=60,
            rules=ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024),
        )
        graphs = FlowInteractionGraphBuilder(GraphConfig(time_window_seconds=60)).build_many(
            batches[:2]
        )
        self.assertTrue(graphs)
        preprocessor = fit_feature_preprocessor(
            graphs,
            include_graph_structural_features=True,
        )
        packed_graphs = transform_graphs(
            graphs,
            preprocessor,
            include_graph_structural_features=True,
        )
        packed_graph = packed_graphs[0]
        self.assertIn("first_4_packet_pattern_code", packed_graph.edge_feature_fields)

        model = GraphAutoEncoder(
            node_input_dim=packed_graph.node_feature_dim,
            edge_input_dim=packed_graph.edge_feature_dim,
            config=GraphAutoEncoderConfig(
                hidden_dim=16,
                latent_dim=8,
                num_layers=2,
                dropout=0.0,
                use_edge_features=True,
                reconstruct_edge_features=True,
                use_temporal_edge_projector=True,
                temporal_edge_hidden_dim=8,
            ),
        )
        output = model(packed_graph)
        self.assertEqual(
            tuple(output.reconstructed_node_features.shape),
            tuple(packed_graph.node_features.shape),
        )
        self.assertIsNotNone(output.reconstructed_edge_features)
        self.assertEqual(
            tuple(output.reconstructed_edge_features.shape),
            tuple(packed_graph.edge_features.shape),
        )

    def test_smoke_mode_runs_and_exports_root_artifacts(self) -> None:
        """Malicious-only input should run in smoke mode and export stable files."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                malicious_inputs=[self.pcap_path],
                config=self._config(packet_limit=300, window_size=60),
            )

            self.assertEqual(result.mode, "smoke")
            self.assertGreater(int(result.summary["malicious_graph_count"]), 0)
            self.assertEqual(int(result.summary["benign_graph_count"]), 0)
            manifest_path = Path(result.export_result.manifest_path)
            self.assertTrue(manifest_path.exists())
            self.assertTrue(
                Path(result.export_result.artifact_paths["pcap_experiment_config_json"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["pcap_experiment_summary_json"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["graph_summary_csv"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["overall_scores_csv"]).exists()
            )
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["experiment_mode"], "smoke")
            self.assertIn("overall_scores_csv", manifest_payload["artifact_paths"])
            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                comparison_summary["overall_metrics_availability"],
                "unavailable_smoke_mode",
            )
            source_score_summary = json.loads(
                Path(result.export_result.artifact_paths["source_score_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(source_score_summary), 1)
            self.assertGreaterEqual(float(source_score_summary[0]["score_max"]), 0.0)
            self.assertGreaterEqual(float(source_score_summary[0]["score_p90"]), 0.0)
            split_score_summary = json.loads(
                Path(result.export_result.artifact_paths["split_score_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            smoke_rows = [
                row for row in split_score_summary if row.get("split_name") == "smoke_evaluation"
            ]
            self.assertEqual(len(smoke_rows), 1)
            self.assertGreaterEqual(float(smoke_rows[0]["score_max"]), 0.0)
            self.assertGreaterEqual(float(smoke_rows[0]["score_p90"]), 0.0)

    def test_binary_mode_runs_with_real_fixture_and_exports_metrics(self) -> None:
        """Benign and malicious inputs should produce a binary mini experiment."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(packet_limit=1000, window_size=30),
            )

            self.assertEqual(result.mode, "binary_evaluation")
            self.assertGreater(int(result.summary["benign_graph_count"]), 0)
            self.assertGreater(int(result.summary["malicious_graph_count"]), 0)
            overall_metrics = result.summary["overall_metrics"]
            self.assertIsInstance(overall_metrics, dict)
            self.assertIn("roc_auc", overall_metrics)
            self.assertTrue(
                Path(result.export_result.artifact_paths["per_attack_metrics_csv"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["attack_scores_csv"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["metrics_summary_root_json"]).exists()
            )
            metrics_summary = json.loads(
                Path(
                    result.export_result.artifact_paths["metrics_summary_root_json"]
                ).read_text(encoding="utf-8")
            )
            self.assertIn("pcap_experiment", metrics_summary)
            self.assertEqual(
                metrics_summary["pcap_experiment"]["mode"],
                "binary_evaluation",
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["comparison_summary_csv"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["train_graph_scores_csv"]).exists()
            )
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_graph_scores)
            first_train_row = train_graph_scores[0]
            self.assertEqual(
                first_train_row["graph_score_reduction"],
                "hybrid_max_rank_flow_node_max",
            )
            self.assertIn("node_score_mean", first_train_row)
            self.assertIn("edge_score_p90", first_train_row)
            self.assertIn("flow_score_max", first_train_row)
            comparison_summary = json.loads(
                Path(
                    result.export_result.artifact_paths["comparison_summary_json"]
                ).read_text(encoding="utf-8")
            )
            self.assertIn("run_id", comparison_summary)
            self.assertIn("scorer_type", comparison_summary)
            self.assertIn("scorer_role", comparison_summary)
            self.assertIn("benign_train_graph_count", comparison_summary)
            self.assertIn("overall_roc_auc", comparison_summary)
            self.assertEqual(
                comparison_summary["graph_score_reduction"],
                "hybrid_max_rank_flow_node_max",
            )
            self.assertEqual(comparison_summary["scorer_role"], "default_candidate")

    def test_non_default_graph_score_reduction_is_exported(self) -> None:
        """A non-default reduction should flow into config and train-score artifacts."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=600,
                    window_size=30,
                    graph_score_reduction="flow_p90",
                ),
            )

            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(comparison_summary["graph_score_reduction"], "flow_p90")
            self.assertEqual(comparison_summary["scorer_role"], "fallback")
            self.assertTrue(train_graph_scores)
            self.assertTrue(
                all(row["graph_score_reduction"] == "flow_p90" for row in train_graph_scores)
            )
            self.assertTrue(
                all("flow_score_p90" in row for row in train_graph_scores)
            )

    def test_hybrid_graph_score_reduction_is_exported(self) -> None:
        """The thin hybrid reduction should flow through config and train artifacts."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=600,
                    window_size=30,
                    graph_score_reduction="hybrid_max_rank_flow_node_max",
                ),
            )

            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                comparison_summary["graph_score_reduction"],
                "hybrid_max_rank_flow_node_max",
            )
            self.assertEqual(comparison_summary["scorer_role"], "default_candidate")
            self.assertTrue(train_graph_scores)
            self.assertTrue(
                all(
                    row["graph_score_reduction"] == "hybrid_max_rank_flow_node_max"
                    for row in train_graph_scores
                )
            )
            self.assertTrue(all("node_score_max" in row for row in train_graph_scores))
            self.assertTrue(all("flow_score_p90" in row for row in train_graph_scores))
            self.assertTrue(all("flow_iat_proxy_p75" in row for row in train_graph_scores))
            self.assertTrue(
                all("flow_pkt_count_topk_mean" in row for row in train_graph_scores)
            )
            self.assertTrue(
                all("short_flow_score_topk_mean" in row for row in train_graph_scores)
            )
            self.assertTrue(
                all("long_flow_score_p75" in row for row in train_graph_scores)
            )
            self.assertTrue(
                all(
                    "component_max_flow_score_topk_mean" in row
                    for row in train_graph_scores
                )
            )

    def test_flow_p90_train_reference_q95_matches_exported_threshold(self) -> None:
        """Exported train-reference scores should reproduce the stored q95 threshold."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=600,
                    window_size=30,
                    graph_score_reduction="flow_p90",
                ),
            )

            with Path(
                result.export_result.artifact_paths["train_graph_scores_csv"]
            ).open(encoding="utf-8") as handle:
                train_scores = [
                    float(row["score"])
                    for row in csv.DictReader(handle)
                    if row.get("score") not in (None, "")
                ]
            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(train_scores)
            self.assertAlmostEqual(
                float(np.quantile(np.asarray(train_scores, dtype=float), 0.95)),
                float(comparison_summary["threshold"]),
                places=6,
            )

    def test_hybrid_train_reference_q95_matches_exported_threshold(self) -> None:
        """The formal hybrid reduction should keep train q95 and stored threshold aligned."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=600,
                    window_size=30,
                    graph_score_reduction="hybrid_max_rank_flow_node_max",
                ),
            )

            with Path(
                result.export_result.artifact_paths["train_graph_scores_csv"]
            ).open(encoding="utf-8") as handle:
                train_scores = [
                    float(row["score"])
                    for row in csv.DictReader(handle)
                    if row.get("score") not in (None, "")
                ]
            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(train_scores)
            self.assertAlmostEqual(
                float(np.quantile(np.asarray(train_scores, dtype=float), 0.95)),
                float(comparison_summary["threshold"]),
                places=6,
            )

    def test_hybrid_default_reduction_uses_tail_and_temporal_context(self) -> None:
        """The default hybrid reducer should lift moderate tails above the legacy max/rank core."""

        reference_rows = [
            {
                "graph_anomaly_score": 0.1,
                "graph_score_reduction": "flow_p90",
                "flow_score_p90": 1.0,
                "node_score_max": 1.0,
                "flow_score_p75": 1.0,
                "flow_score_topk_mean": 1.0,
                "node_score_p75": 1.0,
                "node_score_topk_mean": 1.0,
                "flow_iat_proxy_p75": 1.0,
                "flow_pkt_rate_p90": 1.0,
                "flow_pkt_count_topk_mean": 1.0,
                "short_flow_score_topk_mean": 1.0,
                "long_flow_score_p75": 1.0,
                "short_flow_ratio": 1.0,
                "component_max_flow_score_topk_mean": 1.0,
                "component_max_node_score_topk_mean": 1.0,
                "server_neighborhood_flow_score_topk_mean": 1.0,
                "server_concentration": 1.0,
                "aggregated_edge_ratio": 1.0,
                "edge_density": 1.0,
            },
            {
                "graph_anomaly_score": 0.2,
                "graph_score_reduction": "flow_p90",
                "flow_score_p90": 2.0,
                "node_score_max": 2.0,
                "flow_score_p75": 2.0,
                "flow_score_topk_mean": 2.0,
                "node_score_p75": 2.0,
                "node_score_topk_mean": 2.0,
                "flow_iat_proxy_p75": 2.0,
                "flow_pkt_rate_p90": 2.0,
                "flow_pkt_count_topk_mean": 2.0,
                "short_flow_score_topk_mean": 2.0,
                "long_flow_score_p75": 2.0,
                "short_flow_ratio": 2.0,
                "component_max_flow_score_topk_mean": 2.0,
                "component_max_node_score_topk_mean": 2.0,
                "server_neighborhood_flow_score_topk_mean": 2.0,
                "server_concentration": 2.0,
                "aggregated_edge_ratio": 2.0,
                "edge_density": 2.0,
            },
            {
                "graph_anomaly_score": 0.3,
                "graph_score_reduction": "flow_p90",
                "flow_score_p90": 3.0,
                "node_score_max": 3.0,
                "flow_score_p75": 3.0,
                "flow_score_topk_mean": 3.0,
                "node_score_p75": 3.0,
                "node_score_topk_mean": 3.0,
                "flow_iat_proxy_p75": 3.0,
                "flow_pkt_rate_p90": 3.0,
                "flow_pkt_count_topk_mean": 3.0,
                "short_flow_score_topk_mean": 3.0,
                "long_flow_score_p75": 3.0,
                "short_flow_ratio": 3.0,
                "component_max_flow_score_topk_mean": 3.0,
                "component_max_node_score_topk_mean": 3.0,
                "server_neighborhood_flow_score_topk_mean": 3.0,
                "server_concentration": 3.0,
                "aggregated_edge_ratio": 3.0,
                "edge_density": 3.0,
            },
        ]
        candidate_row = {
            "graph_anomaly_score": 0.0,
            "flow_score_p90": 1.1,
            "node_score_max": 1.1,
            "flow_score_p75": 2.7,
            "flow_score_topk_mean": 2.9,
            "node_score_p75": 2.6,
            "node_score_topk_mean": 2.8,
            "flow_iat_proxy_p75": 2.7,
            "flow_pkt_rate_p90": 2.6,
            "flow_pkt_count_topk_mean": 2.8,
            "short_flow_score_topk_mean": 2.8,
            "long_flow_score_p75": 2.7,
            "short_flow_ratio": 2.6,
            "component_max_flow_score_topk_mean": 2.9,
            "component_max_node_score_topk_mean": 2.8,
            "server_neighborhood_flow_score_topk_mean": 2.7,
            "server_concentration": 2.0,
            "aggregated_edge_ratio": 2.0,
            "edge_density": 2.0,
        }
        rescored = _apply_graph_score_reduction_to_rows(
            [candidate_row],
            reduction_method="hybrid_max_rank_flow_node_max",
            reference_rows=reference_rows,
        )
        self.assertEqual(len(rescored), 1)
        legacy_reference = 1.0 / 3.0
        self.assertGreater(float(rescored[0]["graph_anomaly_score"]), legacy_reference)
        self.assertEqual(
            rescored[0]["graph_score_reduction"],
            "hybrid_max_rank_flow_node_max",
        )

    def test_paper_inspired_reductions_can_rescore_graph_rows(self) -> None:
        """Paper-inspired graph-score reductions should rescore from stored summaries."""

        reference_rows = [
            {
                "graph_anomaly_score": 0.1,
                "graph_score_reduction": "flow_p90",
                "node_score_mean": 1.0,
                "node_score_p90": 2.0,
                "node_score_max": 3.0,
                "flow_score_p90": 1.0,
                "server_node_count": 10,
                "aggregated_edge_count": 3,
                "communication_edge_count": 6,
                "association_edge_count": 9,
                "edge_count": 12,
                "node_count": 6,
                "edge_density": 2.0,
                "aggregated_edge_ratio": 0.5,
                "association_edge_ratio": 1.5,
            },
            {
                "graph_anomaly_score": 0.2,
                "graph_score_reduction": "flow_p90",
                "node_score_mean": 2.0,
                "node_score_p90": 4.0,
                "node_score_max": 6.0,
                "flow_score_p90": 2.0,
                "server_node_count": 20,
                "aggregated_edge_count": 6,
                "communication_edge_count": 6,
                "association_edge_count": 12,
                "edge_count": 24,
                "node_count": 8,
                "edge_density": 3.0,
                "aggregated_edge_ratio": 1.0,
                "association_edge_ratio": 2.0,
            },
            {
                "graph_anomaly_score": 0.3,
                "graph_score_reduction": "flow_p90",
                "node_score_mean": 3.0,
                "node_score_p90": 6.0,
                "node_score_max": 9.0,
                "flow_score_p90": 3.0,
                "server_node_count": 30,
                "aggregated_edge_count": 9,
                "communication_edge_count": 6,
                "association_edge_count": 18,
                "edge_count": 48,
                "node_count": 12,
                "edge_density": 4.0,
                "aggregated_edge_ratio": 1.5,
                "association_edge_ratio": 3.0,
            },
        ]
        candidate_row = {
            "graph_anomaly_score": 0.0,
            "node_score_mean": 2.5,
            "node_score_p90": 5.0,
            "node_score_max": 7.0,
            "flow_score_p90": 2.5,
            "server_node_count": 25,
            "aggregated_edge_count": 8.0,
            "communication_edge_count": 6.0,
            "association_edge_count": 15.0,
            "edge_count": 30.0,
            "node_count": 10.0,
            "edge_density": 3.0,
            "aggregated_edge_ratio": 8.0 / 6.0,
            "association_edge_ratio": 15.0 / 6.0,
        }
        expected_score = 2.0 / 3.0
        for reduction_name in (
            "decision_topk_flow_node",
            "relation_max_flow_server_count",
            "structural_fig_max",
        ):
            rescored = _apply_graph_score_reduction_to_rows(
                [candidate_row],
                reduction_method=reduction_name,  # type: ignore[arg-type]
                reference_rows=reference_rows,
            )
            self.assertEqual(len(rescored), 1)
            self.assertAlmostEqual(
                float(rescored[0]["graph_anomaly_score"]),
                expected_score,
                places=9,
            )
            self.assertEqual(rescored[0]["graph_score_reduction"], reduction_name)

    def test_paper_inspired_graph_score_reduction_is_exported(self) -> None:
        """Paper-inspired reductions should export structural side-summary columns."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=1000,
                    window_size=30,
                    graph_score_reduction="decision_topk_flow_node",
                ),
            )

            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                comparison_summary["graph_score_reduction"],
                "decision_topk_flow_node",
            )
            self.assertEqual(comparison_summary["scorer_role"], "experimental")
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_graph_scores)
            first_row = train_graph_scores[0]
            self.assertEqual(first_row["graph_score_reduction"], "decision_topk_flow_node")
            self.assertIn("server_node_count", first_row)
            self.assertIn("edge_density", first_row)
            self.assertIn("aggregated_edge_ratio", first_row)

    def test_hybrid_decision_tail_balance_is_exported(self) -> None:
        """The weak-anomaly experimental reducer should flow through exports."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[self.pcap_path],
                malicious_inputs=[self.pcap_path],
                config=self._config(
                    packet_limit=1000,
                    window_size=30,
                    graph_score_reduction="hybrid_decision_tail_balance",
                ),
            )

            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                comparison_summary["graph_score_reduction"],
                "hybrid_decision_tail_balance",
            )
            self.assertEqual(comparison_summary["scorer_role"], "experimental")
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_graph_scores)
            first_row = train_graph_scores[0]
            self.assertEqual(first_row["graph_score_reduction"], "hybrid_decision_tail_balance")
            self.assertIn("node_score_p75", first_row)
            self.assertIn("node_score_topk_mean", first_row)
            self.assertIn("flow_score_p75", first_row)
            self.assertIn("flow_score_topk_mean", first_row)
            self.assertIn("short_flow_score_topk_mean", first_row)
            self.assertIn("long_flow_score_p75", first_row)
            self.assertIn("component_max_flow_score_topk_mean", first_row)

    def test_multiple_benign_inputs_are_recorded_with_split_statistics(self) -> None:
        """Merged benign inputs should keep source lists and split counts in the summary."""

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            benign_a = self._link_fixture(temp_path, "benign-a.pcap")
            benign_b = self._link_fixture(temp_path, "benign-b.pcap")
            malicious = self._link_fixture(temp_path, "malicious-a.pcap")
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[benign_a, benign_b],
                malicious_inputs=[malicious],
                config=self._config(packet_limit=400, window_size=30),
            )

            summary = result.summary
            self.assertEqual(len(summary["benign_inputs"]), 2)
            self.assertEqual(len(summary["malicious_inputs"]), 1)
            split_graph_counts = summary["split_graph_counts"]
            self.assertGreaterEqual(int(split_graph_counts["benign_total"]), 2)
            self.assertGreaterEqual(int(split_graph_counts["train"]), 1)
            self.assertGreaterEqual(int(split_graph_counts["malicious_test"]), 1)
            self.assertIn("train", summary["split_graph_ids"])
            self.assertIn("comparison_summary", summary)
            benign_source_rows = [
                row for row in summary["source_summaries"] if row["source_role"] == "benign"
            ]
            self.assertEqual(len(benign_source_rows), 2)
            for row in benign_source_rows:
                self.assertIn("assignment_counts", row)
                self.assertIn("graph_ids_by_split", row)

    def test_multiple_malicious_inputs_export_source_level_metrics(self) -> None:
        """Multiple malicious inputs should emit source-level metrics and score summaries."""

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            benign = self._link_fixture(temp_path, "benign-only.pcap")
            malicious_a = self._link_fixture(temp_path, "malicious-a.pcap")
            malicious_b = self._link_fixture(temp_path, "malicious-b.pcap")
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                benign_inputs=[benign],
                malicious_inputs=[malicious_a, malicious_b],
                config=self._config(packet_limit=500, window_size=30),
            )

            artifact_paths = result.export_result.artifact_paths
            self.assertTrue(Path(artifact_paths["source_score_summary_csv"]).exists())
            self.assertTrue(Path(artifact_paths["split_score_summary_csv"]).exists())
            self.assertTrue(Path(artifact_paths["malicious_source_metrics_csv"]).exists())

            malicious_source_metrics = json.loads(
                Path(artifact_paths["malicious_source_metrics_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(malicious_source_metrics), 2)
            self.assertEqual(
                {row["source_name"] for row in malicious_source_metrics},
                {"malicious-a", "malicious-b"},
            )
            self.assertEqual(len(result.summary["source_score_summaries"]), 3)
            self.assertEqual(len(result.summary["malicious_source_metrics"]), 2)
            comparison_summary = json.loads(
                Path(artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("worst_malicious_source_id", comparison_summary)
            self.assertIn("worst_malicious_pr_auc", comparison_summary)

    def test_fallback_path_exports_new_analysis_artifacts(self) -> None:
        """The deterministic fallback path should still write the new summary artifacts."""

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            benign = self._link_fixture(temp_path, "benign-fallback.pcap")
            malicious = self._link_fixture(temp_path, "malicious-fallback.pcap")
            with mock.patch(
                "traffic_graph.pipeline.pcap_graph_experiment._has_torch",
                return_value=False,
            ):
                result = run_pcap_graph_experiment(
                    export_dir=temp_dir,
                    benign_inputs=[benign],
                    malicious_inputs=[malicious],
                    config=self._config(packet_limit=300, window_size=60),
                )

            self.assertEqual(result.backend, "fallback")
            self.assertIn("source_score_summary_json", result.export_result.artifact_paths)
            self.assertIn("split_score_summary_json", result.export_result.artifact_paths)
            self.assertIn("malicious_source_metrics_json", result.export_result.artifact_paths)
            self.assertIn("comparison_summary_json", result.export_result.artifact_paths)
            self.assertIn("train_graph_scores_jsonl", result.export_result.artifact_paths)
            self.assertTrue(
                Path(result.export_result.artifact_paths["source_score_summary_json"]).exists()
            )
            self.assertTrue(
                Path(result.export_result.artifact_paths["split_score_summary_json"]).exists()
            )
            train_graph_scores = [
                json.loads(line)
                for line in Path(
                    result.export_result.artifact_paths["train_graph_scores_jsonl"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(train_graph_scores)
            self.assertEqual(
                {row["split_name"] for row in train_graph_scores},
                {"benign_train_reference"},
            )
            self.assertTrue(all("source_id" in row for row in train_graph_scores))

    def test_cli_parser_exposes_reproducible_pcap_experiment_flags(self) -> None:
        """The CLI should expose the reproducible PCAP experiment switches."""

        parser = build_parser()
        action_names = {action.dest for action in parser._actions}  # type: ignore[attr-defined]
        self.assertIn("run_pcap_graph_experiment", action_names)
        self.assertIn("pcap_benign_input", action_names)
        self.assertIn("pcap_malicious_input", action_names)
        self.assertIn("pcap_benign_train_ratio", action_names)
        self.assertIn("pcap_experiment_label", action_names)
        self.assertIn("pcap_graph_score_reduction", action_names)
        graph_reduction_action = next(
            action for action in parser._actions if action.dest == "pcap_graph_score_reduction"  # type: ignore[attr-defined]
        )
        self.assertIn("hybrid_max_rank_flow_node_max", graph_reduction_action.choices)
        self.assertIn("hybrid_decision_tail_balance", graph_reduction_action.choices)
        self.assertIn("decision_topk_flow_node", graph_reduction_action.choices)
        self.assertIn("relation_max_flow_server_count", graph_reduction_action.choices)
        self.assertIn("structural_fig_max", graph_reduction_action.choices)
        self.assertEqual(graph_reduction_action.default, "hybrid_max_rank_flow_node_max")

    def test_experiments_doc_mentions_hybrid_default_and_flow_p90_fallback(self) -> None:
        """The experiment notes should describe the current default and rollback scorer."""

        docs_text = (PROJECT_ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
        self.assertIn("hybrid_max_rank_flow_node_max", docs_text)
        self.assertIn("flow_p90", docs_text)
        self.assertIn("current graph-mode default candidate", docs_text)

    def test_experiment_label_is_written_to_summary_and_manifest(self) -> None:
        """Optional experiment labels should flow into root summaries and manifest metadata."""

        with TemporaryDirectory() as temp_dir:
            result = run_pcap_graph_experiment(
                export_dir=temp_dir,
                malicious_inputs=[self.pcap_path],
                experiment_label="pcap-label-check",
                config=self._config(packet_limit=200, window_size=60),
            )

            self.assertEqual(result.summary["experiment_label"], "pcap-label-check")
            manifest_payload = json.loads(
                Path(result.export_result.manifest_path).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_payload["experiment_label"], "pcap-label-check")
            comparison_summary = json.loads(
                Path(result.export_result.artifact_paths["comparison_summary_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(comparison_summary["experiment_label"], "pcap-label-check")

    def test_cli_can_run_pcap_graph_experiment(self) -> None:
        """The CLI should execute the reproducible PCAP experiment end to end."""

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pcap-experiment"
            exit_code = main(
                [
                    "--run-pcap-graph-experiment",
                    "--pcap-malicious-input",
                    str(self.pcap_path),
                    "--pcap-experiment-label",
                    "cli-smoke-label",
                    "--pcap-output-dir",
                    str(output_dir),
                    "--pcap-packet-limit",
                    "200",
                    "--pcap-smoke-graph-limit",
                    "2",
                    "--pcap-train-epochs",
                    "1",
                    "--pcap-batch-size",
                    "1",
                    "--pcap-window-size",
                    "60",
                    "--pcap-threshold-percentile",
                    "90",
                    "--pcap-graph-score-reduction",
                    "flow_p90",
                ]
            )
            self.assertEqual(exit_code, 0)
            manifest_paths = list(output_dir.rglob("manifest.json"))
            self.assertTrue(manifest_paths)
            config_paths = list(output_dir.rglob("pcap_experiment_config.json"))
            summary_paths = list(output_dir.rglob("pcap_experiment_summary.json"))
            comparison_paths = list(output_dir.rglob("comparison_summary.json"))
            train_score_paths = list(output_dir.rglob("train_graph_scores.csv"))
            self.assertTrue(config_paths)
            self.assertTrue(summary_paths)
            self.assertTrue(comparison_paths)
            self.assertTrue(train_score_paths)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
