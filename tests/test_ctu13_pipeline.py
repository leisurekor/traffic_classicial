"""Small regression tests for the CTU-13 adapter and edge-centric additions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.data import FlowDataset, FlowRecord
from traffic_graph.data.packet_prefix_graph import PacketPrefixNode, encode_packet_prefix_graph
from traffic_graph.datasets import (
    CTU13LabeledFlow,
    align_flow_dataset_to_ctu13_labels,
    infer_local_ctu13_manifest_entry,
    load_ctu13_manifest,
    merge_ctu13_manifest_with_local_raw,
)
from traffic_graph.features import fit_feature_preprocessor, transform_graph
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EndpointNode,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.graph.nx_compat import MultiDiGraph
from traffic_graph.pipeline.edge_calibration import (
    EdgeGraphScoreBreakdown,
    apply_edge_calibration,
    build_support_summary_aware_decision,
    calibrate_edge_profile,
    default_edge_calibration_profiles,
    suppressed_graph_score,
)
from traffic_graph.pipeline.candidate_region_proposal import propose_candidate_regions
from traffic_graph.pipeline.episode_graph_builder import build_episode_graph
from traffic_graph.pipeline.episode_proposal import propose_episodes
from traffic_graph.pipeline.episode_sessionization import sessionize_episodes
from traffic_graph.pipeline.micrograph_verifier import aggregate_micrograph_decisions, verify_candidate_region
from traffic_graph.pipeline.graph_extraction_modes import extract_flow_groups
from traffic_graph.pipeline.nuisance_boundary import (
    calibrate_nuisance_boundary,
    default_nuisance_boundary_profiles,
    score_graph_nuisance_aware,
)
from traffic_graph.pipeline.nuisance_aware_scoring import (
    aggregate_episode_graph_decision,
    calibrate_nuisance_aware_scores,
    relabel_episode_scores,
    score_episode_nuisance_aware,
)

def _load_script_module(name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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

    def test_multi_scenario_manifest_merge_from_local_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir) / "raw"
            scenario_48 = raw_root / "scenario_48"
            scenario_49 = raw_root / "scenario_49"
            scenario_48.mkdir(parents=True)
            scenario_49.mkdir(parents=True)
            (scenario_48 / "capture20110816-2.truncated.pcap").write_bytes(b"pcap")
            (scenario_48 / "capture20110816-2.binetflow.2format").write_text("StartTime\n", encoding="utf-8")
            (scenario_49 / "capture20110816-3.truncated.pcap").write_bytes(b"pcap")
            (scenario_49 / "capture20110816-3.binetflow.2format").write_text("StartTime\n", encoding="utf-8")
            merged = merge_ctu13_manifest_with_local_raw([], raw_root=raw_root, scenario_ids=("48", "49"))
        self.assertEqual([entry.scenario_id for entry in merged], ["48", "49"])
        self.assertEqual(merged[0].download_status, "downloaded")

    def test_infer_local_manifest_entry_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir) / "raw"
            scenario_dir = raw_root / "scenario_52"
            scenario_dir.mkdir(parents=True)
            (scenario_dir / "capture20110818-2.truncated.pcap").write_bytes(b"pcap")
            (scenario_dir / "capture20110818-2.binetflow.2format").write_text("StartTime\n", encoding="utf-8")
            entry = infer_local_ctu13_manifest_entry("52", raw_root=raw_root)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.scenario_id, "52")
        self.assertTrue(entry.pcap_path.endswith(".pcap"))

    def test_edge_calibration_profile_smoke(self) -> None:
        profile = next(item for item in default_edge_calibration_profiles() if item.require_dual_threshold)
        decision = calibrate_edge_profile(
            profile,
            [
                EdgeGraphScoreBreakdown(1.0, 1.0, 1.0, 2, 0.5, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8),
                EdgeGraphScoreBreakdown(2.0, 2.0, 2.0, 3, 0.6, 0.75, 0.7, 0.65, 1.5, 1.4, 0.75),
                EdgeGraphScoreBreakdown(3.0, 3.0, 3.0, 4, 0.8, 0.7, 0.65, 0.6, 2.1, 1.9, 0.7),
            ],
        )
        self.assertGreater(decision.score_threshold, 0.0)
        self.assertIsNotNone(decision.concentration_threshold)

    def test_unknown_suppression_decision_smoke(self) -> None:
        profile = next(item for item in default_edge_calibration_profiles() if item.name == "heldout_q95_top1")
        benign_breakdowns = [
            EdgeGraphScoreBreakdown(1.0, 1.0, 1.0, 2, 0.5, 0.9, 0.9, 0.8, 0.8, 0.7, 0.8, local_support_score=0.2, neighborhood_persistence_score=0.2, temporal_consistency_score=0.2),
            EdgeGraphScoreBreakdown(1.2, 1.2, 1.2, 2, 0.6, 0.85, 0.85, 0.8, 0.9, 0.8, 0.75, local_support_score=0.3, neighborhood_persistence_score=0.3, temporal_consistency_score=0.3),
        ]
        suppressed_decision = calibrate_edge_profile(
            profile,
            benign_breakdowns,
            suppression_enabled=True,
        )
        sparse_spike = EdgeGraphScoreBreakdown(2.5, 2.5, 2.5, 1, 0.05, 0.25, 0.25, 0.25, 2.4, 2.2, 0.96, local_support_score=0.1, neighborhood_persistence_score=0.1, temporal_consistency_score=0.1)
        concentrated = EdgeGraphScoreBreakdown(2.5, 2.5, 2.4, 3, 0.6, 0.9, 0.9, 0.85, 2.1, 2.0, 0.95, local_support_score=0.7, neighborhood_persistence_score=0.6, temporal_consistency_score=0.6)
        predictions = apply_edge_calibration([sparse_spike, concentrated], suppressed_decision)
        self.assertEqual(predictions, [0, 1])
        self.assertLess(
            suppressed_graph_score(sparse_spike, suppressed_decision),
            suppressed_graph_score(concentrated, suppressed_decision),
        )

    def test_support_aware_decision_smoke(self) -> None:
        profile = next(item for item in default_edge_calibration_profiles() if item.name == "heldout_q95_top1")
        benign_breakdowns = [
            EdgeGraphScoreBreakdown(1.0, 1.0, 0.9, 2, 0.4, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8, local_support_score=0.2, neighborhood_persistence_score=0.2, temporal_consistency_score=0.2),
            EdgeGraphScoreBreakdown(1.1, 1.1, 1.0, 2, 0.45, 0.75, 0.75, 0.7, 0.85, 0.75, 0.78, local_support_score=0.25, neighborhood_persistence_score=0.25, temporal_consistency_score=0.25),
        ]
        decision = build_support_summary_aware_decision(
            profile,
            benign_breakdowns,
            support_summary_mode="combined_support_summary",
        )
        weak_spike = EdgeGraphScoreBreakdown(2.0, 2.0, 1.9, 2, 0.5, 0.8, 0.8, 0.7, 1.5, 1.4, 0.8, local_support_score=0.1, neighborhood_persistence_score=0.1, temporal_consistency_score=0.1)
        supported = EdgeGraphScoreBreakdown(2.0, 2.0, 1.9, 2, 0.5, 0.8, 0.8, 0.7, 1.5, 1.4, 0.8, local_support_score=0.8, neighborhood_persistence_score=0.6, temporal_consistency_score=0.6)
        self.assertEqual(apply_edge_calibration([weak_spike, supported], decision), [0, 1])

    def test_local_support_density_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_local", "scripts/run_ctu13_binary_benchmark.py")
        sample = SimpleNamespace(
            graph=SimpleNamespace(
                stats=SimpleNamespace(node_count=4),
                nodes=(
                    EndpointNode("client:10.0.0.1:1111:tcp", "client", "10.0.0.1", 1111, "tcp"),
                    EndpointNode("client:10.0.0.1:2222:tcp", "client", "10.0.0.1", 2222, "tcp"),
                    EndpointNode("server:8.8.8.8:443:tcp", "server", "8.8.8.8", 443, "tcp"),
                    EndpointNode("server:8.8.8.9:443:tcp", "server", "8.8.8.9", 443, "tcp"),
                ),
                edges=(
                    CommunicationEdge("e1", "client:10.0.0.1:1111:tcp", "server:8.8.8.8:443:tcp", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:10.0.0.1:2222:tcp", "server:8.8.8.8:443:tcp", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:10.0.0.1:1111:tcp", "server:8.8.8.9:443:tcp", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                ),
            ),
        )
        summary = module._local_support_summary(sample, [0, 1, 2], np.asarray([0.9, 0.85, 0.2]), [0, 1])
        self.assertGreaterEqual(summary.local_support_edge_count, 2)
        self.assertGreater(summary.max_local_support_density, 0.5)

    def test_nuisance_boundary_smoke(self) -> None:
        profile = next(item for item in default_edge_calibration_profiles() if item.name == "heldout_q95_top1")
        benign_breakdowns = [
            EdgeGraphScoreBreakdown(1.0, 1.0, 0.9, 2, 0.4, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8, local_support_score=0.2, neighborhood_persistence_score=0.2, temporal_consistency_score=0.2, local_support_edge_density=0.3, cross_neighborhood_support_ratio=0.2),
            EdgeGraphScoreBreakdown(1.1, 1.1, 1.0, 2, 0.45, 0.75, 0.75, 0.7, 0.85, 0.75, 0.78, local_support_score=0.25, neighborhood_persistence_score=0.25, temporal_consistency_score=0.25, local_support_edge_density=0.35, cross_neighborhood_support_ratio=0.25),
        ]
        nuisance_breakdowns = [
            EdgeGraphScoreBreakdown(1.6, 1.7, 1.5, 3, 0.6, 0.9, 0.9, 0.8, 1.1, 0.9, 0.9, local_support_score=0.7, neighborhood_persistence_score=0.6, temporal_consistency_score=0.5, local_support_edge_density=0.8, cross_neighborhood_support_ratio=0.7),
            EdgeGraphScoreBreakdown(1.7, 1.8, 1.6, 3, 0.65, 0.88, 0.85, 0.82, 1.2, 0.95, 0.88, local_support_score=0.75, neighborhood_persistence_score=0.55, temporal_consistency_score=0.52, local_support_edge_density=0.82, cross_neighborhood_support_ratio=0.72),
        ]
        base_decision = build_support_summary_aware_decision(
            profile,
            benign_breakdowns,
            support_summary_mode="local_support_density",
        )
        boundary = calibrate_nuisance_boundary(
            base_decision,
            benign_breakdowns,
            nuisance_breakdowns,
            profile=default_nuisance_boundary_profiles()[0],
        )
        unknown_like = score_graph_nuisance_aware(nuisance_breakdowns[0], boundary)
        supported = score_graph_nuisance_aware(
            EdgeGraphScoreBreakdown(2.4, 2.5, 2.2, 4, 0.7, 0.92, 0.9, 0.85, 1.8, 1.4, 0.92, local_support_score=0.95, neighborhood_persistence_score=0.85, temporal_consistency_score=0.8, local_support_edge_density=0.9, cross_neighborhood_support_ratio=0.4),
            boundary,
        )
        self.assertIn(unknown_like.final_internal_state, {"nuisance_like", "malicious_like"})
        self.assertEqual(supported.final_internal_state, "malicious_like")

    def test_cross_neighborhood_persistence_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_neighborhood", "scripts/run_ctu13_binary_benchmark.py")
        sample = SimpleNamespace(
            graph=SimpleNamespace(
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:b", "server:y", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:c", "server:z", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                ),
            ),
        )
        summary = module._neighborhood_persistence_summary(sample, [0, 1, 2], [0, 1])
        self.assertEqual(summary.abnormal_neighborhood_count, 2)
        self.assertGreater(summary.cross_neighborhood_support_ratio, 0.5)

    def test_temporal_consistency_summary_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_temporal", "scripts/run_ctu13_binary_benchmark.py")
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", start_time=window_start, end_time=window_start + timedelta(seconds=0.5)),
                SimpleNamespace(logical_flow_id="lf2", start_time=window_start + timedelta(seconds=2), end_time=window_start + timedelta(seconds=2.5)),
            ),
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=3),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                ),
            ),
        )
        summary = module._temporal_consistency_summary(sample, [0, 1], [0, 1])
        self.assertEqual(summary.slice_abnormal_presence_count, 2)
        self.assertGreaterEqual(summary.slice_repeated_support_endpoints, 2)

    def test_ctu13_benchmark_output_schema_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark", "scripts/run_ctu13_binary_benchmark.py")
        row = {
            "evaluation_mode": "merged",
            "scenario_id": "merged_48_49_52",
            "model_name": "edge_temporal_binary_v2_nuisance_aware",
            "calibration_profile": "heldout_q95_top1",
            "percentile_setting": 95.0,
            "top_k_setting": 1,
            "suppression_enabled": False,
            "extraction_mode": "per_src_ip_within_window",
            "support_summary_mode": "local_support_density",
            "nuisance_boundary_mode": "nuisance_q95_margin025",
            "region_proposal_mode": None,
            "verifier_mode": None,
            "final_decision_mode": "nuisance_aware_boundary",
            "concentration_threshold_setting": None,
            "component_ratio_setting": None,
            "neighborhood_ratio_setting": None,
            "local_density_threshold": 0.4,
            "neighborhood_persistence_threshold": 0.3,
            "temporal_consistency_threshold": 0.2,
            "benign_boundary_value": 1.1,
            "nuisance_boundary_value": 0.8,
            "candidate_region_count_mean": 2.5,
            "selected_region_count_mean": 1.5,
            "selected_region_coverage_mean": 0.4,
            "single_edge_region_ratio": 0.25,
            "mean_candidate_time_span": 0.8,
            "train_benign_graphs": 4,
            "calib_benign_graphs": 2,
            "test_benign_graphs": 3,
            "test_malicious_graphs": 2,
            "test_unknown_graphs": 5,
            "threshold": 1.0,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 0.67,
            "fpr": 0.25,
            "roc_auc": 0.8,
            "pr_auc": 0.7,
            "background_hit_ratio": 0.2,
            "nuisance_rejection_rate": 0.6,
            "nuisance_like_false_positive_rate": 0.1,
            "benign_misrejected_as_nuisance_rate": 0.1,
            "malicious_blocked_by_nuisance_rate": 0.2,
            "unknown_count": 5,
            "unknown_score_mean": 0.5,
            "unknown_score_median": 0.4,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "benchmark.csv"
            md_path = Path(temp_dir) / "benchmark.md"
            module._write_csv(csv_path, [row])
            module._write_markdown(md_path, [row])
            csv_text = csv_path.read_text(encoding="utf-8")
            md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("calibration_profile", csv_text)
        self.assertIn("support_summary_mode", csv_text)
        self.assertIn("nuisance_boundary_mode", csv_text)
        self.assertIn("region_proposal_mode", csv_text)
        self.assertIn("single_edge_region_ratio", csv_text)
        self.assertIn("suppression_enabled", csv_text)
        self.assertIn("edge_temporal_binary_v2_nuisance_aware", md_text)

    def test_primary_graph_coverage_summary_schema_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_schema", "scripts/run_ctu13_binary_benchmark.py")
        row = {
            "scenario_id": "52",
            "window_size": 2,
            "graph_grouping_policy": "per_src_ip_within_window",
            "candidate_graph_count": 12,
            "benign_graph_count": 8,
            "malicious_graph_count": 2,
            "unknown_heavy_graph_count": 4,
            "filtered_out_reason": "none",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "primary_graph_extraction_summary.csv"
            md_path = Path(temp_dir) / "primary_graph_extraction_summary.md"
            module._write_simple_csv(csv_path, [row])
            module._write_primary_extraction_markdown(md_path, [row])
            csv_text = csv_path.read_text(encoding="utf-8")
            md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("graph_grouping_policy", csv_text)
        self.assertIn("unknown_heavy_graph_count", md_text)

    def test_unknown_suppression_markdown_schema_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_unknown", "scripts/run_ctu13_binary_benchmark.py")
        row = {
            "scenario_id": "52",
            "graph_id": "1:10.0.0.1",
            "label_type": "unknown",
            "top1_edge_score": 1.2,
            "topk_edge_mean": 1.0,
            "local_support_edge_count": 1,
            "local_support_edge_density": 0.3,
            "local_support_node_coverage": 0.2,
            "max_local_support_density": 0.3,
            "abnormal_neighborhood_count": 1,
            "cross_neighborhood_support_ratio": 0.3,
            "slice_abnormal_presence_count": 1,
            "slice_abnormal_consistency_ratio": 0.33,
            "decision_old": True,
            "decision_new": False,
            "suspected_failure_mode": "short_lived_slice_spike",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "unknown.csv"
            md_path = Path(temp_dir) / "unknown.md"
            module._write_simple_csv(csv_path, [row])
            module._write_unknown_suppression_markdown(md_path, [row])
            self.assertIn("decision_new", csv_path.read_text(encoding="utf-8"))
            self.assertIn("short_lived_slice_spike", md_path.read_text(encoding="utf-8"))

    def test_nuisance_boundary_markdown_schema_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_nuisance_schema", "scripts/run_ctu13_binary_benchmark.py")
        row = {
            "scenario_id": "52",
            "graph_id": "1:10.0.0.1",
            "label_type": "unknown",
            "anomaly_score": 1.4,
            "nuisance_score": 0.9,
            "malicious_support_score": 0.4,
            "benign_boundary_value": 1.0,
            "nuisance_boundary_value": 0.8,
            "final_internal_state": "nuisance_like",
            "final_binary_decision": 0,
            "top1_edge_score": 1.5,
            "topk_edge_mean": 1.3,
            "local_support_score": 0.8,
            "neighborhood_persistence_score": 0.6,
            "temporal_consistency_score": 0.4,
            "suspected_failure_mode": "nuisance_boundary_miss",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "nuisance_boundary.csv"
            md_path = Path(temp_dir) / "nuisance_boundary.md"
            module._write_simple_csv(csv_path, [row])
            module._write_nuisance_boundary_markdown(md_path, [row])
            self.assertIn("final_internal_state", csv_path.read_text(encoding="utf-8"))
            self.assertIn("nuisance_like", md_path.read_text(encoding="utf-8"))

    def test_candidate_region_proposal_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", start_time=window_start, end_time=window_start + timedelta(seconds=0.2)),
                SimpleNamespace(logical_flow_id="lf2", start_time=window_start + timedelta(seconds=0.3), end_time=window_start + timedelta(seconds=0.5)),
                SimpleNamespace(logical_flow_id="lf3", start_time=window_start + timedelta(seconds=2.0), end_time=window_start + timedelta(seconds=2.2)),
            ),
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=3),
                nodes=(
                    EndpointNode("client:a", "client", "10.0.0.1", 1, "tcp"),
                    EndpointNode("client:b", "client", "10.0.0.1", 2, "tcp"),
                    EndpointNode("server:x", "server", "8.8.8.8", 443, "tcp"),
                    EndpointNode("server:y", "server", "8.8.8.9", 443, "tcp"),
                ),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:b", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:a", "server:y", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                ),
            ),
        )
        regions = propose_candidate_regions(
            sample,
            [0, 1, 2],
            np.asarray([0.9, 0.8, 0.2]),
            proposal_mode="edge_seed_region",
            top_k=2,
            slice_count=3,
        )
        self.assertGreaterEqual(len(regions), 1)
        self.assertGreaterEqual(regions[0].candidate_edge_count, 2)

    def test_graph_extraction_mode_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        batch = SimpleNamespace(
            window_start=window_start,
            window_end=window_start + timedelta(seconds=3),
        )
        flows = [
            SimpleNamespace(
                logical_flow_id="lf1",
                src_ip="10.0.0.1",
                dst_ip="8.8.8.8",
                dst_port=443,
                protocol="tcp",
                start_time=window_start,
                end_time=window_start + timedelta(seconds=0.2),
            ),
            SimpleNamespace(
                logical_flow_id="lf2",
                src_ip="10.0.0.1",
                dst_ip="8.8.8.8",
                dst_port=443,
                protocol="tcp",
                start_time=window_start + timedelta(seconds=1.2),
                end_time=window_start + timedelta(seconds=1.4),
            ),
        ]
        groups = extract_flow_groups(batch, flows, extraction_mode="short_temporal_slice_src_pair", slice_count=3)
        self.assertGreaterEqual(len(groups), 2)

    def test_micrograph_verifier_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", start_time=window_start, end_time=window_start + timedelta(seconds=0.2)),
                SimpleNamespace(logical_flow_id="lf2", start_time=window_start + timedelta(seconds=0.4), end_time=window_start + timedelta(seconds=0.6)),
            ),
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=2),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                ),
            ),
        )
        candidate = propose_candidate_regions(
            sample,
            [0, 1],
            np.asarray([0.9, 0.8]),
            proposal_mode="temporal_burst_region",
            top_k=2,
            slice_count=3,
        )[0]
        result = verify_candidate_region(sample, candidate, [0, 1], np.asarray([0.9, 0.8]), slice_count=3)
        self.assertGreater(result.micrograph_density_score, 0.0)
        self.assertGreater(result.micrograph_score, 0.0)

    def test_episode_proposal_and_nuisance_scoring_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            scenario_id="52",
            window_index=0,
            group_key="client-a",
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", start_time=window_start, end_time=window_start + timedelta(seconds=0.2)),
                SimpleNamespace(logical_flow_id="lf2", start_time=window_start + timedelta(seconds=0.4), end_time=window_start + timedelta(seconds=0.7)),
                SimpleNamespace(logical_flow_id="lf3", start_time=window_start + timedelta(seconds=1.1), end_time=window_start + timedelta(seconds=1.4)),
            ),
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=2),
                stats=SimpleNamespace(communication_edge_count=3),
                nodes=(
                    EndpointNode("client:a", "client", "10.0.0.1", 1111, "tcp"),
                    EndpointNode("server:x", "server", "8.8.8.8", 443, "tcp"),
                    EndpointNode("server:y", "server", "8.8.8.9", 443, "tcp"),
                ),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:a", "server:y", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                ),
            ),
        )
        episodes = propose_episodes(
            sample,
            [0, 1, 2],
            np.asarray([0.9, 0.85, 0.3]),
            proposal_mode="repeated_pair_episode",
            top_k=2,
            slice_count=3,
        )
        self.assertGreaterEqual(len(episodes), 1)
        self.assertGreaterEqual(episodes[0].edge_count, 2)
        episode_graph = build_episode_graph(sample, episodes)
        episode_scores = [score_episode_nuisance_aware(episode, episode_graph) for episode in episodes]
        calibration = calibrate_nuisance_aware_scores(episode_scores, episode_scores, percentile=95.0)
        relabeled = relabel_episode_scores(episode_scores, calibration)
        decision = aggregate_episode_graph_decision(
            relabeled,
            episodes,
            final_decision_mode="consistency_aware_episode",
            total_edge_count=3,
            graph_threshold=calibration.consistency_threshold,
        )
        self.assertGreaterEqual(episode_graph.episode_count, 1)
        self.assertGreaterEqual(decision.selected_episode_coverage, 0.0)

    def test_episode_sessionization_repeated_pair_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=6),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:a", "server:x", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                ),
            ),
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start, end_time=window_start + timedelta(seconds=0.2), directions=("forward",), prefix_behavior_signature=1),
                SimpleNamespace(logical_flow_id="lf2", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=1), end_time=window_start + timedelta(seconds=1.2), directions=("forward",), prefix_behavior_signature=1),
                SimpleNamespace(logical_flow_id="lf3", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=2), end_time=window_start + timedelta(seconds=2.2), directions=("forward",), prefix_behavior_signature=1),
            ),
        )
        episodes = sessionize_episodes(
            sample,
            [0, 1, 2],
            np.asarray([0.8, 0.7, 0.75]),
            stitching_mode="repeated_pair_temporal_continuity",
            slice_count=3,
            max_gap_seconds=2.0,
        )
        self.assertEqual(len(episodes), 1)
        self.assertGreaterEqual(episodes[0].flow_count, 3)
        self.assertGreater(episodes[0].continuity_span, 0.0)

    def test_episode_sessionization_protocol_chain_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=6),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:y", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                ),
            ),
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start, end_time=window_start + timedelta(seconds=0.2), directions=("forward",), prefix_behavior_signature=5),
                SimpleNamespace(logical_flow_id="lf2", src_ip="10.0.0.1", dst_ip="8.8.8.9", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=0.8), end_time=window_start + timedelta(seconds=1.0), directions=("forward",), prefix_behavior_signature=5),
            ),
        )
        episodes = sessionize_episodes(
            sample,
            [0, 1],
            np.asarray([0.9, 0.85]),
            stitching_mode="protocol_consistent_interaction_chain",
            slice_count=3,
            max_gap_seconds=2.0,
        )
        self.assertEqual(len(episodes), 1)
        self.assertGreaterEqual(episodes[0].protocol_consistency_score, 0.6)

    def test_episode_sessionization_burst_smoke(self) -> None:
        window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        sample = SimpleNamespace(
            graph=SimpleNamespace(
                window_start=window_start,
                window_end=window_start + timedelta(seconds=6),
                edges=(
                    CommunicationEdge("e1", "client:a", "server:x", "communication", "lf1", 1, 1, 1.0, 1, False, ("lf1",)),
                    CommunicationEdge("e2", "client:a", "server:x", "communication", "lf2", 1, 1, 1.0, 1, False, ("lf2",)),
                    CommunicationEdge("e3", "client:a", "server:y", "communication", "lf3", 1, 1, 1.0, 1, False, ("lf3",)),
                    CommunicationEdge("e4", "client:a", "server:y", "communication", "lf4", 1, 1, 1.0, 1, False, ("lf4",)),
                ),
            ),
            logical_flows=(
                SimpleNamespace(logical_flow_id="lf1", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start, end_time=window_start + timedelta(seconds=0.2), directions=("forward",), prefix_behavior_signature=1),
                SimpleNamespace(logical_flow_id="lf2", src_ip="10.0.0.1", dst_ip="8.8.8.8", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=0.4), end_time=window_start + timedelta(seconds=0.6), directions=("forward",), prefix_behavior_signature=1),
                SimpleNamespace(logical_flow_id="lf3", src_ip="10.0.0.1", dst_ip="8.8.8.9", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=2.1), end_time=window_start + timedelta(seconds=2.3), directions=("forward",), prefix_behavior_signature=1),
                SimpleNamespace(logical_flow_id="lf4", src_ip="10.0.0.1", dst_ip="8.8.8.9", dst_port=443, protocol="tcp", start_time=window_start + timedelta(seconds=2.4), end_time=window_start + timedelta(seconds=2.6), directions=("forward",), prefix_behavior_signature=1),
            ),
        )
        episodes = sessionize_episodes(
            sample,
            [0, 1, 2, 3],
            np.asarray([0.7, 0.8, 0.85, 0.9]),
            stitching_mode="repeated_local_burst_stitching",
            slice_count=3,
            max_gap_seconds=2.0,
        )
        self.assertGreaterEqual(len(episodes), 1)
        self.assertGreaterEqual(episodes[0].burst_repeat_count, 1)

    def test_two_stage_final_decision_smoke(self) -> None:
        candidate_a = SimpleNamespace(relative_edge_indices=(0, 1))
        candidate_b = SimpleNamespace(relative_edge_indices=(2,))
        result_a = SimpleNamespace(micrograph_score=0.9)
        result_b = SimpleNamespace(micrograph_score=0.7)
        decision = aggregate_micrograph_decisions(
            [result_a, result_b],
            [candidate_a, candidate_b],
            final_decision_mode="consistency_aware_aggregation",
            total_edge_count=4,
            score_threshold=0.8,
        )
        self.assertTrue(decision.is_positive)
        self.assertGreater(decision.selected_region_coverage, 0.5)

    def test_proposal_quality_diagnosis_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_quality", "scripts/run_ctu13_binary_benchmark.py")
        candidate_rows = [
            {
                "scenario_id": "52",
                "extraction_mode": "short_temporal_slice_src_pair",
                "candidate_region_id": "edge_seed_region_v2:0",
                "candidate_edge_count": 3,
                "candidate_time_span": 1.0,
                "repeated_endpoint_count": 2,
                "support_cluster_density": 0.8,
                "candidate_score_mean": 0.9,
                "label_type": "malicious",
            },
            {
                "scenario_id": "52",
                "extraction_mode": "short_temporal_slice_src_pair",
                "candidate_region_id": "edge_seed_region_v2:1",
                "candidate_edge_count": 1,
                "candidate_time_span": 0.1,
                "repeated_endpoint_count": 0,
                "support_cluster_density": 0.2,
                "candidate_score_mean": 0.3,
                "label_type": "unknown",
            },
        ]
        micro_rows = [
            {
                "scenario_id": "52",
                "extraction_mode": "short_temporal_slice_src_pair",
                "candidate_region_id": "edge_seed_region_v2:0",
                "micrograph_score": 0.8,
                "label_type": "malicious",
            },
            {
                "scenario_id": "52",
                "extraction_mode": "short_temporal_slice_src_pair",
                "candidate_region_id": "edge_seed_region_v2:1",
                "micrograph_score": 0.2,
                "label_type": "unknown",
            },
        ]
        rows = module._proposal_quality_diagnosis_rows(candidate_rows, micro_rows)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["proposal_score_gap"], 0.0)
        self.assertGreater(rows[0]["micrograph_score_gap"], 0.0)

    def test_episode_quality_diagnosis_smoke(self) -> None:
        module = _load_script_module("run_ctu13_binary_benchmark_episode_quality", "scripts/run_ctu13_binary_benchmark.py")
        construction_rows = [
            {
                "scenario_id": "52",
                "proposal_mode": "repeated_pair_episode",
                "episode_route_version": "sessionized_v2",
                "stitching_mode": "repeated_pair_temporal_continuity",
                "episode_edge_count": 3,
                "episode_time_span": 1.0,
                "involved_flows": 3,
                "repeated_pair_count": 1,
                "support_cluster_density": 0.8,
                "protocol_consistency_score": 0.9,
                "burst_persistence": 0.7,
                "label_type": "malicious",
            },
            {
                "scenario_id": "52",
                "proposal_mode": "repeated_pair_episode",
                "episode_route_version": "sessionized_v2",
                "stitching_mode": "repeated_pair_temporal_continuity",
                "episode_edge_count": 2,
                "episode_time_span": 0.2,
                "involved_flows": 1,
                "repeated_pair_count": 0,
                "support_cluster_density": 0.3,
                "protocol_consistency_score": 0.4,
                "burst_persistence": 0.2,
                "label_type": "unknown",
            },
        ]
        nuisance_rows = [
            {
                "scenario_id": "52",
                "episode_id": "repeated_pair_episode:0",
                "episode_route_version": "sessionized_v2",
                "stitching_mode": "repeated_pair_temporal_continuity",
                "episode_score": 0.8,
                "nuisance_score": 0.2,
                "label_type": "malicious",
            },
            {
                "scenario_id": "52",
                "episode_id": "repeated_pair_episode:1",
                "episode_route_version": "sessionized_v2",
                "stitching_mode": "repeated_pair_temporal_continuity",
                "episode_score": 0.3,
                "nuisance_score": 0.7,
                "label_type": "unknown",
            },
        ]
        rows = module._episode_quality_diagnosis_rows(construction_rows, nuisance_rows)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["episode_score_gap"], 0.0)
        self.assertGreater(rows[0]["nuisance_score_gap"], 0.0)
        self.assertGreater(rows[0]["mean_protocol_consistency_score"], 0.0)

    def test_profile_tiebreak_prefers_lower_background_hit_ratio(self) -> None:
        module = _load_script_module("run_ctu13_edge_centric_comparison", "scripts/run_ctu13_edge_centric_comparison.py")
        selected = module._select_edge_row(
            [
                {
                    "model_name": "edge_temporal_binary_v2",
                    "calibration_profile": "a",
                    "f1": "0.8",
                    "recall": "0.9",
                    "fpr": "0.1",
                    "background_hit_ratio": "0.4",
                    "support_summary_mode": "old_concentration",
                },
                {
                    "model_name": "edge_temporal_binary_v2",
                    "calibration_profile": "b",
                    "f1": "0.8",
                    "recall": "0.9",
                    "fpr": "0.1",
                    "background_hit_ratio": "0.2",
                    "support_summary_mode": "combined_support_summary",
                },
            ]
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["calibration_profile"], "b")

    def test_nuisance_tiebreak_prefers_lower_background_hit_ratio(self) -> None:
        module = _load_script_module("run_ctu13_edge_centric_comparison_nuisance", "scripts/run_ctu13_edge_centric_comparison.py")
        selected = module._select_nuisance_row(
            [
                {
                    "model_name": "edge_temporal_binary_v2_nuisance_aware",
                    "calibration_profile": "a",
                    "f1": "0.8",
                    "recall": "0.9",
                    "fpr": "0.1",
                    "background_hit_ratio": "0.4",
                    "malicious_blocked_by_nuisance_rate": "0.2",
                    "support_summary_mode": "old_concentration",
                },
                {
                    "model_name": "edge_temporal_binary_v2_nuisance_aware",
                    "calibration_profile": "b",
                    "f1": "0.8",
                    "recall": "0.9",
                    "fpr": "0.1",
                    "background_hit_ratio": "0.2",
                    "malicious_blocked_by_nuisance_rate": "0.1",
                    "support_summary_mode": "local_support_density",
                },
            ]
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["calibration_profile"], "b")

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
