"""Model interfaces and minimal unsupervised graph autoencoder components."""

from traffic_graph.models.base import DetectorSpec, UnsupervisedDetector
from traffic_graph.models.encoders import (
    CategoricalEdgeProjector,
    GraphEncoder,
    GraphEncoderOutput,
    GraphSAGELayer,
    TemporalEdgeProjector,
)
from traffic_graph.models.gae import GraphAutoEncoder
from traffic_graph.models.losses import (
    GraphAutoEncoderLossOutput,
    ReconstructionLossWeights,
    reconstruction_loss,
)
from traffic_graph.models.model_types import (
    DEFAULT_TEMPORAL_EDGE_FIELD_NAMES,
    GraphAutoEncoderConfig,
    GraphAutoEncoderOutput,
    GraphTensorBatch,
    packed_graph_to_tensors,
)
from traffic_graph.models.scoring import (
    build_edge_score_row,
    build_edge_score_rows,
    build_flow_score_row,
    build_flow_score_rows,
    build_graph_score_row,
    build_node_score_row,
    build_node_score_rows,
    compute_edge_anomaly_scores,
    compute_graph_anomaly_scores,
    compute_node_anomaly_scores,
)

__all__ = [
    "DetectorSpec",
    "DEFAULT_TEMPORAL_EDGE_FIELD_NAMES",
    "CategoricalEdgeProjector",
    "GraphAutoEncoder",
    "GraphAutoEncoderConfig",
    "GraphAutoEncoderLossOutput",
    "GraphAutoEncoderOutput",
    "GraphEncoder",
    "GraphEncoderOutput",
    "GraphSAGELayer",
    "GraphTensorBatch",
    "ReconstructionLossWeights",
    "TemporalEdgeProjector",
    "UnsupervisedDetector",
    "build_edge_score_row",
    "build_edge_score_rows",
    "build_flow_score_row",
    "build_flow_score_rows",
    "build_graph_score_row",
    "build_node_score_row",
    "build_node_score_rows",
    "compute_edge_anomaly_scores",
    "compute_graph_anomaly_scores",
    "compute_node_anomaly_scores",
    "packed_graph_to_tensors",
    "reconstruction_loss",
]
