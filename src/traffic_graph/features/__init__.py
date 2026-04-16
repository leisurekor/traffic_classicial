"""Feature extraction, normalization, and packing utilities for graphs."""

from traffic_graph.features.feature_pack import (
    build_model_feature_view,
    extract_node_structure_features,
    fit_feature_preprocessor,
    transform_graph,
    transform_graphs,
)
from traffic_graph.features.feature_types import (
    EDGE_BASE_FEATURE_FIELDS,
    EDGE_DISCRETE_FEATURE_FIELDS,
    EDGE_PACKED_FEATURE_FIELDS,
    EDGE_PATTERN_CODE_FIELDS,
    NODE_BASE_FEATURE_FIELDS,
    NODE_PACKED_FEATURE_FIELDS,
    NODE_STRUCTURE_FEATURE_FIELDS,
    EdgeFeatureSet,
    GraphFeatureView,
    NodeFeatureSet,
)
from traffic_graph.features.graph_tensor_view import (
    PackedGraphInput,
    PackedGraphMetadata,
    summarize_packed_graph_input,
)
from traffic_graph.features.normalization import FeaturePreprocessor, MatrixNormalizer
from traffic_graph.features.stats_features import (
    EDGE_TYPE_ENCODING,
    ENDPOINT_TYPE_ENCODING,
    PROTO_ENCODING,
    build_base_feature_views,
    extract_edge_base_features,
    extract_node_base_features,
    summarize_feature_view,
)

__all__ = [
    "EDGE_BASE_FEATURE_FIELDS",
    "EDGE_DISCRETE_FEATURE_FIELDS",
    "EDGE_PACKED_FEATURE_FIELDS",
    "EDGE_PATTERN_CODE_FIELDS",
    "EDGE_TYPE_ENCODING",
    "ENDPOINT_TYPE_ENCODING",
    "EdgeFeatureSet",
    "FeaturePreprocessor",
    "GraphFeatureView",
    "MatrixNormalizer",
    "NODE_BASE_FEATURE_FIELDS",
    "NODE_PACKED_FEATURE_FIELDS",
    "NODE_STRUCTURE_FEATURE_FIELDS",
    "NodeFeatureSet",
    "PROTO_ENCODING",
    "PackedGraphInput",
    "PackedGraphMetadata",
    "build_base_feature_views",
    "build_model_feature_view",
    "extract_edge_base_features",
    "extract_node_base_features",
    "extract_node_structure_features",
    "fit_feature_preprocessor",
    "summarize_feature_view",
    "summarize_packed_graph_input",
    "transform_graph",
    "transform_graphs",
]
