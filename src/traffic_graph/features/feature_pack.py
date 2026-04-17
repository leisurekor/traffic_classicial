"""Feature packing and numpy-based model-input preparation for graphs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from traffic_graph.config import FeatureNormalizationConfig
from traffic_graph.features.feature_types import (
    EDGE_PACKED_FEATURE_FIELDS,
    NODE_PACKED_FEATURE_FIELDS,
    NODE_STRUCTURE_FEATURE_FIELDS,
    EdgeFeatureSet,
    FeatureRow,
    GraphFeatureView,
    NodeFeatureSet,
)
from traffic_graph.features.graph_tensor_view import PackedGraphInput, PackedGraphMetadata
from traffic_graph.features.normalization import FeaturePreprocessor
from traffic_graph.features.stats_features import (
    extract_edge_base_features,
    extract_node_base_features,
)
from traffic_graph.graph.graph_types import CommunicationEdge, EndpointNode, InteractionGraph


def _incident_edges(
    graph_sample: InteractionGraph,
    node_id: str,
) -> list[CommunicationEdge]:
    """Return all graph edges incident to a node identifier."""

    return [
        edge
        for edge in graph_sample.edges
        if edge.source_node_id == node_id or edge.target_node_id == node_id
    ]


def _node_structure_feature_row(
    graph_sample: InteractionGraph,
    node: EndpointNode,
) -> FeatureRow:
    """Build lightweight graph-structure features for one endpoint node."""

    incident_edges = _incident_edges(graph_sample, node.node_id)
    communication_in_degree = sum(
        edge.edge_type == "communication" and edge.target_node_id == node.node_id
        for edge in graph_sample.edges
    )
    communication_out_degree = sum(
        edge.edge_type == "communication" and edge.source_node_id == node.node_id
        for edge in graph_sample.edges
    )
    unique_neighbor_count = len(
        {
            edge.target_node_id if edge.source_node_id == node.node_id else edge.source_node_id
            for edge in incident_edges
        }
    )
    return {
        "total_degree": len(incident_edges),
        "communication_in_degree": communication_in_degree,
        "communication_out_degree": communication_out_degree,
        "unique_neighbor_count": unique_neighbor_count,
    }


def extract_node_structure_features(graph_sample: InteractionGraph) -> NodeFeatureSet:
    """Extract fixed-order structural node features for every graph node."""

    ordered_node_ids = tuple(node.node_id for node in graph_sample.nodes)
    feature_rows = tuple(
        _node_structure_feature_row(graph_sample, node) for node in graph_sample.nodes
    )
    feature_by_node_id = {
        node_id: feature_row
        for node_id, feature_row in zip(ordered_node_ids, feature_rows, strict=True)
    }
    feature_matrix = tuple(
        tuple(float(feature_row[field_name]) for field_name in NODE_STRUCTURE_FEATURE_FIELDS)
        for feature_row in feature_rows
    )
    return NodeFeatureSet(
        field_names=NODE_STRUCTURE_FEATURE_FIELDS,
        ordered_node_ids=ordered_node_ids,
        feature_rows=feature_rows,
        feature_by_node_id=feature_by_node_id,
        feature_matrix=feature_matrix,
    )


def _merge_node_feature_sets(feature_sets: tuple[NodeFeatureSet, ...]) -> NodeFeatureSet:
    """Merge multiple node feature sets that share the same node ordering."""

    if not feature_sets:
        raise ValueError("At least one node feature set is required for merging.")

    ordered_node_ids = feature_sets[0].ordered_node_ids
    for feature_set in feature_sets[1:]:
        if feature_set.ordered_node_ids != ordered_node_ids:
            raise ValueError("Node feature sets must share the same node order.")

    field_names = tuple(
        field_name
        for feature_set in feature_sets
        for field_name in feature_set.field_names
    )
    feature_rows = []
    for row_index, node_id in enumerate(ordered_node_ids):
        merged_row: FeatureRow = {}
        for feature_set in feature_sets:
            merged_row.update(feature_set.feature_rows[row_index])
        feature_rows.append(merged_row)

    feature_rows_tuple = tuple(feature_rows)
    feature_matrix = tuple(
        tuple(float(feature_row[field_name]) for field_name in field_names)
        for feature_row in feature_rows_tuple
    )
    feature_by_node_id = {
        node_id: feature_row
        for node_id, feature_row in zip(ordered_node_ids, feature_rows_tuple, strict=True)
    }
    return NodeFeatureSet(
        field_names=field_names,
        ordered_node_ids=ordered_node_ids,
        feature_rows=feature_rows_tuple,
        feature_by_node_id=feature_by_node_id,
        feature_matrix=feature_matrix,
    )


def build_model_feature_view(
    graph_sample: InteractionGraph,
    *,
    include_graph_structural_features: bool = True,
) -> GraphFeatureView:
    """Build the pre-normalization feature view used for packed graph inputs."""

    node_base_features = extract_node_base_features(graph_sample)
    if include_graph_structural_features:
        node_structure_features = extract_node_structure_features(graph_sample)
        merged_node_features = _merge_node_feature_sets(
            (node_base_features, node_structure_features)
        )
    else:
        merged_node_features = node_base_features
    edge_features = extract_edge_base_features(graph_sample)
    return GraphFeatureView(
        node_features=merged_node_features,
        edge_features=edge_features,
    )


def _stack_feature_matrices(matrices: Iterable[np.ndarray], width: int) -> np.ndarray:
    """Stack 2D feature matrices while preserving empty-matrix shape semantics."""

    matrix_list = [matrix for matrix in matrices if matrix.size > 0]
    if not matrix_list:
        return np.zeros((0, width), dtype=float)
    return np.vstack(matrix_list)


def _matrix_from_feature_rows(
    feature_matrix: tuple[tuple[float, ...], ...],
    width: int,
) -> np.ndarray:
    """Convert tuple-based feature rows into a stable 2D numpy matrix."""

    if not feature_matrix:
        return np.zeros((0, width), dtype=float)
    return np.asarray(feature_matrix, dtype=float)


def fit_feature_preprocessor(
    graphs: Iterable[InteractionGraph],
    normalization_config: FeatureNormalizationConfig | None = None,
    *,
    include_graph_structural_features: bool = True,
) -> FeaturePreprocessor:
    """Fit node and edge normalizers on a batch of graphs."""

    config = normalization_config or FeatureNormalizationConfig()
    graph_list = list(graphs)
    feature_views = [
        build_model_feature_view(
            graph,
            include_graph_structural_features=include_graph_structural_features,
        )
        for graph in graph_list
    ]
    node_field_names = (
        feature_views[0].node_features.field_names
        if feature_views
        else NODE_PACKED_FEATURE_FIELDS
    )
    edge_field_names = (
        feature_views[0].edge_features.field_names
        if feature_views
        else EDGE_PACKED_FEATURE_FIELDS
    )
    preprocessor = FeaturePreprocessor(
        node_field_names=node_field_names,
        edge_field_names=edge_field_names,
        node_excluded_fields=config.exclude_node_fields,
        edge_excluded_fields=config.exclude_edge_fields,
        method=config.method,
        enabled=config.enabled,
    )

    node_matrix = _stack_feature_matrices(
        (
            np.asarray(feature_view.node_features.feature_matrix, dtype=float)
            for feature_view in feature_views
        ),
        len(node_field_names),
    )
    edge_matrix = _stack_feature_matrices(
        (
            np.asarray(feature_view.edge_features.feature_matrix, dtype=float)
            for feature_view in feature_views
        ),
        len(edge_field_names),
    )
    return preprocessor.fit(node_matrix=node_matrix, edge_matrix=edge_matrix)


def transform_graph(
    graph_sample: InteractionGraph,
    preprocessor: FeaturePreprocessor,
    *,
    include_graph_structural_features: bool = True,
) -> PackedGraphInput:
    """Transform one graph into a packed numpy-based model input object."""

    feature_view = build_model_feature_view(
        graph_sample,
        include_graph_structural_features=include_graph_structural_features,
    )
    node_matrix = _matrix_from_feature_rows(
        feature_view.node_features.feature_matrix,
        len(feature_view.node_features.field_names),
    )
    edge_matrix = _matrix_from_feature_rows(
        feature_view.edge_features.feature_matrix,
        len(feature_view.edge_features.field_names),
    )
    normalized_node_matrix = preprocessor.transform_node_matrix(node_matrix)
    normalized_edge_matrix = preprocessor.transform_edge_matrix(edge_matrix)
    normalized_node_matrix = np.nan_to_num(
        np.asarray(normalized_node_matrix, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized_edge_matrix = np.nan_to_num(
        np.asarray(normalized_edge_matrix, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    node_ids = feature_view.node_features.ordered_node_ids
    edge_ids = feature_view.edge_features.ordered_edge_ids
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_id_to_index = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    edge_pairs = [
        (
            node_id_to_index[edge.source_node_id],
            node_id_to_index[edge.target_node_id],
        )
        for edge in graph_sample.edges
    ]
    edge_index = (
        np.asarray(edge_pairs, dtype=np.int64).T
        if edge_pairs
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_types = tuple(
        int(feature_view.edge_features.feature_rows[index]["edge_type"])
        for index, _edge_id in enumerate(edge_ids)
    )

    return PackedGraphInput(
        node_features=normalized_node_matrix,
        edge_features=normalized_edge_matrix,
        edge_index=edge_index,
        node_ids=node_ids,
        edge_ids=edge_ids,
        node_id_to_index=node_id_to_index,
        edge_id_to_index=edge_id_to_index,
        edge_types=edge_types,
        node_feature_fields=feature_view.node_features.field_names,
        edge_feature_fields=feature_view.edge_features.field_names,
        node_discrete_mask=preprocessor.node_discrete_mask,
        edge_discrete_mask=preprocessor.edge_discrete_mask,
        metadata=PackedGraphMetadata(
            window_index=graph_sample.window_index,
            window_start=graph_sample.window_start,
            window_end=graph_sample.window_end,
            node_count=graph_sample.node_count,
            edge_count=graph_sample.edge_count,
            communication_edge_count=graph_sample.stats.communication_edge_count,
            association_edge_count=graph_sample.stats.association_edge_count,
        ),
    )


def transform_graphs(
    graphs: Iterable[InteractionGraph],
    preprocessor: FeaturePreprocessor,
    *,
    include_graph_structural_features: bool = True,
) -> list[PackedGraphInput]:
    """Transform multiple graphs into packed model-input objects."""

    return [
        transform_graph(
            graph,
            preprocessor,
            include_graph_structural_features=include_graph_structural_features,
        )
        for graph in graphs
    ]


__all__ = [
    "build_model_feature_view",
    "extract_node_structure_features",
    "fit_feature_preprocessor",
    "transform_graph",
    "transform_graphs",
]
