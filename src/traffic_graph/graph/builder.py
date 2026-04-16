"""Compatibility wrappers for endpoint interaction graph construction."""

from __future__ import annotations

from collections.abc import Iterable

from traffic_graph.config import GraphConfig
from traffic_graph.data.preprocessing import LogicalFlowBatch
from traffic_graph.graph.endpoint_graph import (
    EndpointGraphBuilder,
    build_endpoint_graph,
    build_endpoint_graphs,
)
from traffic_graph.graph.graph_types import InteractionGraph

GraphSnapshot = InteractionGraph


class FlowInteractionGraphBuilder:
    """Compatibility wrapper around the endpoint graph builder."""

    def __init__(self, config: GraphConfig | None = None) -> None:
        """Store graph configuration and expose the endpoint graph builder API."""

        self.config = config or GraphConfig()
        self._builder = EndpointGraphBuilder(self.config)

    def build(self, window_batch: LogicalFlowBatch) -> InteractionGraph:
        """Build an interaction graph for one logical-flow window batch."""

        return build_endpoint_graph(window_batch, graph_config=self.config)

    def build_many(
        self,
        window_batches: Iterable[LogicalFlowBatch],
    ) -> list[InteractionGraph]:
        """Build interaction graphs for multiple logical-flow window batches."""

        return build_endpoint_graphs(window_batches, graph_config=self.config)
