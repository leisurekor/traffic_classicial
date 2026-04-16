"""Compatibility layer for `networkx` graph objects used by the graph builder."""

from __future__ import annotations


try:
    import networkx as nx

    HAS_NETWORKX = True
    MultiDiGraph = nx.MultiDiGraph
except ImportError:
    HAS_NETWORKX = False

    class MultiDiGraph:
        """Small fallback subset of `networkx.MultiDiGraph` used in tests and CLI."""

        def __init__(self) -> None:
            """Initialize empty node and edge stores."""

            self._nodes: dict[str, dict[str, object]] = {}
            self._edges: dict[tuple[str, str, str], dict[str, object]] = {}
            self._edge_order: list[tuple[str, str, str]] = []

        def add_node(self, node_for_adding: str, **attrs: object) -> None:
            """Add or update a node with attribute data."""

            existing = self._nodes.setdefault(node_for_adding, {})
            existing.update(attrs)

        def add_edge(
            self,
            u_of_edge: str,
            v_of_edge: str,
            key: str | None = None,
            **attrs: object,
        ) -> str:
            """Add a directed edge and return its key."""

            edge_key = key or str(len(self._edge_order))
            composite_key = (u_of_edge, v_of_edge, edge_key)
            self._edges[composite_key] = dict(attrs)
            self._edge_order.append(composite_key)
            return edge_key

        def number_of_nodes(self) -> int:
            """Return the number of unique nodes."""

            return len(self._nodes)

        def number_of_edges(self) -> int:
            """Return the number of directed edges."""

            return len(self._edges)

        def has_node(self, node: str) -> bool:
            """Return `True` when the graph already contains the node."""

            return node in self._nodes

        def nodes(self, data: bool = False) -> list[object]:
            """Return nodes optionally paired with their attributes."""

            if data:
                return list(self._nodes.items())
            return list(self._nodes.keys())

        def edges(
            self,
            data: bool = False,
            keys: bool = False,
        ) -> list[object]:
            """Return edges optionally including keys and attributes."""

            rendered_edges: list[object] = []
            for u_of_edge, v_of_edge, edge_key in self._edge_order:
                attrs = dict(self._edges[(u_of_edge, v_of_edge, edge_key)])
                if keys and data:
                    rendered_edges.append((u_of_edge, v_of_edge, edge_key, attrs))
                elif keys:
                    rendered_edges.append((u_of_edge, v_of_edge, edge_key))
                elif data:
                    rendered_edges.append((u_of_edge, v_of_edge, attrs))
                else:
                    rendered_edges.append((u_of_edge, v_of_edge))
            return rendered_edges

    class _FallbackNetworkXModule:
        """Simple module-like wrapper exposing the fallback `MultiDiGraph`."""

        MultiDiGraph = MultiDiGraph

    nx = _FallbackNetworkXModule()


__all__ = ["HAS_NETWORKX", "MultiDiGraph", "nx"]
