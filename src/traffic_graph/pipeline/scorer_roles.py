"""Thin normalization helpers for graph scorer names and role labels."""

from __future__ import annotations

GRAPH_SCORER_ROLE_MAP: dict[str, str] = {
    "hybrid_max_rank_flow_node_max": "default_candidate",
    "flow_p90": "fallback",
    "decision_topk_flow_node": "experimental",
    "hybrid_decision_tail_balance": "experimental",
}
TABULAR_CONTROL_ROLE = "tabular_control"


def normalize_graph_scorer_role(scorer_name: str | None) -> str:
    """Return the stable role label for one graph scorer name when known."""

    if scorer_name is None:
        return ""
    return GRAPH_SCORER_ROLE_MAP.get(str(scorer_name).strip(), "")


def normalize_run_scorer_role(
    *,
    backend_name: str,
    scorer_name: str | None,
) -> str:
    """Return a stable scorer role label for graph and tabular run consumers."""

    if str(backend_name).strip().lower() == "tabular":
        return TABULAR_CONTROL_ROLE
    return normalize_graph_scorer_role(scorer_name)


__all__ = [
    "GRAPH_SCORER_ROLE_MAP",
    "TABULAR_CONTROL_ROLE",
    "normalize_graph_scorer_role",
    "normalize_run_scorer_role",
]
