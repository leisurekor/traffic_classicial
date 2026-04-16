"""Extract decision paths and rule records from surrogate trees."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import cast

import numpy as np

from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.rule_records import RuleOperator, RulePathCondition, RuleRecord
from traffic_graph.explain.surrogate_types import SurrogateTreeArtifact


def _coerce_prediction(value: object) -> object:
    """Convert a model prediction into a stable scalar."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _coerce_numeric(value: object) -> float:
    """Convert a feature value into a stable float."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, numbers.Real) or isinstance(value, (np.integer, np.floating)):
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return 0.0
        if np.isfinite(converted):
            return converted
    return 0.0


def _tree_model(tree_artifact: SurrogateTreeArtifact) -> object:
    """Return the fitted tree model from an artifact or raise a clear error."""

    model = tree_artifact.model
    tree = getattr(model, "tree_", None)
    if tree is None:
        raise ValueError("The surrogate tree artifact does not contain a fitted tree.")
    return model


def _validate_scope(tree_artifact: SurrogateTreeArtifact, sample: ExplanationSample) -> None:
    """Ensure the sample scope matches the fitted tree scope."""

    if sample.scope != tree_artifact.summary.sample_scope:
        raise ValueError(
            "Sample scope does not match surrogate-tree scope: "
            f"sample={sample.scope}, tree={tree_artifact.summary.sample_scope}"
        )


def _feature_vector(
    tree_artifact: SurrogateTreeArtifact,
    sample: ExplanationSample,
) -> np.ndarray:
    """Build a feature vector aligned to the surrogate-tree feature order."""

    feature_names = tree_artifact.feature_names
    if feature_names == ("constant_bias",):
        return np.asarray([[1.0]], dtype=float)
    row = _sample_feature_row(sample, feature_names=feature_names)
    return np.asarray([row], dtype=float)


def _sample_feature_row(
    sample: ExplanationSample,
    feature_names: Sequence[str],
) -> list[float]:
    """Build a row vector aligned with the surrogate-tree feature order."""

    row: list[float] = []
    for feature_name in feature_names:
        if feature_name == "constant_bias":
            row.append(1.0)
            continue
        section_name, _, field_name = feature_name.partition(".")
        section = getattr(sample, section_name, {})
        if isinstance(section, dict):
            row.append(_coerce_numeric(section.get(field_name)))
        else:
            row.append(0.0)
    return row


def _extract_path_conditions(
    tree_artifact: SurrogateTreeArtifact,
    sample: ExplanationSample,
    feature_vector: np.ndarray,
) -> tuple[RulePathCondition, ...]:
    """Traverse the tree and collect ordered path conditions for one sample."""

    model = _tree_model(tree_artifact)
    tree = getattr(model, "tree_")
    feature_names = tree_artifact.feature_names
    node_index = 0
    path_conditions: list[RulePathCondition] = []

    while tree.children_left[node_index] != tree.children_right[node_index]:
        feature_index = int(tree.feature[node_index])
        threshold = float(tree.threshold[node_index])
        sample_value = float(feature_vector[0, feature_index])
        operator = "<=" if sample_value <= threshold else ">"
        path_conditions.append(
            RulePathCondition(
                feature_name=feature_names[feature_index],
                operator=cast(RuleOperator, operator),
                threshold=threshold,
                sample_value=sample_value,
                tree_node_index=int(node_index),
            )
        )
        node_index = int(
            tree.children_left[node_index]
            if sample_value <= threshold
            else tree.children_right[node_index]
        )
    return tuple(path_conditions)


def extract_rule_for_sample(
    tree_artifact: SurrogateTreeArtifact,
    sample: ExplanationSample,
) -> RuleRecord:
    """Extract one structured rule record for a single explanation sample."""

    _validate_scope(tree_artifact, sample)
    feature_vector = _feature_vector(tree_artifact, sample)
    path_conditions = _extract_path_conditions(tree_artifact, sample, feature_vector)
    model = _tree_model(tree_artifact)
    prediction = _coerce_prediction(model.predict(feature_vector)[0])
    leaf_id = int(model.apply(feature_vector)[0])
    feature_names_used = tuple(
        dict.fromkeys(condition.feature_name for condition in path_conditions)
    )
    rule_id = f"{sample.sample_id}:{tree_artifact.config.mode}:leaf-{leaf_id}"
    return RuleRecord(
        rule_id=rule_id,
        sample_id=sample.sample_id,
        scope=sample.scope,
        tree_mode=tree_artifact.config.mode,
        predicted_score_or_class=prediction,
        leaf_id=leaf_id,
        path_conditions=path_conditions,
        feature_names_used=feature_names_used,
    )


def extract_rules_for_samples(
    tree_artifact: SurrogateTreeArtifact,
    samples: Sequence[ExplanationSample],
) -> list[RuleRecord]:
    """Extract structured rule records for a batch of explanation samples."""

    if not samples:
        return []
    return [extract_rule_for_sample(tree_artifact, sample) for sample in samples]


__all__ = [
    "extract_rule_for_sample",
    "extract_rules_for_samples",
]
