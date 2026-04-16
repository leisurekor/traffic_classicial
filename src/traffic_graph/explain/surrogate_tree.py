"""Train and persist surrogate decision trees from explanation-ready samples."""

from __future__ import annotations

import json
import pickle
import numbers
from math import isfinite
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np

from traffic_graph.explain.explanation_types import ExplanationSample, ExplanationScope
from traffic_graph.explain.surrogate_types import (
    SurrogateTrainingMatrix,
    SurrogateTreeArtifact,
    SurrogateTreeConfig,
    SurrogateTreeSaveResult,
    SurrogateTreeSummary,
    SurrogateTreeMode,
)

_NUMERIC_SUMMARY_SECTIONS: tuple[str, ...] = (
    "stats_summary",
    "graph_summary",
    "feature_summary",
)


def _is_numeric_scalar(value: object) -> bool:
    """Return ``True`` when a value can be treated as a stable numeric feature."""

    return isinstance(
        value,
        (bool, int, float, np.integer, np.floating, np.bool_),
    ) and not isinstance(value, complex)


def _coerce_numeric(value: object) -> float:
    """Convert a supported numeric scalar into a finite float."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, numbers.Real) or isinstance(value, (np.integer, np.floating)):
        converted = float(value)
        return converted if isfinite(converted) else 0.0
    return 0.0


def _extract_feature_space(
    samples: Sequence[ExplanationSample],
) -> tuple[str, ...]:
    """Collect the stable set of numeric feature names across all samples."""

    feature_names: list[str] = []
    seen: set[str] = set()
    for section_name in _NUMERIC_SUMMARY_SECTIONS:
        section_keys: set[str] = set()
        for sample in samples:
            section = getattr(sample, section_name)
            if isinstance(section, dict):
                for key, value in section.items():
                    if _is_numeric_scalar(value):
                        section_keys.add(str(key))
        for key in sorted(section_keys):
            feature_name = f"{section_name}.{key}"
            if feature_name not in seen:
                seen.add(feature_name)
                feature_names.append(feature_name)
    if not feature_names:
        return ("constant_bias",)
    return tuple(feature_names)


def _sample_feature_row(
    sample: ExplanationSample,
    *,
    feature_names: Sequence[str],
) -> list[float]:
    """Build one numeric feature row from an explanation sample."""

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


def _infer_scope(samples: Sequence[ExplanationSample]) -> ExplanationScope:
    """Infer the shared explanation scope from the sample sequence."""

    scope_names = {sample.scope for sample in samples}
    if not scope_names:
        raise ValueError("At least one explanation sample is required.")
    if len(scope_names) != 1:
        raise ValueError("Surrogate tree training expects samples from exactly one scope.")
    scope_name = next(iter(scope_names))
    if scope_name not in {"graph", "flow", "node"}:
        raise ValueError(f"Unsupported explanation scope: {scope_name}")
    return cast(ExplanationScope, scope_name)


def extract_training_matrix(
    samples: Sequence[ExplanationSample],
    config: SurrogateTreeConfig | None = None,
) -> SurrogateTrainingMatrix:
    """Extract a stable numeric matrix from explanation-ready samples."""

    if not samples:
        raise ValueError("At least one explanation sample is required.")
    resolved_config = config or SurrogateTreeConfig()
    sample_scope = _infer_scope(samples)
    feature_names = _extract_feature_space(samples)
    features = np.asarray(
        [_sample_feature_row(sample, feature_names=feature_names) for sample in samples],
        dtype=float,
    )

    if resolved_config.mode == "regression":
        targets = np.asarray([sample.anomaly_score for sample in samples], dtype=float)
        target_name = "anomaly_score"
    else:
        target_values: list[int] = []
        for sample in samples:
            if sample.is_alert is not None:
                target_values.append(1 if sample.is_alert else 0)
                continue
            if sample.threshold is None:
                raise ValueError(
                    "Classification mode requires `is_alert` or `threshold` for every sample."
                )
            target_values.append(1 if sample.anomaly_score >= sample.threshold else 0)
        targets = np.asarray(target_values, dtype=int)
        target_name = "is_alert"

    return SurrogateTrainingMatrix(
        features=features,
        targets=targets,
        feature_names=feature_names,
        sample_ids=tuple(sample.sample_id for sample in samples),
        sample_scope=sample_scope,
        target_name=target_name,
    )


def train_surrogate_tree(
    samples: Sequence[ExplanationSample],
    config: SurrogateTreeConfig | None = None,
) -> SurrogateTreeArtifact:
    """Fit a surrogate decision tree on explanation-ready samples."""

    resolved_config = config or SurrogateTreeConfig()
    matrix = extract_training_matrix(samples, resolved_config)

    if resolved_config.mode == "regression":
        from sklearn.tree import DecisionTreeRegressor

        model: object = DecisionTreeRegressor(
            max_depth=resolved_config.max_depth,
            min_samples_leaf=resolved_config.min_samples_leaf,
            random_state=resolved_config.random_state,
        )
    else:
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            max_depth=resolved_config.max_depth,
            min_samples_leaf=resolved_config.min_samples_leaf,
            random_state=resolved_config.random_state,
        )

    model.fit(matrix.features, matrix.targets)
    tree = getattr(model, "tree_", None)
    summary = SurrogateTreeSummary(
        sample_count=int(matrix.features.shape[0]),
        sample_scope=matrix.sample_scope,
        mode=resolved_config.mode,
        feature_count=int(matrix.features.shape[1]),
        tree_depth=int(tree.max_depth) if tree is not None else 0,
        leaf_count=int(tree.n_leaves) if tree is not None else 0,
        target_name=matrix.target_name,
    )
    return SurrogateTreeArtifact(
        model=model,
        feature_names=matrix.feature_names,
        config=resolved_config,
        summary=summary,
    )


def save_surrogate_tree_artifact(
    artifact: SurrogateTreeArtifact,
    output_dir: str | Path,
) -> SurrogateTreeSaveResult:
    """Persist a surrogate-tree artifact into a directory."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.pkl"
    metadata_path = directory / "metadata.json"

    with model_path.open("wb") as handle:
        pickle.dump(artifact.model, handle)

    metadata_payload = artifact.to_dict()
    metadata_payload["model_path"] = model_path.as_posix()
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return SurrogateTreeSaveResult(
        output_directory=directory.as_posix(),
        model_path=model_path.as_posix(),
        metadata_path=metadata_path.as_posix(),
        artifact_paths={
            "surrogate_model_pickle": model_path.as_posix(),
            "surrogate_metadata_json": metadata_path.as_posix(),
        },
    )


def load_surrogate_tree_artifact(path: str | Path) -> SurrogateTreeArtifact:
    """Load a persisted surrogate-tree artifact from disk."""

    candidate = Path(path)
    directory = candidate if candidate.is_dir() else candidate.parent
    metadata_path = directory / "metadata.json"
    model_path = directory / "model.pkl"
    if candidate.name == "metadata.json":
        metadata_path = candidate
    if candidate.name == "model.pkl":
        model_path = candidate
        metadata_path = candidate.parent / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    config_payload = metadata.get("config", {})
    summary_payload = metadata.get("summary", {})
    config = SurrogateTreeConfig(
        mode=cast(
            SurrogateTreeMode,
            str(config_payload.get("mode", "regression")),
        ),
        max_depth=(
            int(config_payload["max_depth"])
            if config_payload.get("max_depth") is not None
            else None
        ),
        min_samples_leaf=int(config_payload.get("min_samples_leaf", 5)),
        random_state=int(config_payload.get("random_state", 42)),
    )
    summary = SurrogateTreeSummary(
        sample_count=int(summary_payload.get("sample_count", 0)),
        sample_scope=cast(
            ExplanationScope,
            str(summary_payload.get("sample_scope", "flow")),
        ),
        mode=cast(
            SurrogateTreeMode,
            str(summary_payload.get("mode", config.mode)),
        ),
        feature_count=int(summary_payload.get("feature_count", 0)),
        tree_depth=int(summary_payload.get("tree_depth", 0)),
        leaf_count=int(summary_payload.get("leaf_count", 0)),
        target_name=str(summary_payload.get("target_name", "anomaly_score")),
    )
    feature_names_raw = metadata.get("feature_names", ())
    if isinstance(feature_names_raw, Iterable) and not isinstance(
        feature_names_raw,
        (str, bytes, bytearray),
    ):
        feature_names = tuple(str(item) for item in feature_names_raw)
    else:
        feature_names = ()
    return SurrogateTreeArtifact(
        model=model,
        feature_names=feature_names,
        config=config,
        summary=summary,
    )


def summarize_surrogate_tree_artifact(artifact: SurrogateTreeArtifact) -> str:
    """Render a compact human-readable summary for a fitted surrogate tree."""

    return (
        "Surrogate tree: "
        f"mode={artifact.summary.mode}, "
        f"scope={artifact.summary.sample_scope}, "
        f"samples={artifact.summary.sample_count}, "
        f"features={artifact.summary.feature_count}, "
        f"depth={artifact.summary.tree_depth}, "
        f"leaves={artifact.summary.leaf_count}, "
        f"target={artifact.summary.target_name}"
    )


__all__ = [
    "extract_training_matrix",
    "load_surrogate_tree_artifact",
    "save_surrogate_tree_artifact",
    "summarize_surrogate_tree_artifact",
    "train_surrogate_tree",
]
