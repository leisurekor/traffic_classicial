#!/usr/bin/env python3
"""Generate the final constrained experiment report from real repository artifacts."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
REPORT_MD = REPORTS_DIR / "final_experiment_report.md"
REPORT_HTML = REPORTS_DIR / "final_experiment_report.html"
REPORT_ASSETS_DIR = REPORTS_DIR / "assets"

SUMMARY_JSON = REPO_ROOT / "outputs" / "summary" / "all_metrics.json"
CTU13_COMPARISON_CSV = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.csv"
CTU13_COMPARISON_MD = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.md"
CTU13_BENCHMARK_MD = REPO_ROOT / "results" / "ctu13_binary_benchmark.md"
CTU13_ALIGNMENT_MD = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.md"
CTU13_EXTRACTION_MD = REPO_ROOT / "results" / "ctu13_primary_graph_extraction_summary.md"
CTU13_ALIGNMENT_CSV = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.csv"
CTU13_EXTRACTION_CSV = REPO_ROOT / "results" / "ctu13_primary_graph_extraction_summary.csv"
CONDA_PYTHON = REPO_ROOT / ".conda-bench" / "bin" / "python"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _rel(path_text: str | None) -> str:
    if not path_text:
        return ""
    try:
        return os.path.relpath(Path(path_text), REPORTS_DIR)
    except Exception:
        return str(path_text)


def _copy_report_asset(source_path_text: str | None) -> tuple[str, str]:
    if not source_path_text:
        return ("", "")
    source = Path(source_path_text)
    if not source.exists():
        return (_rel(source_path_text), "")
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORT_ASSETS_DIR / source.name
    shutil.copy2(source, target)
    return (_rel(target.as_posix()), source.as_posix())


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_pm(mean_value: Any, std_value: Any, digits: int = 4) -> str:
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return _fmt(mean_value, digits)
    return f"{float(mean_value):.{digits}f} ± {float(std_value):.{digits}f}"


def _load_merge01_row() -> dict[str, Any]:
    rows = _load_json(SUMMARY_JSON)
    if not isinstance(rows, list):
        raise ValueError("outputs/summary/all_metrics.json is not a list.")
    for row in rows:
        if row.get("experiment_name") == "CICIoT_csv_merge_a":
            return row
    raise ValueError("Could not find CICIoT_csv_merge_a in outputs/summary/all_metrics.json.")


def _load_ctu13_rows() -> list[dict[str, str]]:
    with CTU13_COMPARISON_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_order = ["48", "49", "52", "merged_48_49_52"]
    selected = [row for row in rows if row.get("scenario_id") in target_order]
    selected.sort(key=lambda row: target_order.index(str(row["scenario_id"])))
    if not selected:
        raise ValueError("Could not find CTU13 48/49/52 rows in results/ctu13_edge_centric_comparison.csv.")
    return selected


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _collect_environment() -> dict[str, str]:
    env = {
        "os": platform.platform(),
        "python": sys.version.split("\n", maxsplit=1)[0],
        "numpy": "unavailable",
        "pandas": "unavailable",
        "sklearn": "unavailable",
        "matplotlib": "unavailable",
        "torch": "unavailable",
        "cuda": "CPU only / CUDA unavailable",
    }
    probe_code = """
import importlib
import json
import sys

payload = {"python": sys.version.split("\\n", 1)[0]}
for name in ["torch", "numpy", "pandas", "sklearn", "matplotlib"]:
    try:
        module = importlib.import_module(name)
        payload[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[name] = f"MISSING:{type(exc).__name__}"
print(json.dumps(payload, ensure_ascii=False))
""".strip()
    py_bin = CONDA_PYTHON if CONDA_PYTHON.exists() else Path(sys.executable)
    completed = subprocess.run(
        [py_bin.as_posix(), "-c", probe_code],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        payload = json.loads(completed.stdout.strip())
        env["python"] = str(payload.get("python", env["python"]))
        env["torch"] = str(payload.get("torch", env["torch"]))
        env["numpy"] = str(payload.get("numpy", env["numpy"]))
        env["pandas"] = str(payload.get("pandas", env["pandas"]))
        env["sklearn"] = str(payload.get("sklearn", env["sklearn"]))
        env["matplotlib"] = str(payload.get("matplotlib", env["matplotlib"]))
    return env


def _extract_merge01_seed_overview(merge_row: dict[str, Any]) -> str:
    seed_runs = merge_row.get("seed_runs", [])
    if not isinstance(seed_runs, list):
        return ""
    lines = [
        "| 随机种子 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed_run in seed_runs:
        metrics = seed_run.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed_run.get("seed", "")),
                    _fmt(metrics.get("accuracy")),
                    _fmt(metrics.get("precision")),
                    _fmt(metrics.get("recall")),
                    _fmt(metrics.get("f1")),
                    _fmt(metrics.get("roc_auc")),
                    _fmt(metrics.get("pr_auc")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _extract_key_csv_configuration(merge_metric_path: Path | None) -> list[str]:
    if merge_metric_path is None or not merge_metric_path.exists():
        return []
    payload = _load_json(merge_metric_path)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    prepared = payload.get("prepared_summary", {}) if isinstance(payload, dict) else {}
    lines = []
    if config:
        lines.extend(
            [
                f"- 阈值分位数（threshold percentile）：`{config.get('threshold_percentile', 'N/A')}`",
                f"- 学习率（learning rate）：`{config.get('learning_rate', 'N/A')}`",
                f"- 批大小（batch size）：`{config.get('batch_size', 'N/A')}`",
                f"- 统一配置中的训练轮数（configured epochs）：`{config.get('epochs', 'N/A')}`",
                f"- 图分数约简策略（graph score reduction）：`{config.get('graph_score_reduction', 'N/A')}`",
                f"- 数据划分：`train={config.get('train_ratio', 'N/A')}` / `val={config.get('val_ratio', 'N/A')}` / `test={config.get('test_ratio', 'N/A')}`",
            ]
        )
    if prepared:
        lines.extend(
            [
                f"- 输入样本数（rows after cleaning）：`{prepared.get('row_count', 'N/A')}`",
                f"- 特征维数（feature dimension）：`{prepared.get('feature_count', 'N/A')}`",
                f"- 训练集 benign-only 约束：`{prepared.get('benign_train_only', 'N/A')}`",
            ]
        )
    return lines


def _ctu13_results_table(ctu13_rows: list[dict[str, str]]) -> str:
    lines = [
        "| 场景 | baseline F1 | graph benchmark F1 | baseline Recall | graph benchmark Recall | baseline FPR | graph benchmark FPR | baseline background hit ratio | graph benchmark background hit ratio | 使用的 graph benchmark profile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ctu13_rows:
        scenario = row["scenario_id"]
        label = "merged 48/49/52" if scenario == "merged_48_49_52" else f"scenario {scenario}"
        profile = f"{row['edge_profile']} + {row['edge_support_summary_mode']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(row.get("baseline_f1")),
                    _fmt(row.get("edge_v2_f1")),
                    _fmt(row.get("baseline_recall")),
                    _fmt(row.get("edge_v2_recall")),
                    _fmt(row.get("baseline_fpr")),
                    _fmt(row.get("edge_v2_fpr")),
                    _fmt(row.get("baseline_background_hit_ratio")),
                    _fmt(row.get("edge_v2_background_hit_ratio")),
                    profile,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _generate_merge01_summary_chart(merge_row: dict[str, Any]) -> str:
    if plt is None:
        return ""
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_ASSETS_DIR / "merge01_metric_summary.png"
    metrics = [
        ("Accuracy", float(merge_row["accuracy_mean"]), float(merge_row["accuracy_std"])),
        ("Precision", float(merge_row["precision_mean"]), float(merge_row["precision_std"])),
        ("Recall", float(merge_row["recall_mean"]), float(merge_row["recall_std"])),
        ("F1", float(merge_row["f1_mean"]), float(merge_row["f1_std"])),
        ("Macro-F1", float(merge_row["macro_f1_mean"]), float(merge_row["macro_f1_std"])),
        ("Balanced Acc.", float(merge_row["balanced_accuracy_mean"]), float(merge_row["balanced_accuracy_std"])),
        ("ROC-AUC", float(merge_row["roc_auc_mean"]), float(merge_row["roc_auc_std"])),
        ("PR-AUC", float(merge_row["pr_auc_mean"]), float(merge_row["pr_auc_std"])),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]
    errors = [item[2] for item in metrics]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    colors = ["#2F5597", "#4472C4", "#5B9BD5", "#70AD47", "#A5A5A5", "#FFC000", "#ED7D31", "#C55A11"]
    ax.bar(labels, values, yerr=errors, color=colors, capsize=4, edgecolor="#2b2b2b", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("CICIoT2023 merge01.csv metric summary", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.02, 1.03), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return _rel(out_path.as_posix())


def _generate_ctu13_benchmark_chart(ctu13_rows: list[dict[str, str]]) -> str:
    if plt is None:
        return ""
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_ASSETS_DIR / "ctu13_graph_benchmark_summary.png"
    scenario_labels = ["48", "49", "52", "合并"]
    scenario_labels = ["48", "49", "52", "Merged"]
    baseline_f1 = [float(row["baseline_f1"]) for row in ctu13_rows]
    edge_f1 = [float(row["edge_v2_f1"]) for row in ctu13_rows]
    baseline_recall = [float(row["baseline_recall"]) for row in ctu13_rows]
    edge_recall = [float(row["edge_v2_recall"]) for row in ctu13_rows]
    baseline_bg = [float(row["baseline_background_hit_ratio"]) for row in ctu13_rows]
    edge_bg = [float(row["edge_v2_background_hit_ratio"]) for row in ctu13_rows]
    x = list(range(len(scenario_labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    width = 0.36
    axes[0].bar([v - width / 2 for v in x], baseline_f1, width=width, color="#A5A5A5", label="baseline F1")
    axes[0].bar([v + width / 2 for v in x], edge_f1, width=width, color="#4472C4", label="graph benchmark F1")
    axes[0].plot(x, edge_recall, marker="o", color="#C00000", linewidth=2, label="graph benchmark Recall")
    axes[0].set_xticks(x, scenario_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("CTU13 primary metrics", fontweight="bold")
    axes[0].set_ylabel("Metric value")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].bar([v - width / 2 for v in x], baseline_bg, width=width, color="#BFBFBF", label="baseline background hit ratio")
    axes[1].bar([v + width / 2 for v in x], edge_bg, width=width, color="#ED7D31", label="graph benchmark background hit ratio")
    axes[1].set_xticks(x, scenario_labels)
    axes[1].set_ylim(0, max(edge_bg + baseline_bg) * 1.2)
    axes[1].set_title("CTU13 background hit ratio", fontweight="bold")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("CTU13 48/49/52 PCAP-based graph benchmark", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return _rel(out_path.as_posix())


def _generate_ctu13_pipeline_chart() -> str:
    if plt is None or not CTU13_ALIGNMENT_CSV.exists() or not CTU13_EXTRACTION_CSV.exists():
        return ""
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_ASSETS_DIR / "ctu13_pipeline_distribution.png"
    alignment_rows = _load_csv_rows(CTU13_ALIGNMENT_CSV)
    extraction_rows = _load_csv_rows(CTU13_EXTRACTION_CSV)
    scenario_labels = [row["scenario_id"] for row in alignment_rows]
    benign = [int(row["benign_count"]) for row in alignment_rows]
    malicious = [int(row["malicious_count"]) for row in alignment_rows]
    unknown = [int(row["unknown_count"]) for row in alignment_rows]
    graph_benign = [int(row["benign_graph_count"]) for row in extraction_rows]
    graph_malicious = [int(row["malicious_graph_count"]) for row in extraction_rows]
    graph_unknown = [int(row["unknown_heavy_graph_count"]) for row in extraction_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    axes[0].bar(scenario_labels, benign, color="#70AD47", label="benign flows")
    axes[0].bar(scenario_labels, malicious, bottom=benign, color="#4472C4", label="malicious flows")
    axes[0].bar(
        scenario_labels,
        unknown,
        bottom=[b + m for b, m in zip(benign, malicious)],
        color="#A5A5A5",
        label="unknown/background flows",
    )
    axes[0].set_title("Flow label alignment distribution", fontweight="bold")
    axes[0].set_ylabel("Flow count")
    axes[0].legend(fontsize=8)

    axes[1].bar(scenario_labels, graph_benign, color="#70AD47", label="benign graphs")
    axes[1].bar(scenario_labels, graph_malicious, bottom=graph_benign, color="#4472C4", label="malicious graphs")
    axes[1].bar(
        scenario_labels,
        graph_unknown,
        bottom=[b + m for b, m in zip(graph_benign, graph_malicious)],
        color="#A5A5A5",
        label="unknown-heavy graphs",
    )
    axes[1].set_title("Primary graph extraction distribution", fontweight="bold")
    axes[1].set_ylabel("Graph count")
    axes[1].legend(fontsize=8)

    fig.suptitle("CTU13 graph benchmark data pipeline statistics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return _rel(out_path.as_posix())


def _markdown_to_html(markdown_text: str, *, title: str) -> str:
    lines = markdown_text.splitlines()
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='utf-8'/>",
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,'Noto Sans CJK SC','Microsoft YaHei',sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;line-height:1.7;color:#222;}",
        "table{border-collapse:collapse;width:100%;margin:16px 0;}",
        "th,td{border:1px solid #ccc;padding:8px 10px;vertical-align:top;}",
        "th{background:#f5f5f5;}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:4px;}",
        "img{max-width:100%;height:auto;margin:12px 0;border:1px solid #ddd;}",
        "p{margin:10px 0;}",
        "ul,ol{margin:10px 0 10px 24px;}",
        "</style>",
        "</head>",
        "<body>",
    ]

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{escape(line[2:].strip())}</h1>")
            index += 1
            continue
        if line.startswith("## "):
            html_lines.append(f"<h2>{escape(line[3:].strip())}</h2>")
            index += 1
            continue
        if line.startswith("### "):
            html_lines.append(f"<h3>{escape(line[4:].strip())}</h3>")
            index += 1
            continue
        if re.match(r"^\d+\.\s", line):
            html_lines.append("<ol>")
            while index < len(lines) and re.match(r"^\d+\.\s", lines[index]):
                item = re.sub(r"^\d+\.\s*", "", lines[index]).strip()
                html_lines.append(f"<li>{escape(item)}</li>")
                index += 1
            html_lines.append("</ol>")
            continue
        if line.startswith("- "):
            html_lines.append("<ul>")
            while index < len(lines) and lines[index].startswith("- "):
                item = lines[index][2:].strip()
                html_lines.append(f"<li>{escape(item)}</li>")
                index += 1
            html_lines.append("</ul>")
            continue
        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            if len(table_lines) >= 2:
                headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                html_lines.append("<table><thead><tr>")
                for cell in headers:
                    html_lines.append(f"<th>{escape(cell)}</th>")
                html_lines.append("</tr></thead><tbody>")
                for body_line in table_lines[2:]:
                    cells = [cell.strip() for cell in body_line.strip("|").split("|")]
                    html_lines.append("<tr>")
                    for cell in cells:
                        html_lines.append(f"<td>{escape(cell)}</td>")
                    html_lines.append("</tr>")
                html_lines.append("</tbody></table>")
            continue
        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", line.strip())
        if image_match:
            alt, src = image_match.groups()
            html_lines.append(f"<p><img alt='{escape(alt)}' src='{escape(src)}'/></p>")
            index += 1
            continue
        html_lines.append(f"<p>{escape(line)}</p>")
        index += 1

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines) + "\n"


def build_report_markdown() -> str:
    merge_row = _load_merge01_row()
    ctu13_rows = _load_ctu13_rows()
    env = _collect_environment()

    merge_log = Path(merge_row["seed_runs"][0]["log_path"]) if merge_row.get("seed_runs") else None
    merge_metric_path = Path(merge_row["seed_runs"][0]["metrics_path"]) if merge_row.get("seed_runs") else None

    merge_confusion, merge_confusion_source = _copy_report_asset(merge_row.get("confusion_matrix_png"))
    merge_roc, merge_roc_source = _copy_report_asset(merge_row.get("roc_curve_png"))
    merge_pr, merge_pr_source = _copy_report_asset(merge_row.get("pr_curve_png"))
    merge_summary_chart = _generate_merge01_summary_chart(merge_row)
    ctu13_benchmark_chart = _generate_ctu13_benchmark_chart(ctu13_rows)
    ctu13_pipeline_chart = _generate_ctu13_pipeline_chart()
    csv_config_lines = _extract_key_csv_configuration(merge_metric_path)

    lines = [
        "# 最终实验报告",
        "",
        "## 1. 实验目标",
        "本项目当前阶段的目标，是在**不引入 nuisance-aware 扩展**的前提下，对现有恶意流量检测系统给出一份收口版、可追溯、可复核的正式实验报告。依据仓库中现有的真实结果文件，最终正式纳入报告的实验结果仅包含两部分：其一为 `CICIoT2023` 的 `merge01.csv` 主实验结果；其二为 `CTU13` 场景 `48/49/52` 的图侧主 benchmark 结果。",
        "选择这两部分结果的原因在于，它们同时满足三项要求：第一，存在真实的日志、指标表与图表；第二，能够共同覆盖“标准化表格输入”和“原始 PCAP 下游图建模输入”两类代表性实验场景；第三，不依赖当前尚未系统完成的跨数据集完整 PCAP 主实验矩阵。因此，本报告主动限制结论边界，不再将其他未完全收口的数据路线纳入最终正式结论。",
        "本报告统一采用 `use_nuisance_aware=false`。在这一设置下，本阶段要回答的核心问题不是复杂背景流量（background / unknown）的最终抑制是否已经彻底解决，而是：**当前主模型在标准流特征输入上的分类能力是否已经被可靠验证；以及从原始 PCAP 到流图建模再到检测的图侧链路是否已经被可信地证明可行。**",
        "由此，本报告希望回答的核心问题可归纳为两点：第一，`merge01.csv` 是否足以作为当前 CSV 主结果，用以证明分类器在标准流特征表格输入上的判别能力；第二，`CTU13 48/49/52` 是否可以作为图侧主 benchmark，并被准确表述为 `PCAP-based graph benchmark`，而不被误写成“完整跨数据集 PCAP 主实验体系已经完成”。",
        "",
        "## 2. 实验设计",
        "当前正式纳入的实验设计由两条互补实验线构成。第一条是 **CSV 主实验线**，其角色是验证检测器在标准化、表格化流特征输入上的分类能力；第二条是 **图实验线（graph benchmark）**，其角色是验证从原始 PCAP 出发，经由流构建（flow construction）、图构建（graph construction）、特征提取（feature extraction）与图级打分（graph-level scoring）之后，检测链路能否稳定工作。",
        "`merge01.csv` 被选为 CSV 主结果，是因为 `outputs/summary/all_metrics.json` 中存在完整的三随机种子运行记录，且其日志、配置与图表均可直接对齐到 `artifacts/cic_iot2023/Merged01.csv`。这使得该结果具备较高的可审计性（auditability）和稳定性（stability），适合作为当前阶段最可信的表格主结果。",
        "`CTU13 48/49/52` 被选为图侧 benchmark，是因为 `results/ctu13_binary_benchmark.*` 与 `results/ctu13_edge_centric_comparison.*` 提供了明确的 scenario-wise 以及 merged benchmark 结果，而 `results/ctu13_flow_label_alignment_summary.*` 与 `results/ctu13_primary_graph_extraction_summary.*` 则为其上游数据流转提供了直接旁证。因而，这部分结果可以严格、准确地写成 `PCAP-based graph benchmark` 或 `PCAP-derived graph benchmark`。",
        "### 2.1 任务定义",
        "在正式报告口径中，当前任务仍然是**二分类任务（binary classification）**：benign / normal 映射为 `0`，malicious / attack / botnet 映射为 `1`。对 `CTU13` 而言，依据 `results/ctu13_binary_benchmark.md` 可确认：primary metrics 只在 benign 与 malicious 上计算，而 background / unknown 只作为附加分析（secondary analysis）报告。因此，本报告的正式结论仍然建立在二分类任务之上。",
        "### 2.2 方法原理",
        "就方法原理而言，项目当前主线是一个**无监督（unsupervised）恶意流量检测框架**。其基本思想是：使用 benign 数据建立正常行为参考分布，再以图结构或流特征描述网络交互模式，并通过异常得分（anomaly score）度量待测样本与正常模式之间的偏离程度。对于 CSV 主实验线，这种偏离体现在结构化流特征的统计分布上；对于图实验线，这种偏离体现在由端点交互与时间窗口构成的图结构模式上。",
        "### 2.3 项目总体设计",
        "从系统设计角度看，本项目采用“**输入层—表示层—判别层**”三层结构。输入层负责接收表格化流特征或原始 PCAP 下游流记录；表示层负责将输入映射为数值特征向量或交互图；判别层则输出 graph-level 或 sample-level 的异常分数并据此进行二分类判定。CSV 主实验与 CTU13 图侧 benchmark 使用的是同一研究问题下的两种观测视角：前者强调分类器在标准输入上的判别上限，后者强调端到端图建模链路的工程可行性与检测有效性。",
        "本报告中的 nuisance-aware 状态固定为关闭，即 `use_nuisance_aware=false`。虽然仓库中保留了 nuisance-aware 相关的历史比较产物，但它们不属于本报告最终正式采纳的结果集合。正式结论只围绕当前 single-stage 主线及其非 nuisance-aware 版本展开。",
        "",
        "## 3. 数据集说明",
        "### 3.1 CICIoT2023 / merge01.csv",
        "`merge01.csv` 对应的真实输入路径为 `/home/xdo/traffic_classicial/artifacts/cic_iot2023/Merged01.csv`，其结果由 `outputs/summary/all_metrics.json` 中的 `CICIoT_csv_merge_a` 条目直接对应。依据 `outputs/metrics/CICIoT_csv_merge_a_seed42-20260417T120010Z.json`，该实验的数据形式是带显式 `Label` 列的表格化流特征输入，并包含固定的一组数值特征列，如 `Header_Length`、`Protocol Type`、`Time_To_Live`、`Rate` 以及若干 TCP flag、计数统计与时序统计特征。",
        "在本项目中，`merge01.csv` 的主要作用，是提供一条工程噪声较低、输入边界清晰的实验线，用于证明检测器在**标准流特征表格输入**上的判别能力。由于其输入已经是结构化表格，因此该实验较少受到 PCAP 解析、会话重建和图构建等上游工程细节的干扰，更适合作为“分类能力验证”的主结果。",
        "",
        "### 3.2 CTU13 场景 48 / 49 / 52",
        "`CTU13` 场景 `48/49/52` 的正式结果来自 `results/ctu13_binary_benchmark.*` 和 `results/ctu13_edge_centric_comparison.*`。依据这些文件以及 `results/ctu13_flow_label_alignment_summary.*`、`results/ctu13_primary_graph_extraction_summary.*`，这部分结果的来源可以描述为：从原始 CTU13 场景数据出发，经过流级标签对齐（flow label alignment）、图提取（graph extraction）、特征打包（feature packing）和图级打分（graph-level scoring）后形成的 benchmark 结果。",
        "因此，这部分结果必须准确写成 `PCAP-based graph benchmark`，或者写成“基于原始 PCAP 构建流图后的 benchmark”。其链路应表述为：`PCAP -> flow -> graph -> feature -> scorer/classifier`。其中，`results/ctu13_flow_label_alignment_summary.md` 给出了 flow-level 对齐摘要，而 `results/ctu13_primary_graph_extraction_summary.md` 给出了进入图级评估前的候选图统计。",
        "需要特别强调的是：尽管这部分结果来自原始 PCAP 的下游图建模链路，但它并不等于“完整多数据集 PCAP 主实验体系已经完成”。本报告不会把这部分结果扩展解释为多数据集、跨数据集的完整 PCAP 实验矩阵，也不会据此声称 `CICIDS2017 PCAP` 已经系统验证完成。",
        "",
        "## 4. 实验流程",
        "### 4.1 CSV 实验流程",
        "依据 `outputs/logs/CICIoT_csv_merge_a_seed42-20260417T120010Z.log` 和对应 metrics 文件可确认，CSV 实验流程包括：读取 `Merged01.csv`；执行 NaN/Inf 清洗；按显式标签映射将 benign 类映射为 `0`、攻击类映射为 `1`；再执行训练集（train）/验证集（validation）/测试集（test）划分并训练现有二分类检测主线。日志明确记录了清洗后的样本量、替换的 NaN/Inf 数量，以及 train/val/test 的样本规模和类别分布。",
        "从现有日志可确认到的范围看，该实验并未启用 nuisance-aware，且训练阶段只使用 benign 样本。这一设置与项目的无监督检测思路一致，即以 benign 数据建立参考分布，再通过异常分数完成检测。日志中记录的 `Training epochs=0` 表明该 CSV 主线采用的是当前稳定表格路径，其训练接口与图模型的多 epoch 训练接口并不完全同义；而 metrics 配置文件中仍保留了统一配置面的 `epochs=2`、`batch_size=2`、`learning_rate=1e-3` 等字段。",
        "评估阶段基于真实导出的 `overall_scores.csv` 重新汇总 Accuracy（准确率）、Precision（精确率）、Recall（召回率）、F1、Macro-F1（宏平均 F1）、Balanced Accuracy（平衡准确率）、ROC-AUC 与 PR-AUC，并从同一批 score 文件生成混淆矩阵（confusion matrix）、ROC 曲线与 PR 曲线。因此，CSV 主结果同时具备可量化的指标表与可视化图表。",
        "",
        "### 4.2 CTU13 图实验流程",
        "对于 CTU13 图侧 benchmark，依据 `results/ctu13_flow_label_alignment_summary.md`、`results/ctu13_primary_graph_extraction_summary.md`、`results/ctu13_binary_benchmark.md` 与 `results/ctu13_edge_centric_comparison.md` 可以确认，其流程至少包括以下环节：首先从原始场景数据中构建流级对齐结果；随后在固定窗口（window）和固定图分组策略（graph grouping policy）下构建候选图；然后对图进行特征提取与图级打分；最后输出 scenario-wise 与 merged benchmark 指标。",
        "就系统实现含义而言，这条链路解决的是一个比 CSV 更复杂的问题：它不仅要区分 benign 与 malicious，还要把原始 PCAP 中的离散包序列转化为可建模的流，再把流组织为具有局部拓扑关系与时间关系的交互图。也正因为如此，CTU13 图实验的结论重点并不在于“它是否代表完整 PCAP 实验矩阵”，而在于“原始 PCAP 下游的图建模链路是否已经被真实实验结果证明可行”。",
        "现有结果文件能够明确证明这是一条从原始 PCAP 下游派生出来的图 benchmark 链路，但无法像 CSV 日志那样逐条恢复每一步的完整运行日志。因此，本报告对 CTU13 图实验流程的描述限定在“依据现有日志、配置与结果文件可确认到的范围”之内。在该范围内，可以确认它是 graph benchmark，而不能确认它已经扩展成一套完整的跨数据集 PCAP 主实验体系。",
        "",
        "## 5. 实验环境",
        f"- 操作系统：`{env['os']}`",
        f"- Python 版本：`{env['python']}`",
        f"- NumPy：`{env['numpy']}`",
        f"- pandas：`{env['pandas']}`",
        f"- scikit-learn：`{env['sklearn']}`",
        f"- matplotlib：`{env['matplotlib']}`",
        f"- PyTorch：`{env['torch']}`",
        f"- CUDA / GPU：`{env['cuda']}`",
        "- nuisance-aware：关闭，即 `use_nuisance_aware=false`。",
        "- 随机种子：CSV 主结果明确使用 `42`、`43`、`44`；CTU13 图 benchmark 的结果文件是 scenario-wise 与 merged benchmark 汇总表，未在最终比较表中逐条展开随机种子维度。",
        "- 可确认的 CSV 主结果配置：`threshold_percentile=95.0`、`random_seed in {42,43,44}`、`batch_size=2`、`epochs=2`、`learning_rate=1e-3`、`weight_decay=0.0`、`window_size=60`。这些字段来自 `outputs/metrics/CICIoT_csv_merge_a_seed42-20260417T120010Z.json`。",
        "- 可确认的 CTU13 图 benchmark 配置：`graph_score_reduction`、`support_summary_mode`、`evaluation_mode`、`scenario_id`、`background_hit_ratio` 等信息存在于 `results/ctu13_binary_benchmark.*` 与 `results/ctu13_edge_centric_comparison.*`。但这些最终结果表并未完整保存底层依赖版本、batch size、epoch 等全部训练环境细节，因此本报告不对这些缺失项作任何编造。",
        "",
        "## 6. 实验结果",
        "### 6.1 merge01.csv 的实验结果",
        "正式纳入的 CSV 主结果是 `CICIoT_csv_merge_a`，对应输入文件 `Merged01.csv`。其三随机种子汇总结果如下。为提高可读性，图 1 进一步给出了主要指标的可视化概览。",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| Accuracy | {_fmt_pm(merge_row.get('accuracy_mean'), merge_row.get('accuracy_std'))} |",
        f"| Precision | {_fmt_pm(merge_row.get('precision_mean'), merge_row.get('precision_std'))} |",
        f"| Recall | {_fmt_pm(merge_row.get('recall_mean'), merge_row.get('recall_std'))} |",
        f"| F1 | {_fmt_pm(merge_row.get('f1_mean'), merge_row.get('f1_std'))} |",
        f"| Macro-F1 | {_fmt_pm(merge_row.get('macro_f1_mean'), merge_row.get('macro_f1_std'))} |",
        f"| Balanced Accuracy | {_fmt_pm(merge_row.get('balanced_accuracy_mean'), merge_row.get('balanced_accuracy_std'))} |",
        f"| ROC-AUC | {_fmt_pm(merge_row.get('roc_auc_mean'), merge_row.get('roc_auc_std'))} |",
        f"| PR-AUC | {_fmt_pm(merge_row.get('pr_auc_mean'), merge_row.get('pr_auc_std'))} |",
        "",
        "图 1 显示，`merge01.csv` 主实验在 F1、ROC-AUC 和 PR-AUC 三个指标上均处于较高水平，且误差条（error bar）很短，说明三随机种子之间波动较小。",
        "",
        f"![merge01 summary chart]({merge_summary_chart})" if merge_summary_chart else "（当前运行环境未生成额外汇总图；原因是 matplotlib 不可用。）",
        "",
        "对应的单种子结果如下：",
        "",
        _extract_merge01_seed_overview(merge_row),
        "",
        "从真实配置文件中可进一步确认，`merge01.csv` 主实验采用了如下关键设置：",
        "",
        *csv_config_lines,
        "",
        "CSV 主结果对应的真实图表路径如下：",
        f"- confusion matrix：报告内嵌副本 `reports/assets/{Path(merge_confusion).name}`；原始来源 `{merge_confusion_source or merge_row.get('confusion_matrix_png')}`",
        f"- ROC curve：报告内嵌副本 `reports/assets/{Path(merge_roc).name}`；原始来源 `{merge_roc_source or merge_row.get('roc_curve_png')}`",
        f"- PR curve：报告内嵌副本 `reports/assets/{Path(merge_pr).name}`；原始来源 `{merge_pr_source or merge_row.get('pr_curve_png')}`",
        "",
        f"![merge01 confusion]({merge_confusion})",
        f"![merge01 ROC]({merge_roc})",
        f"![merge01 PR]({merge_pr})",
        "",
        "### 6.2 CTU13 48/49/52 的图侧 benchmark 结果",
        "正式纳入的图侧结果来自 `results/ctu13_edge_centric_comparison.csv` 中的 scenario-wise 与 merged 条目。根据当前正式口径，本报告只采用**非 nuisance-aware** 的图侧主 benchmark，即表中的 `edge_v2` 列，不把 nuisance-aware 结果纳入最终正式结论。图 2 概览了各场景下 baseline 与 graph benchmark 在主指标和背景命中率上的差异。",
        "",
        _ctu13_results_table(ctu13_rows),
        "",
        f"![ctu13 benchmark summary]({ctu13_benchmark_chart})" if ctu13_benchmark_chart else "（当前运行环境未生成 CTU13 汇总图；原因是 matplotlib 不可用。）",
        "",
        "作为图侧 benchmark 的支持性结果文件如下：",
        f"- `results/ctu13_edge_centric_comparison.md`：`{_rel(CTU13_COMPARISON_MD.as_posix())}`",
        f"- `results/ctu13_binary_benchmark.md`：`{_rel(CTU13_BENCHMARK_MD.as_posix())}`",
        f"- `results/ctu13_flow_label_alignment_summary.md`：`{_rel(CTU13_ALIGNMENT_MD.as_posix())}`",
        f"- `results/ctu13_primary_graph_extraction_summary.md`：`{_rel(CTU13_EXTRACTION_MD.as_posix())}`",
        "",
        "从结果解释角度看，CTU13 的 benchmark 不能只看最终 F1 与 Recall，还必须结合上游对齐统计与图提取统计一起理解。图 3 展示了 flow label alignment 与 primary graph extraction 后的样本组成变化。",
        "",
        f"![ctu13 pipeline chart]({ctu13_pipeline_chart})" if ctu13_pipeline_chart else "（当前运行环境未生成 CTU13 流转统计图；原因是 matplotlib 不可用。）",
        "",
        "其中，flow-level 对齐摘要为：",
        "",
        _read_text(CTU13_ALIGNMENT_MD).strip(),
        "",
        "进入 primary graph extraction 前的图统计为：",
        "",
        _read_text(CTU13_EXTRACTION_MD).strip(),
        "",
        "在当前正式采纳的结果集合中，没有发现与 CTU13 48/49/52 graph benchmark 一一对应的独立 PNG 曲线图文件；因此本报告对 CTU13 部分只引用真实存在的 benchmark 表和摘要文件，不补造任何额外图表。",
        "",
        "## 7. 结果分析",
        "### 7.1 merge01.csv 结果的含义",
        "`merge01.csv` 的结果主要证明的是：在标准化、表格化的流特征输入上，当前分类器具备很强的二分类能力。其 F1、ROC-AUC 与 PR-AUC 都处于较高水平，且三随机种子的波动很小。这说明，只要输入已经被规整为稳定的流特征表，现有模型在 benign 与 malicious 的区分上是可靠的。换言之，`merge01.csv` 更接近“模型判别能力上限”的观测窗口，而不是“完整链路工程复杂性”的观测窗口。",
        "### 7.2 CTU13 图侧 benchmark 结果的含义",
        "`CTU13 48/49/52` 的结果主要证明的是：从原始 PCAP 出发，经由 flow 构建、graph 构建、feature 提取，再到 graph-level scorer/classifier 的链路是可行的，而且在主 benchmark 上能够取得明显优于 node-centric baseline 的结果。依据 `results/ctu13_edge_centric_comparison.csv`，merged 48/49/52 上图主线的 F1 达到 `0.9412`，Recall 达到 `0.9412`，在 primary 指标上明显优于 baseline。",
        "### 7.3 为什么 CTU13 可以称为 PCAP-based graph benchmark",
        "之所以可以把 CTU13 结果称为 `PCAP-based graph benchmark`，是因为这套结果并不是直接从表格 CSV 得到的，而是来自原始 PCAP 下游的图建模链路。`ctu13_flow_label_alignment_summary` 与 `ctu13_primary_graph_extraction_summary` 说明了 flow 对齐和 graph extraction 过程的存在，`ctu13_binary_benchmark` 与 `ctu13_edge_centric_comparison` 则给出了最终的图级评估结果。换言之，这是一套以原始 PCAP 为源头、以图结构建模为中间表示、以 graph-level detection 为最终输出的 benchmark。",
        "### 7.4 为什么 CTU13 不能被写成完整 PCAP 主实验体系",
        "CTU13 的这套结果不能直接写成“完整 PCAP 主实验体系已经完成”。原因在于：第一，本报告没有纳入 `CICIDS2017 PCAP` 等其他目标数据集的系统结果；第二，当前正式结论只覆盖 CTU13 这一图 benchmark 数据来源；第三，CTU13 的这套结果虽然是 PCAP-based graph benchmark，但并不等于已经形成了一套跨数据集、跨模态、可全面横向对比的完整 PCAP 主实验矩阵。",
        "### 7.5 nuisance-aware 关闭后的结论边界",
        "在 nuisance-aware 关闭的情况下，当前主结论仍然成立。对 CSV 主结果而言，`use_nuisance_aware=false` 并未妨碍模型在 `merge01.csv` 上取得稳定高分；对 CTU13 图 benchmark 而言，正式纳入的 `edge_temporal_binary_v2` 非 nuisance-aware 结果已经足以构成图侧主 benchmark。因此，nuisance-aware 在当前最终报告中不是必要项。不过，这一结论只说明“在当前正式结果范围内不是必要项”，并不意味着 nuisance-aware 对更复杂背景流量场景永远没有研究价值。",
        "",
        "## 8. 局限性",
        "当前正式结果只覆盖 `merge01.csv` 与 `CTU13 48/49/52`。这意味着，本报告的结论范围是受限的：它能够支持“标准表格二分类主结果已经成立”和“存在一套 PCAP-based graph benchmark 主结果”，但不能支持“完整多数据集 PCAP 主实验矩阵已经建立”。",
        "本报告没有纳入 `CICIDS2017 PCAP` 的系统结果，也没有纳入一套完整跨数据集的 packet-side 对照结果。因此，不能声称已经完成了完整跨数据集 PCAP 对照验证，更不能把 CTU13 这套 graph benchmark 推广为“多数据集完整 PCAP 体系已经全部验证完成”。",
        "对 CTU13 而言，现有结果还显示出一个重要事实：在 flow 对齐与 primary graph extraction 后，unknown-heavy flows / graphs 占据了明显主导比例。这意味着图侧主 benchmark 虽然已经足以证明链路可行，但并不能单凭当前结果推导出“复杂背景流量问题已被完全解决”。",
        "此外，仓库中虽然存在更多历史结果文件、补充实验产物以及 nuisance-aware 相关比较行，但它们并不都满足“稳定、正式、当前最终采纳”的标准。因此，这些重路线或补充实验没有被纳入最终正式结论，其存在只能作为历史痕迹，而不能扩大本报告的结论边界。",
        "",
        "## 9. 结论",
        "第一，`merge01.csv` 足以支撑当前阶段的 CSV 主结论。基于真实的三随机种子运行结果，可以明确得出：现有模型在标准流特征表格输入上具有较强且稳定的二分类能力。",
        "第二，`CTU13 48/49/52` 可以作为图侧主 benchmark，而且应当被准确写成 `PCAP-based graph benchmark` 或“基于原始 PCAP 构建流图后的 benchmark”。这部分结果能够证明原始 PCAP 下游图建模链路是可行的，并且在主 benchmark 上取得了有竞争力的结果。",
        "第三，nuisance-aware 在当前最终结果里不是必要项。本报告的正式结果全部建立在 `use_nuisance_aware=false` 的前提上，且仍然可以形成清晰、可提交的主结论。",
        "第四，当前并没有完成完整 PCAP 主实验体系。更准确的表述应当是：本项目**已有 CTU13 的 PCAP-based graph benchmark 结果**，但**尚未形成完整的跨数据集 PCAP 主实验矩阵**。",
        "",
        "## 10. 可复现性说明",
        "本报告优先引用以下目录中的真实结果文件：`outputs/summary/*`、`outputs/metrics/*`、`outputs/figures/*`、`outputs/logs/*`、`results/*` 和 `reports/*`。其中，CSV 主结果的核心索引文件是 `outputs/summary/all_metrics.json`，原始图表来源于 `outputs/figures/CICIoT_csv_merge_a_*`，对应日志来自 `outputs/logs/CICIoT_csv_merge_a_seed*.log`，配置与输入摘要来自 `outputs/metrics/CICIoT_csv_merge_a_seed*.json`。为保证 GitHub 克隆后的报告渲染稳定，本报告实际引用的是从上述原始图表复制到 `reports/assets/` 下的只读副本，新增的汇总图也全部在 `reports/assets/` 下生成。",
        "CTU13 图 benchmark 的核心索引文件是 `results/ctu13_edge_centric_comparison.csv` 与 `results/ctu13_binary_benchmark.csv`，其说明性文件是 `results/ctu13_edge_centric_comparison.md`、`results/ctu13_binary_benchmark.md`、`results/ctu13_flow_label_alignment_summary.md` 和 `results/ctu13_primary_graph_extraction_summary.md`。如果后续需要复查 CTU13 结果，应优先检查这些文件。",
        "如果需要复核 CSV 主结果，应优先检查：`outputs/summary/all_metrics.json` 中的 `CICIoT_csv_merge_a` 条目、对应的 `outputs/metrics/CICIoT_csv_merge_a_seed*.json`、`outputs/logs/CICIoT_csv_merge_a_seed*.log` 以及三张图表文件。若需要复核 CTU13 图 benchmark，应优先检查 `results/ctu13_edge_centric_comparison.csv` 与 `results/ctu13_binary_benchmark.csv`，再结合对齐与图提取摘要文件理解其来源。",
        "仓库中存在历史命名不完全统一的结果文件，例如 `CICIoT_csv_merge_a` 实际对应 `Merged01.csv`，CTU13 的不同研究分支也共存于 `results/` 目录下。因此，本报告对文件的采用遵循“只采纳与最终口径直接对应的真实文件”这一原则，而不对命名历史做无依据猜测。",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown = build_report_markdown()
    REPORT_MD.write_text(markdown, encoding="utf-8")
    REPORT_HTML.write_text(
        _markdown_to_html(markdown, title="最终实验报告"),
        encoding="utf-8",
    )
    print(REPORT_MD.as_posix())
    print(REPORT_HTML.as_posix())


if __name__ == "__main__":
    main()
