"""CTU-13 dataset discovery, manifest management, and label-file parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import csv
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


CTU13_OFFICIAL_PAGE = "https://www.stratosphereips.org/datasets-ctu13"
CTU13_SCENARIO_BASE = "https://mcfp.felk.cvut.cz/publicDatasets/"


@dataclass(frozen=True, slots=True)
class CTU13DirectoryEntry:
    """One file or subdirectory entry extracted from an Apache-style listing."""

    href: str
    name: str
    size_text: str | None = None


@dataclass(frozen=True, slots=True)
class CTU13ScenarioManifestEntry:
    """Manifest entry describing one CTU-13 scenario and local assets."""

    scenario_id: str
    scenario_name: str
    scenario_url: str
    pcap_source_url: str | None = None
    label_source_url: str | None = None
    readme_source_url: str | None = None
    pcap_path: str | None = None
    label_file_path: str | None = None
    readme_path: str | None = None
    pcap_compressed_path: str | None = None
    download_status: str = "pending"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CTU13ScenarioManifestEntry":
        return cls(
            scenario_id=str(data.get("scenario_id", "")).strip(),
            scenario_name=str(data.get("scenario_name", "")).strip(),
            scenario_url=str(data.get("scenario_url", "")).strip(),
            pcap_source_url=_optional_str(data.get("pcap_source_url")),
            label_source_url=_optional_str(data.get("label_source_url")),
            readme_source_url=_optional_str(data.get("readme_source_url")),
            pcap_path=_optional_str(data.get("pcap_path")),
            label_file_path=_optional_str(data.get("label_file_path")),
            readme_path=_optional_str(data.get("readme_path")),
            pcap_compressed_path=_optional_str(data.get("pcap_compressed_path")),
            download_status=str(data.get("download_status", "pending")).strip() or "pending",
            notes=tuple(str(item) for item in data.get("notes", ()) if str(item).strip()),
        )


@dataclass(frozen=True, slots=True)
class CTU13LabeledFlow:
    """One official CTU-13 labeled bidirectional flow row."""

    scenario_id: str
    start_time: datetime
    end_time: datetime
    protocol: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    label_text: str
    binary_label: str
    raw_row: dict[str, str]


def _optional_str(value: object) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(text: str) -> datetime:
    normalized = text.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported CTU-13 timestamp: {text}")


def _parse_duration_seconds(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _parse_port(value: str, *, protocol: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)
    service_name = text.lower()
    service_overrides = {
        ("tcp", "http"): 80,
        ("tcp", "https"): 443,
        ("tcp", "ssl"): 443,
        ("tcp", "smtp"): 25,
        ("tcp", "pop3"): 110,
        ("tcp", "imap"): 143,
        ("udp", "domain"): 53,
        ("tcp", "domain"): 53,
    }
    return service_overrides.get((protocol.lower(), service_name), 0)


def ctu13_binary_label(label_text: str) -> str:
    """Map one official CTU-13 flow label into benign/malicious/unknown."""

    normalized = label_text.strip().lower()
    if "background" in normalized:
        return "unknown"
    if "botnet" in normalized or "c&c" in normalized or "cc" in normalized:
        return "malicious"
    if "normal" in normalized:
        return "benign"
    return "unknown"


class _HrefParser(HTMLParser):
    """Very small HTML parser for robust href extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = "".join(self._current_text_parts).strip()
        self.entries.append((self._current_href, text))
        self._current_href = None
        self._current_text_parts = []


def _fetch_text(url: str, *, timeout: float = 30.0) -> str:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def discover_ctu13_scenario_urls() -> dict[str, str]:
    """Discover official CTU-13 scenario URLs from the official page."""

    parser = _HrefParser()
    parser.feed(_fetch_text(CTU13_OFFICIAL_PAGE))
    scenario_urls: dict[str, str] = {}
    for href, _text in parser.entries:
        if "CTU-Malware-Capture-Botnet-" not in href:
            continue
        absolute = urljoin(CTU13_OFFICIAL_PAGE, href)
        match = re.search(r"Botnet-(\d+)", absolute)
        if match:
            scenario_urls[match.group(1)] = absolute.rstrip("/") + "/"
    return dict(sorted(scenario_urls.items()))


def parse_ctu13_directory_index(url: str) -> list[CTU13DirectoryEntry]:
    """Parse one official CTU-13 directory index into stable entries."""

    html = _fetch_text(url)
    parser = _HrefParser()
    parser.feed(html)
    size_by_href: dict[str, str] = {}
    table_row_pattern = re.compile(
        r'<tr><td[^>]*>.*?</td><td><a href="([^"]+)">([^<]+)</a></td>'
        r'<td[^>]*>[^<]*</td><td align="right">\s*([^<]+)</td>',
        re.IGNORECASE,
    )
    for match in table_row_pattern.finditer(html):
        size_by_href[match.group(1)] = match.group(3).strip()
    entries: list[CTU13DirectoryEntry] = []
    seen: set[str] = set()
    for href, text in parser.entries:
        if href in seen:
            continue
        seen.add(href)
        entries.append(
            CTU13DirectoryEntry(
                href=href,
                name=text or href,
                size_text=size_by_href.get(href),
            )
        )
    return entries


def build_ctu13_manifest_entry(scenario_id: str, scenario_url: str) -> CTU13ScenarioManifestEntry:
    """Discover downloadable assets for one CTU-13 scenario directory."""

    entries = parse_ctu13_directory_index(scenario_url)
    pcap_entry = next(
        (entry for entry in entries if entry.href.endswith(".truncated.pcap.bz2")),
        None,
    )
    label_entry = next(
        (
            entry
            for entry in entries
            if entry.href.endswith(".binetflow.2format") or entry.href.endswith(".binetflow")
        ),
        None,
    )
    readme_entry = next(
        (
            entry
            for entry in entries
            if entry.href.lower().startswith("readme.")
        ),
        None,
    )
    scenario_name = f"CTU-13 Scenario {scenario_id}"
    return CTU13ScenarioManifestEntry(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        scenario_url=scenario_url,
        pcap_source_url=urljoin(scenario_url, pcap_entry.href) if pcap_entry else None,
        label_source_url=urljoin(scenario_url, label_entry.href) if label_entry else None,
        readme_source_url=urljoin(scenario_url, readme_entry.href) if readme_entry else None,
    )


def load_ctu13_manifest(path: str | Path) -> list[CTU13ScenarioManifestEntry]:
    """Load a CTU-13 manifest from disk."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("CTU-13 manifest must be a list.")
    return [CTU13ScenarioManifestEntry.from_mapping(item) for item in payload]


def save_ctu13_manifest(
    entries: list[CTU13ScenarioManifestEntry],
    path: str | Path,
) -> None:
    """Write a CTU-13 manifest to disk."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_dict() for entry in entries]
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def infer_local_ctu13_manifest_entry(
    scenario_id: str,
    *,
    raw_root: str | Path,
) -> CTU13ScenarioManifestEntry | None:
    """Infer one manifest entry directly from a locally materialized scenario directory."""

    scenario_dir = Path(raw_root) / f"scenario_{scenario_id}"
    if not scenario_dir.exists():
        return None

    pcap_path = next(
        (
            candidate
            for candidate in sorted(scenario_dir.glob("*.truncated.pcap"))
            if candidate.is_file()
        ),
        None,
    )
    compressed_path = next(
        (
            candidate
            for candidate in sorted(scenario_dir.glob("*.truncated.pcap.bz2"))
            if candidate.is_file()
        ),
        None,
    )
    label_path = next(
        (
            candidate
            for candidate in sorted(scenario_dir.glob("*.binetflow.2format"))
            if candidate.is_file()
        ),
        None,
    )
    if label_path is None:
        label_path = next(
            (
                candidate
                for candidate in sorted(scenario_dir.glob("*.binetflow"))
                if candidate.is_file()
            ),
            None,
        )
    readme_path = next(
        (
            candidate
            for candidate in sorted(scenario_dir.glob("README*"))
            if candidate.is_file()
        ),
        None,
    )
    if pcap_path is None or label_path is None:
        status = "partial"
    else:
        status = "downloaded"

    pcap_name = pcap_path.name if pcap_path is not None else (
        compressed_path.name[:-4] if compressed_path and compressed_path.name.endswith(".bz2") else None
    )
    label_name = label_path.name if label_path is not None else None
    readme_name = readme_path.name if readme_path is not None else None
    scenario_name = f"CTU-Malware-Capture-Botnet-{scenario_id}"
    scenario_url = f"{CTU13_SCENARIO_BASE}{scenario_name}/"
    return CTU13ScenarioManifestEntry(
        scenario_id=str(scenario_id),
        scenario_name=scenario_name,
        scenario_url=scenario_url,
        pcap_source_url=urljoin(scenario_url, f"{pcap_name}.bz2") if pcap_name else None,
        label_source_url=urljoin(scenario_url, label_name) if label_name else None,
        readme_source_url=urljoin(scenario_url, readme_name) if readme_name else None,
        pcap_path=pcap_path.as_posix() if pcap_path is not None else None,
        label_file_path=label_path.as_posix() if label_path is not None else None,
        readme_path=readme_path.as_posix() if readme_path is not None else None,
        pcap_compressed_path=compressed_path.as_posix() if compressed_path is not None else None,
        download_status=status,
        notes=("inferred from local raw directory",),
    )


def merge_ctu13_manifest_with_local_raw(
    entries: list[CTU13ScenarioManifestEntry],
    *,
    raw_root: str | Path,
    scenario_ids: list[str] | tuple[str, ...] | None = None,
) -> list[CTU13ScenarioManifestEntry]:
    """Backfill manifest entries from local raw directories without clobbering existing data."""

    merged: dict[str, CTU13ScenarioManifestEntry] = {entry.scenario_id: entry for entry in entries}
    if scenario_ids is None:
        scenario_dirs = sorted(Path(raw_root).glob("scenario_*"))
        candidate_ids = [
            directory.name.split("_", 1)[1]
            for directory in scenario_dirs
            if directory.is_dir() and "_" in directory.name
        ]
    else:
        candidate_ids = [str(item) for item in scenario_ids]

    for scenario_id in candidate_ids:
        inferred = infer_local_ctu13_manifest_entry(scenario_id, raw_root=raw_root)
        if inferred is None:
            continue
        existing = merged.get(scenario_id)
        if existing is None:
            merged[scenario_id] = inferred
            continue
        merged[scenario_id] = CTU13ScenarioManifestEntry(
            scenario_id=existing.scenario_id,
            scenario_name=existing.scenario_name or inferred.scenario_name,
            scenario_url=existing.scenario_url or inferred.scenario_url,
            pcap_source_url=existing.pcap_source_url or inferred.pcap_source_url,
            label_source_url=existing.label_source_url or inferred.label_source_url,
            readme_source_url=existing.readme_source_url or inferred.readme_source_url,
            pcap_path=existing.pcap_path or inferred.pcap_path,
            label_file_path=existing.label_file_path or inferred.label_file_path,
            readme_path=existing.readme_path or inferred.readme_path,
            pcap_compressed_path=existing.pcap_compressed_path or inferred.pcap_compressed_path,
            download_status=existing.download_status if existing.download_status != "pending" else inferred.download_status,
            notes=tuple(dict.fromkeys((*existing.notes, *inferred.notes))),
        )
    return [merged[key] for key in sorted(merged)]


def parse_ctu13_label_file(
    path: str | Path,
    *,
    scenario_id: str,
) -> list[CTU13LabeledFlow]:
    """Parse one official CTU-13 labeled bidirectional flow file."""

    label_path = Path(path)
    rows: list[CTU13LabeledFlow] = []
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            try:
                start_time = _parse_datetime(raw_row["StartTime"])
            except (KeyError, ValueError):
                continue
            duration = _parse_duration_seconds(raw_row.get("Dur", "0"))
            protocol = raw_row.get("Proto", "unknown").strip().lower()
            label_text = raw_row.get("Label", "").strip()
            rows.append(
                CTU13LabeledFlow(
                    scenario_id=scenario_id,
                    start_time=start_time,
                    end_time=start_time + timedelta(seconds=duration),
                    protocol=protocol,
                    src_ip=raw_row.get("SrcAddr", "").strip(),
                    src_port=_parse_port(raw_row.get("Sport", "0"), protocol=protocol),
                    dst_ip=raw_row.get("DstAddr", "").strip(),
                    dst_port=_parse_port(raw_row.get("Dport", "0"), protocol=protocol),
                    label_text=label_text,
                    binary_label=ctu13_binary_label(label_text),
                    raw_row={key: value for key, value in raw_row.items()},
                )
            )
    return rows


__all__ = [
    "CTU13DirectoryEntry",
    "CTU13LabeledFlow",
    "CTU13ScenarioManifestEntry",
    "CTU13_OFFICIAL_PAGE",
    "build_ctu13_manifest_entry",
    "ctu13_binary_label",
    "discover_ctu13_scenario_urls",
    "infer_local_ctu13_manifest_entry",
    "load_ctu13_manifest",
    "merge_ctu13_manifest_with_local_raw",
    "parse_ctu13_directory_index",
    "parse_ctu13_label_file",
    "save_ctu13_manifest",
]
