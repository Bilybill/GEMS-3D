import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable
from zipfile import BadZipFile

import numpy as np


DEFAULT_PARTITION_ALIASES = {
    "train-river": "development",
    "tests-river": "heldout",
}

FORWARD_STATUS_KEYS = {
    "gravity": "gravity_status",
    "magnetic": "magnetic_status",
    "mt": "mt_status",
    "seismic": "seismic_status",
}


@dataclass(frozen=True)
class ForwardIndexRecord:
    bundle_path: str
    bundle_relpath: str
    raw_partition: str
    partition: str
    source_relpath: str
    background_id: str
    anomaly_type: str
    anomaly_name_en: str
    gravity_status: str
    magnetic_status: str
    mt_status: str
    seismic_status: str
    has_all_modalities: bool

    def to_dict(self) -> dict:
        return {
            "bundle_path": self.bundle_path,
            "bundle_relpath": self.bundle_relpath,
            "raw_partition": self.raw_partition,
            "partition": self.partition,
            "source_relpath": self.source_relpath,
            "background_id": self.background_id,
            "anomaly_type": self.anomaly_type,
            "anomaly_name_en": self.anomaly_name_en,
            "gravity_status": self.gravity_status,
            "magnetic_status": self.magnetic_status,
            "mt_status": self.mt_status,
            "seismic_status": self.seismic_status,
            "has_all_modalities": self.has_all_modalities,
        }


def _safe_item(bundle, key: str, default=""):
    if key not in bundle:
        return default
    value = bundle[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def _derive_background_id(bundle_relpath: str, source_relpath: str) -> str:
    if source_relpath:
        normalized = source_relpath.replace("\\", "/").strip("./")
        return normalized
    parts = bundle_relpath.split(os.sep)
    if len(parts) >= 4:
        return "/".join(parts[1:3])
    if len(parts) >= 2:
        return "/".join(parts[:-2])
    return bundle_relpath


def _normalize_relpath(path: str) -> str:
    return str(path).replace("\\", "/").strip("./")


def _matches_any_prefix(path: str, prefixes: Iterable[str] | None) -> bool:
    if not prefixes:
        return True
    normalized_path = _normalize_relpath(path)
    for prefix in prefixes:
        normalized_prefix = _normalize_relpath(prefix)
        if normalized_prefix and normalized_path.startswith(normalized_prefix):
            return True
    return False


def _iter_forward_bundles(
    root: str,
    include_top_levels: set[str] | None = None,
    development_source_prefixes: Iterable[str] | None = None,
    heldout_source_prefixes: Iterable[str] | None = None,
    partition_aliases: dict[str, str] | None = None,
) -> Iterable[tuple[str, str]]:
    root = os.path.abspath(root)
    alias_map = dict(DEFAULT_PARTITION_ALIASES)
    if partition_aliases:
        alias_map.update(partition_aliases)

    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        if "forward_bundle.npz" not in filenames:
            continue
        bundle_path = os.path.join(current_root, "forward_bundle.npz")
        relpath = os.path.relpath(bundle_path, root)
        raw_partition = relpath.split(os.sep, 1)[0]
        if include_top_levels and raw_partition not in include_top_levels:
            continue
        partition = alias_map.get(raw_partition, raw_partition)
        bundle_dir_relpath = os.path.dirname(relpath)
        if partition == "development" and not _matches_any_prefix(bundle_dir_relpath, development_source_prefixes):
            continue
        if partition == "heldout" and not _matches_any_prefix(bundle_dir_relpath, heldout_source_prefixes):
            continue
        yield bundle_path, relpath


def build_forward_index(
    root: str,
    include_top_levels: Iterable[str] | None = None,
    partition_aliases: dict[str, str] | None = None,
    require_all_modalities: bool = True,
    development_source_prefixes: Iterable[str] | None = None,
    heldout_source_prefixes: Iterable[str] | None = None,
) -> list[dict]:
    alias_map = dict(DEFAULT_PARTITION_ALIASES)
    if partition_aliases:
        alias_map.update(partition_aliases)

    top_levels = set(include_top_levels) if include_top_levels else None
    records: list[dict] = []

    for bundle_path, bundle_relpath in _iter_forward_bundles(
        root,
        include_top_levels=top_levels,
        development_source_prefixes=development_source_prefixes,
        heldout_source_prefixes=heldout_source_prefixes,
        partition_aliases=alias_map,
    ):
        raw_partition = bundle_relpath.split(os.sep, 1)[0]
        partition = alias_map.get(raw_partition, raw_partition)

        try:
            bundle_context = np.load(bundle_path, allow_pickle=True)
        except (BadZipFile, EOFError, OSError, ValueError) as exc:
            print(
                f"[forward-index] warning: skipping unreadable bundle "
                f"{bundle_relpath}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        with bundle_context as bundle:
            gravity_status = str(_safe_item(bundle, "gravity_status", "missing"))
            magnetic_status = str(_safe_item(bundle, "magnetic_status", "missing"))
            mt_status = str(_safe_item(bundle, "mt_status", "missing"))
            seismic_status = str(_safe_item(bundle, "seismic_status", "missing"))
            has_all_modalities = all(
                status == "ok"
                for status in (gravity_status, magnetic_status, mt_status, seismic_status)
            )
            if require_all_modalities and not has_all_modalities:
                continue

            source_relpath = str(_safe_item(bundle, "source_relpath", ""))
            record = ForwardIndexRecord(
                bundle_path=os.path.abspath(bundle_path),
                bundle_relpath=bundle_relpath,
                raw_partition=raw_partition,
                partition=partition,
                source_relpath=source_relpath,
                background_id=_derive_background_id(bundle_relpath, source_relpath),
                anomaly_type=str(_safe_item(bundle, "anomaly_type", "unknown")),
                anomaly_name_en=str(_safe_item(bundle, "anomaly_name_en", "")),
                gravity_status=gravity_status,
                magnetic_status=magnetic_status,
                mt_status=mt_status,
                seismic_status=seismic_status,
                has_all_modalities=has_all_modalities,
            )
            records.append(record.to_dict())

    records.sort(key=lambda item: (item["partition"], item["background_id"], item["anomaly_type"], item["bundle_relpath"]))
    return records


def filter_records_by_source_prefixes(
    records: list[dict],
    development_source_prefixes: Iterable[str] | None = None,
    heldout_source_prefixes: Iterable[str] | None = None,
    development_partition: str = "development",
    heldout_partition: str = "heldout",
) -> list[dict]:
    filtered: list[dict] = []
    for record in records:
        path_to_match = record.get("source_relpath") or record.get("bundle_relpath", "")
        partition = record.get("partition", "")
        if partition == development_partition:
            if _matches_any_prefix(path_to_match, development_source_prefixes):
                filtered.append(record)
            continue
        if partition == heldout_partition:
            if _matches_any_prefix(path_to_match, heldout_source_prefixes):
                filtered.append(record)
            continue
        filtered.append(record)
    return filtered


def _parse_partition_aliases(raw_aliases: list[str] | None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for raw_alias in raw_aliases or []:
        if "=" not in raw_alias:
            raise ValueError(f"Invalid partition alias '{raw_alias}'. Expected format raw=alias.")
        raw_name, alias_name = raw_alias.split("=", 1)
        raw_name = raw_name.strip()
        alias_name = alias_name.strip()
        if not raw_name or not alias_name:
            raise ValueError(f"Invalid partition alias '{raw_alias}'.")
        alias_map[raw_name] = alias_name
    return alias_map


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "..", "DATAFOLDER", "PretrainDataset_forward"))

    parser = argparse.ArgumentParser(description="Build a canonical forward-benchmark index for GMES-3D.")
    parser.add_argument("--root", type=str, default=default_root, help="Root directory containing forward_bundle.npz files.")
    parser.add_argument(
        "--include-top-levels",
        type=str,
        nargs="*",
        default=list(DEFAULT_PARTITION_ALIASES),
        help="Optional raw top-level directories to include.",
    )
    parser.add_argument(
        "--partition-alias",
        type=str,
        nargs="*",
        default=None,
        help="Optional raw=alias mappings. Defaults map train-river to development and tests-river to heldout.",
    )
    parser.add_argument(
        "--development-source-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Optional source_relpath prefixes used to keep only selected development records.",
    )
    parser.add_argument(
        "--heldout-source-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Optional source_relpath prefixes used to keep only selected held-out records.",
    )
    parser.add_argument("--allow-missing-modalities", action="store_true", help="Keep bundles even if one or more modalities did not finish successfully.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    records = build_forward_index(
        args.root,
        include_top_levels=args.include_top_levels,
        partition_aliases=_parse_partition_aliases(args.partition_alias),
        require_all_modalities=not args.allow_missing_modalities,
        development_source_prefixes=args.development_source_prefixes,
        heldout_source_prefixes=args.heldout_source_prefixes,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=True, indent=2)
    print(f"[+] Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
