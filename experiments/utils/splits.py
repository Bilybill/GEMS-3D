from __future__ import annotations

import random
from collections import defaultdict


def split_records_by_background(
    records: list[dict],
    validation_fraction: float = 0.1,
    seed: int = 0,
    development_partition: str = "development",
    heldout_partition: str = "heldout",
) -> tuple[list[dict], list[dict], list[dict]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    heldout_records: list[dict] = []
    for record in records:
        if record["partition"] == heldout_partition:
            heldout_records.append(record)
        elif record["partition"] == development_partition:
            grouped[record["background_id"]].append(record)

    background_ids = sorted(grouped)
    if not background_ids:
        raise ValueError("No development records were found for background-disjoint splitting.")

    rng = random.Random(seed)
    rng.shuffle(background_ids)

    num_val_backgrounds = max(1, int(round(len(background_ids) * validation_fraction)))
    val_backgrounds = set(background_ids[:num_val_backgrounds])

    train_records: list[dict] = []
    val_records: list[dict] = []
    for background_id, background_records in grouped.items():
        if background_id in val_backgrounds:
            val_records.extend(background_records)
        else:
            train_records.extend(background_records)

    train_records.sort(key=lambda item: (item["background_id"], item["anomaly_type"], item["bundle_relpath"]))
    val_records.sort(key=lambda item: (item["background_id"], item["anomaly_type"], item["bundle_relpath"]))
    heldout_records.sort(key=lambda item: (item["background_id"], item["anomaly_type"], item["bundle_relpath"]))
    return train_records, val_records, heldout_records
