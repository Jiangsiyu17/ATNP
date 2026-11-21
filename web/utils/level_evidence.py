"""Lookup exclusive A-D evidence levels for compound/plant pairs."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path


DEFAULT_FILE = Path(
    "/data2/jiangsiyu/ATNP_Database/level_data/all_pairs_with_levels.csv"
)


def _ik14(value):
    return str(value or "").strip().upper()[:14]


def _species(value):
    value = re.sub(r"\([^)]*\)", "", str(value or "").strip())
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return " ".join(value.split()[:2])


def _level(row):
    # The file stores mutually exclusive boolean columns; use the explicit
    # priority as a safeguard if a row was produced by another version.
    for label in ("A", "B", "C", "D"):
        if str(row.get(f"level_{label}", "")).strip().casefold() in {"true", "1", "yes"}:
            return label
    return "Unclassified"


@lru_cache(maxsize=1)
def _load_levels(path):
    result = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (_ik14(row.get("inchikey")), _species(row.get("plant")))
                if key[0] and key[1]:
                    result[key] = _level(row)
    except (FileNotFoundError, OSError):
        return {}
    return result


def get_pair_level(inchikey, plant):
    path = os.environ.get("ATNP_LEVELS_FILE", str(DEFAULT_FILE))
    return _load_levels(path).get((_ik14(inchikey), _species(plant)), "Unclassified")
