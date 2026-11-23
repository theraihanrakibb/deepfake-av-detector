#!/usr/bin/env python3
"""Validate a rich or legacy train/validation/test split manifest."""

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = Path(sys.argv[1])
    raw = json.loads(path.read_text())
    data = raw.get("splits", raw)
    summary = {}
    paths_by_split = {}
    groups_by_split = {}
    for split in ("train", "validation", "test"):
        rows = data[split]
        if not rows:
            raise ValueError(f"Empty split: {split}")
        labels = set()
        paths = set()
        groups = set()
        for row in rows:
            if isinstance(row, dict):
                video_path, label = row["path"], int(row["label"])
                group = row.get("group")
            else:
                video_path, label = row[0], int(row[1])
                group = None
            if label not in (0, 1):
                raise ValueError(f"Invalid label in {split}: {label}")
            candidate = Path(video_path)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            labels.add(label)
            paths.add(str(candidate))
            if group is not None:
                groups.add(str(group))
        if labels != {0, 1}:
            raise ValueError(f"Split {split} lacks both classes: {labels}")
        summary[split] = len(rows)
        paths_by_split[split] = paths
        groups_by_split[split] = groups
    for left, right in combinations(paths_by_split, 2):
        if paths_by_split[left] & paths_by_split[right]:
            raise ValueError(f"Path leakage between {left} and {right}")
        if groups_by_split[left] and groups_by_split[right]:
            if groups_by_split[left] & groups_by_split[right]:
                raise ValueError(f"Group leakage between {left} and {right}")
    print(f"Validated {path}: {summary}")


if __name__ == "__main__":
    main()
