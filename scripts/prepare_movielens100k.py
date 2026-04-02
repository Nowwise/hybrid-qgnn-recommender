#!/usr/bin/env python3
"""
Download MovieLens-100K and emit train.txt / test.txt in the same LightGCN list format as amazon-book.

- Implicit feedback: ratings >= min_rating (default 4).
- Split: per user, sort by timestamp; last interaction -> test, rest -> train
  (users with only one positive interaction: train only).

Usage (from repo root):
  python scripts/prepare_movielens100k.py --project-root .
  # writes dataset/movielens-100k/{train,test}.txt
  python scripts/prepare_movielens100k.py --project-root . --min-rating 3
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

ML_100K_ZIP = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_zip_bytes(url: str, timeout: int = 120) -> bytes:
    print(f"Downloading {url} …", file=sys.stderr)
    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def parse_u_data_from_zip(zf: zipfile.ZipFile) -> list[tuple[int, int, int, int]]:
    """Return list of (user_raw, item_raw, rating, ts)."""
    names = zf.namelist()
    path = next((n for n in names if n.endswith("u.data")), None)
    if path is None:
        raise FileNotFoundError("u.data not found in zip (expected ml-100k/u.data)")
    raw = zf.read(path).decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        u, i, r, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        rows.append((u, i, r, ts))
    return rows


def build_splits(
    rows: list[tuple[int, int, int, int]], min_rating: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    user_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, i, r, ts in rows:
        if r >= min_rating:
            user_hist[u].append((i, ts))
    train_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []
    for u, lst in user_hist.items():
        lst.sort(key=lambda x: x[1])
        if len(lst) < 2:
            for item_id, _ in lst:
                train_pairs.append((u, item_id))
        else:
            for item_id, _ in lst[:-1]:
                train_pairs.append((u, item_id))
            test_pairs.append((u, lst[-1][0]))
    return train_pairs, test_pairs


def remap_and_write(
    train_pairs: list[tuple[int, int]],
    test_pairs: list[tuple[int, int]],
    out_dir: Path,
) -> None:
    users = sorted({u for u, _ in train_pairs} | {u for u, _ in test_pairs})
    items = sorted({i for _, i in train_pairs} | {i for _, i in test_pairs})
    u_map = {u: j for j, u in enumerate(users)}
    i_map = {i: j for j, i in enumerate(items)}

    train_by_u: dict[int, list[int]] = defaultdict(list)
    for u, i in train_pairs:
        train_by_u[u_map[u]].append(i_map[i])
    test_by_u: dict[int, list[int]] = defaultdict(list)
    for u, i in test_pairs:
        test_by_u[u_map[u]].append(i_map[i])

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    test_path = out_dir / "test.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for u in sorted(train_by_u):
            items_u = sorted(set(train_by_u[u]))
            f.write(str(u) + " " + " ".join(map(str, items_u)) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for u in sorted(test_by_u):
            items_u = sorted(set(test_by_u[u]))
            f.write(str(u) + " " + " ".join(map(str, items_u)) + "\n")

    print(
        f"Wrote {train_path} and {test_path} "
        f"({len(users)} users, {len(items)} items, "
        f"{len(train_pairs)} train pairs, {len(test_pairs)} test pairs)",
        file=sys.stderr,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MovieLens-100K in LightGCN list format.")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repo root (output: <root>/dataset/movielens-100k/)",
    )
    ap.add_argument("--min-rating", type=int, default=4, help="Minimum rating for implicit positive (default 4).")
    ap.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Use local ml-100k.zip instead of downloading.",
    )
    args = ap.parse_args()
    root: Path = args.project_root
    out_dir = root / "dataset" / "movielens-100k"

    if args.zip_path is not None:
        zdata = args.zip_path.read_bytes()
    else:
        zdata = download_zip_bytes(ML_100K_ZIP)

    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        rows = parse_u_data_from_zip(zf)

    train_pairs, test_pairs = build_splits(rows, args.min_rating)
    if not train_pairs:
        sys.exit("No training pairs after filtering; check min_rating.")
    remap_and_write(train_pairs, test_pairs, out_dir)


if __name__ == "__main__":
    main()
