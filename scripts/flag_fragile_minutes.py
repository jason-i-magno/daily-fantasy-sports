#!/usr/bin/env python3
"""
Flag players with projected minutes below a threshold.

Input: CSV with player_name, salary, projected_minutes (case-insensitive, common aliases supported).
Output: prints the flagged rows and writes them to a CSV (default: data/processed/fragile_minutes.csv).
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

import pandas as pd

from utils import coerce_numeric, ensure_output_path, normalize_columns, print_table


def filter_fragile(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return rows where projected_minutes is below the given threshold."""
    df = df.copy()
    df["projected_minutes"] = coerce_numeric(df["projected_minutes"])
    df["salary"] = coerce_numeric(df["salary"])

    filtered = df[df["projected_minutes"] < threshold].dropna(
        subset=["projected_minutes"]
    )
    return filtered.sort_values(
        by=["projected_minutes", "salary"], ascending=[True, False]
    )

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with player_name/salary/projected_minutes columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Flag players with projected minutes below this value.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/fragile_minutes.csv",
        help="Path to write flagged players CSV (default: data/processed/fragile_minutes.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    df = pd.read_csv(args.input)
    df = normalize_columns(
        df, required=["player_name", "salary", "projected_minutes"]
    )

    fragile = filter_fragile(df, args.threshold)

    print_table(
        fragile,
        cols=["player_name", "salary", "projected_minutes"],
        empty_msg="No players below the threshold.",
    )

    output_path = ensure_output_path(args.output)
    fragile.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
