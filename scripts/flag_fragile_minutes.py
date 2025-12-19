#!/usr/bin/env python3
"""
Flag players with projected minutes below a threshold.

Input: CSV with player_name, salary, projected_minutes (case-insensitive, common aliases supported).
Output: prints the flagged rows and writes them to a CSV (default: data/processed/fragile_minutes.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

# Canonical column names and common aliases found in provider exports.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "player_name": ("player_name", "player", "name", "PLAYER", "DKName"),
    "salary": ("salary", "Salary", "SALARY"),
    "projected_minutes": ("projected_minutes", "minutes", "Minutes", "MINUTES"),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using aliases; fail if a required column is missing."""
    df = df.copy()
    renamed = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            renamed[canonical] = canonical
            continue

        match = next((col for col in df.columns if col in aliases), None)
        if match:
            renamed[match] = canonical
        else:
            missing = ", ".join([canonical] + list(aliases))
            raise ValueError(f"Missing required column; expected one of: {missing}")

    return df.rename(columns=renamed)


def filter_fragile(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return rows where projected_minutes is below the given threshold."""
    df = df.copy()
    df["projected_minutes"] = pd.to_numeric(df["projected_minutes"], errors="coerce")
    df["salary"] = pd.to_numeric(
        df["salary"].replace("[\\$,]", "", regex=True), errors="coerce"
    )

    filtered = df[df["projected_minutes"] < threshold].dropna(
        subset=["projected_minutes"]
    )
    return filtered.sort_values(
        by=["projected_minutes", "salary"], ascending=[True, False]
    )


def print_table(df: pd.DataFrame) -> None:
    cols = ["player_name", "salary", "projected_minutes"]
    if df.empty:
        print("No players below the threshold.")
    else:
        print(df[cols].to_string(index=False))


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
    df = normalize_columns(df)

    fragile = filter_fragile(df, args.threshold)

    print_table(fragile)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fragile.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
