#!/usr/bin/env python3
"""
Compare projections from two sources and highlight meaningful disagreements.

Inputs: two CSVs with player_name, projected_points, projected_minutes (common aliases supported).
Output: merged table with both projections, differences, sorted by absolute point difference.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

# Canonical column names and common aliases found in provider exports.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "player_name": ("player_name", "player", "name", "PLAYER", "DKName"),
    "projected_points": ("projected_points", "proj", "Proj", "FPTS", "PTS"),
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


def normalize_name(name: str) -> str:
    """Lowercase, strip, remove punctuation/spacing accents for simple matching."""
    if not isinstance(name, str):
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    cleaned = "".join(ch for ch in normalized if ch.isalnum())
    return cleaned.strip()


def load_projection(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = df[["player_name", "projected_points", "projected_minutes"]].copy()
    df["player_key"] = df["player_name"].map(normalize_name)
    df["projected_points"] = pd.to_numeric(df["projected_points"], errors="coerce")
    df["projected_minutes"] = pd.to_numeric(df["projected_minutes"], errors="coerce")
    return df.dropna(subset=["player_key"])


def merge_and_diff(etr: pd.DataFrame, rg: pd.DataFrame) -> pd.DataFrame:
    merged = etr.merge(
        rg,
        on="player_key",
        how="inner",
        suffixes=("_etr", "_rg"),
    )
    merged["points_diff"] = merged["projected_points_etr"] - merged["projected_points_rg"]
    merged["abs_diff"] = merged["points_diff"].abs()
    merged["minutes_diff"] = merged["projected_minutes_etr"] - merged["projected_minutes_rg"]
    return merged.sort_values("abs_diff", ascending=False)


def print_table(df: pd.DataFrame, limit: int = 20) -> None:
    cols = [
        "player_name_etr",
        "player_name_rg",
        "projected_points_etr",
        "projected_points_rg",
        "points_diff",
        "projected_minutes_etr",
        "projected_minutes_rg",
        "minutes_diff",
    ]
    preview = df.head(limit)[cols] if not df.empty else df
    if preview.empty:
        print("No overlapping players after normalization.")
    else:
        print(preview.to_string(index=False))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etr", required=True, help="Path to ETR projections CSV.")
    parser.add_argument("--rg", required=True, help="Path to RotoGrinders projections CSV.")
    parser.add_argument(
        "--output",
        default="data/processed/projection_differences.csv",
        help="Path to write merged projection differences (CSV).",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=0.0,
        help="Filter to rows with absolute point difference at or above this value.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of rows to show in stdout preview.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    etr_df = load_projection(args.etr)
    rg_df = load_projection(args.rg)

    merged = merge_and_diff(etr_df, rg_df)
    if args.min_diff > 0:
        merged = merged[merged["abs_diff"] >= args.min_diff]

    print_table(merged, limit=args.preview_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
