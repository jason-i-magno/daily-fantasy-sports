#!/usr/bin/env python3
"""
Compare projections from two sources and highlight meaningful disagreements.

Inputs: two CSVs with player_name, projected_fpts, projected_minutes (common aliases supported).
Output: merged table with both projections, differences, sorted by absolute point difference.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

import pandas as pd
from utils import (
    ensure_output_path,
    load_projection_csv,
    print_table,
)


def merge_and_diff(etr: pd.DataFrame, rg: pd.DataFrame) -> pd.DataFrame:
    merged = etr.merge(
        rg,
        on="player_key",
        how="inner",
        suffixes=("_etr", "_rg"),
    )
    merged["fpts_diff"] = merged["proj_fpts_etr"] - merged["proj_fpts_rg"]
    merged["minutes_diff"] = merged["proj_minutes_etr"] - merged["proj_minutes_rg"]
    merged["abs_diff"] = merged["minutes_diff"].abs()
    new_order = [
        "player_key",
        "proj_minutes_etr",
        "proj_minutes_rg",
        "minutes_diff",
        "abs_diff",
        "proj_fpts_etr",
        "proj_fpts_rg",
        "fpts_diff",
    ]
    merged = merged[new_order]
    return merged.sort_values("abs_diff", ascending=False)


def print_preview(df: pd.DataFrame, limit: int = 20) -> None:
    cols = [
        "player_key",
        "proj_fpts_etr",
        "proj_fpts_rg",
        "fpts_diff",
        "proj_minutes_etr",
        "proj_minutes_rg",
        "minutes_diff",
    ]
    print_table(
        df,
        cols=cols,
        limit=limit,
        empty_msg="No overlapping players after normalization.",
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etr", required=True, help="Path to ETR projections CSV.")
    parser.add_argument(
        "--rg", required=True, help="Path to RotoGrinders projections CSV."
    )
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

    etr_df, _, _ = load_projection_csv(args.etr)
    rg_df, _, _ = load_projection_csv(args.rg)

    merged = merge_and_diff(etr_df, rg_df)
    if args.min_diff > 0:
        merged = merged[merged["abs_diff"] >= args.min_diff]

    print_preview(merged, limit=args.preview_rows)

    output_path = ensure_output_path(args.output)
    merged.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
