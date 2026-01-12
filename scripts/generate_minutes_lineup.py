#!/usr/bin/env python3
"""
Build a DraftKings NBA lineup that provably maximizes total projected minutes
using Integer Linear Programming (PuLP).

This is a diagnostic / baseline tool, NOT a fantasy-point optimizer.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pulp
from utils import (
    ResultsFileMeta,
    adjusted_score,
    blend_projections,
    load_projection_csv,
    normalize_name,
    parse_slate_id,
    total_fragile_minutes,
)

# ----------------------------
# Helpers
# ----------------------------


def build_player_index_map(
    df: pd.DataFrame, id_col: str = "player_name"
) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(df[id_col])}


def lineup_df_to_player_keys(
    lineup_df: pd.DataFrame,
    id_col: str = "player_name",
) -> list[int]:
    player_keys = []
    for name in lineup_df[id_col]:
        player_keys.append(normalize_name(name))
    return player_keys


def write_lineup(
    lineup: pd.DataFrame, cols: list[str], out_file: io.TextIOWrapper
) -> None:
    out_file.write(lineup[["slot"] + cols].to_string(index=False))
    totals = {
        "total_salary": lineup["salary"].sum(),
        "total_proj_minutes": lineup["proj_minutes"].sum(),
        "total_proj_fpts": lineup["proj_fpts"].sum(),
        "total_ceiling": lineup["ceiling"].sum(),
    } | {"total_floor": lineup["floor"].sum() if "floor" in cols else 0.0}

    out_file.write("\nTotals:")
    for k, v in totals.items():
        out_file.write(f"  {k}: {v:.2f}\n")


def write_lineups_to_file(
    out_dir: str,
    meta: ResultsFileMeta,
    slate_size: int,
    proj_source: str,
    top_fpts_player_keys: list[list[int]],
    top_minutes_player_keys: list[list[int]],
):
    payload = {
        "slate_id": meta.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "projection_source": proj_source,
        "slate_size (games)": slate_size,
        "top_fpts": top_fpts_player_keys,
        "top_minutes": top_minutes_player_keys,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = (
        out_path
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.date}.json"
    )
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    return file_path


# ----------------------------
# ILP Model
# ----------------------------
SLOTS: List[Dict] = [
    {"name": "PG", "allowed": {"PG"}},
    {"name": "SG", "allowed": {"SG"}},
    {"name": "SF", "allowed": {"SF"}},
    {"name": "PF", "allowed": {"PF"}},
    {"name": "C", "allowed": {"C"}},
    {"name": "G", "allowed": {"PG", "SG"}},
    {"name": "F", "allowed": {"SF", "PF"}},
    {"name": "UTIL", "allowed": None},
]


def build_ilp_with_slots(
    df: pd.DataFrame,
    maximize_fpts: bool = False,
):
    prob = pulp.LpProblem("dk_max_minutes_with_slots", pulp.LpMaximize)

    n_players = len(df)
    n_slots = len(SLOTS)

    y = {
        (i, s): pulp.LpVariable(f"y_{i}_{s}", cat="Binary")
        for i in range(n_players)
        for s in range(n_slots)
    }

    # Objective
    if maximize_fpts:
        # Maximize fpts
        prob += pulp.lpSum(
            df.loc[i, "proj_fpts"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )

    else:
        # Maximize minutes
        prob += pulp.lpSum(
            df.loc[i, "proj_minutes"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )

    # Salary cap
    salary_cap = 50000
    prob += (
        pulp.lpSum(
            df.loc[i, "salary"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )
        <= salary_cap
    )

    # Each slot filled once
    for s in range(n_slots):
        prob += pulp.lpSum(y[(i, s)] for i in range(n_players)) == 1

    # Each player used at most once
    for i in range(n_players):
        prob += pulp.lpSum(y[(i, s)] for s in range(n_slots)) <= 1

    # Eligibility constraints
    for i in range(n_players):
        ppos = df.loc[i, "positions"]
        for s in range(n_slots):
            allowed = SLOTS[s]["allowed"]
            if allowed is not None and not (ppos & allowed):
                prob += y[(i, s)] == 0

    # Force top projected scorer(s) into the lineup
    # top_fpts_ids = df.sort_values("proj_fpts", ascending=False).head(0).index.tolist()
    # for top_fpts_id in top_fpts_ids:
    #     prob += pulp.lpSum(y[(top_fpts_id, s)] for s in range(len(SLOTS))) == 1

    return prob, y


def extract_lineup(df: pd.DataFrame, y, cols: list[str]) -> pd.DataFrame:
    rows = []
    for (i, s), var in y.items():
        if var.value() == 1:
            rows.append(
                {"slot": SLOTS[s]["name"]}
                | {col: df.loc[i, col] for col in cols}
                | {"player_index": i, "slot_index": s}
            )

    if len(rows) != len(SLOTS):
        raise RuntimeError("Invalid solution extracted")

    return pd.DataFrame(rows)


def solve_top_k_lineups(
    df: pd.DataFrame,
    cols: list[str],
    k: int = 10,
    maximize_fpts: bool = False,
):
    prob, y = build_ilp_with_slots(df, maximize_fpts=maximize_fpts)

    lineups = []
    chosen_players = set()

    for n in range(k):
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            break

        lineup_df = extract_lineup(df, y, cols)
        total_minutes = lineup_df["proj_minutes"].sum()

        lineups.append(
            {
                "rank": n + 1,
                "total_minutes": total_minutes,
                "lineup": lineup_df.copy(),
            }
        )

        # Exclude this player set (DFS-unique lineup)
        chosen_players = set(lineup_df["player_index"])

        prob += (
            pulp.lpSum(y[(i, s)] for i in chosen_players for s in range(len(SLOTS)))
            <= len(chosen_players) - 1
        )

    return lineups


# ----------------------------
# CLI
# ----------------------------


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slate-id", required=True, help="Slate ID")
    parser.add_argument(
        "-p",
        "--projection-source",
        type=str,
        choices=["blend", "etr", "rg"],
        required=True,
        help="Source of projection data.",
    )
    parser.add_argument(
        "-k", "--k-lineups", type=int, default=1, help="Number of lineups to generate"
    )
    parser.add_argument(
        "--maximize-fpts",
        action="store_true",
        help="Set lineup generation objective to maximize FPTS",
    )
    parser.add_argument(
        "--output",
        default="data/processed/dk_max_minutes_lineup.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "-w",
        "--write-candidate-lineups",
        action="store_true",
        help="Flag to write candidate lineups to a file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    meta = parse_slate_id(args.slate_id)
    print(meta.id)

    etr_proj_path = Path(
        f"data/raw/etr/{meta.sport}_{meta.slate}_{meta.site}_etr_projections_{meta.date}.csv"
    )

    if args.projection_source == "etr" and not etr_proj_path.is_file():
        logging.error(f"ETR projection file not found '{etr_proj_path}'")

        return 1

    rg_proj_path = Path(
        f"data/raw/rotogrinders/{meta.sport}_{meta.slate}_{meta.site}_rg_projections_{meta.date}.csv"
    )

    if not rg_proj_path.is_file():
        logging.error(f"Rotogrinders projection file not found '{rg_proj_path}'")

        return 1

    rg_df, slate_games, cols = load_projection_csv(rg_proj_path)
    etr_df, slate_games, cols = load_projection_csv(etr_proj_path, remove_nan=False)

    etr_slate_df = etr_df[etr_df["player_key"].isin(rg_df["player_key"])].copy()

    rg_salary_lookup = rg_df[["player_key", "salary"]].rename(
        columns={"salary": "rg_salary"}
    )
    etr_slate_df = etr_slate_df.merge(
        rg_salary_lookup,
        on="player_key",
        how="left",
    )
    etr_slate_df["salary"] = etr_slate_df["rg_salary"].combine_first(
        etr_slate_df["salary"]
    )
    etr_slate_df = etr_slate_df.drop(columns=["rg_salary"])

    # Players in ETR but not RG (still on slate teams)
    extra_etr_players = set(etr_slate_df["player_key"]) - set(rg_df["player_key"])
    print(f"{extra_etr_players=}")

    # Players in RG but missing from ETR
    missing_rg_players = set(rg_df["player_key"]) - set(etr_slate_df["player_key"])
    print(f"{missing_rg_players=}")

    blend = blend_projections(rg_df, etr_slate_df)

    if args.projection_source == "rg":
        df = rg_df
    elif args.projection_source == "etr":
        df = etr_slate_df
    elif args.projection_source == "blend":
        df = blend

    # Generate top k maximum minutes lineups
    max_minutes_lineups = solve_top_k_lineups(
        df,
        cols=cols,
        k=args.k_lineups,
        maximize_fpts=False,
    )

    # Generate top k maximum FPTs lineus
    max_fpts_lineups = solve_top_k_lineups(
        df,
        cols=cols,
        k=args.k_lineups,
        maximize_fpts=True,
    )

    # Generate top k adjusted lineups
    # Disallow any player under minutes floor.
    df = df[df["proj_minutes"] >= 22].reset_index(drop=True)

    adjusted_lineups = solve_top_k_lineups(
        df,
        cols=cols,
        k=args.k_lineups,
        maximize_fpts=True if args.maximize_fpts else False,
    )

    for lu in adjusted_lineups:
        tfm = total_fragile_minutes(lu["lineup"])
        lu["total_fragile_minutes"] = tfm
        # Adjusted score penalizes fragile minutes; acts as a tie-breaker favoring safer minutes on larger slates.
        lu["adjusted_score"] = adjusted_score(
            lu["lineup"]["proj_fpts"].sum(),
            lu["lineup"]["proj_minutes"].sum(),
            tfm,
            slate_games,
        )

    adjusted_lineups = sorted(
        adjusted_lineups, key=lambda x: x["adjusted_score"], reverse=True
    )

    top_fpts_player_keys = [
        lineup_df_to_player_keys(lu["lineup"]) for lu in max_fpts_lineups
    ]

    top_minutes_player_keys = [
        lineup_df_to_player_keys(lu["lineup"]) for lu in max_minutes_lineups
    ]

    if args.projection_source == "rg":
        candidate_subdir = "rotogrinders"
    elif args.projection_source == "etr":
        candidate_subdir = "etr"
    elif args.projection_source == "blend":
        candidate_subdir = "blend"

    if args.write_candidate_lineups:
        write_lineups_to_file(
            out_dir=f"data/candidate_lineups/{candidate_subdir}",
            meta=meta,
            slate_size=slate_games,
            proj_source=args.projection_source,
            top_fpts_player_keys=top_fpts_player_keys,
            top_minutes_player_keys=top_minutes_player_keys,
        )

    max_fpts = 0
    max_minutes = 0
    max_ceil = 0
    max_floor = 0

    with open(args.output, "w") as out_file:
        # Write max minute lineup info to output file
        out_file.write("\nMax Minutes Lineup ")
        write_lineup(max_minutes_lineups[0]["lineup"], cols, out_file)

        # Write max fpts lineup info to output file
        out_file.write("\nMax FPTS Lineup ")
        write_lineup(max_fpts_lineups[0]["lineup"], cols, out_file)

        # Print top k lineups to stdout and write them to the output file
        for i, lineup in enumerate(adjusted_lineups):
            out_file.write(f"\nLineup #{i} ")
            write_lineup(lineup["lineup"], cols, out_file)

            max_fpts = max(max_fpts, lineup["lineup"]["proj_fpts"].sum())
            max_minutes = max(max_minutes, lineup["lineup"]["proj_minutes"].sum())
            max_ceil = max(max_ceil, lineup["lineup"]["ceiling"].sum())

            if "floor" in cols:
                max_floor = max(max_floor, lineup["lineup"]["floor"].sum())

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
