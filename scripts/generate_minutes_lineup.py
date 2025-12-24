#!/usr/bin/env python3
"""
Build a DraftKings NBA lineup that provably maximizes total projected minutes
using Integer Linear Programming (PuLP).

This is a diagnostic / baseline tool, NOT a fantasy-point optimizer.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from typing import Dict, Iterable, List, Set

import pandas as pd
import pulp
from utils import coerce_numeric, normalize_columns, print_table

# ----------------------------
# Helpers
# ----------------------------


def load_players(path: str) -> tuple[pd.DataFrame, int, list[str]]:
    cols = []
    if "rotogrinders" in path:
        cols = [
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "ceiling",
            "floor",
            "team",
        ]
    elif "etr" in path:
        cols = [
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "ceiling",
            "team",
        ]
    df = pd.read_csv(path)

    df = normalize_columns(
        df,
        required=[
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "team",
        ],
    )
    df = df[cols].copy()
    df["salary"] = coerce_numeric(df["salary"])
    df["proj_minutes"] = coerce_numeric(df["proj_minutes"])
    df["proj_fpts"] = coerce_numeric(df["proj_fpts"])
    df["positions"] = df["position"].map(parse_positions)
    df["ceiling"] = coerce_numeric(df["ceiling"])

    if "rotogrinders" in path:
        df["floor"] = coerce_numeric(df["floor"])

    df = df.dropna(subset=cols)
    df = df[df["positions"].map(bool)]

    def infer_slate_games(frame: pd.DataFrame) -> int:
        # Infer slate size from unique teams (approx games = teams/2). Fallback to 8 if missing.
        teams = set(frame["team"].dropna().unique())

        if len(teams) % 2 != 0:
            raise ValueError("Number of teams must be even.")

        return len(teams) / 2

    slate_games = infer_slate_games(df)

    return df.reset_index(drop=True), slate_games, cols


def parse_positions(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    return {p.strip().upper() for p in raw.split("/") if p.strip()}


def print_lineup(df: pd.DataFrame, cols: list[str]) -> None:
    print_table(
        df,
        cols=["slot"] + cols,
        empty_msg="No valid lineup found under the given cap and slot constraints.",
    )
    totals = {
        "total_salary": df["salary"].sum(),
        "total_proj_minutes": df["proj_minutes"].sum(),
        "total_proj_fpts": df["proj_fpts"].sum(),
        "total_ceiling": df["ceiling"].sum(),
    } | {"total_floor": df["floor"].sum() if "floor" in cols else 0.0}
    print("\nTotals:")
    for k, v in totals.items():
        print(f"  {k}: {v:.2f}")


def write_lineup(
    lineup: pd.DataFrame, cols: list[str], out_file: io.TextIOWrapper
) -> None:
    out_file.write(str(lineup[["slot"] + cols]))
    totals = {
        "total_salary": lineup["salary"].sum(),
        "total_proj_minutes": lineup["proj_minutes"].sum(),
        "total_proj_fpts": lineup["proj_fpts"].sum(),
        "total_ceiling": lineup["ceiling"].sum(),
    } | {"total_floor": lineup["floor"].sum() if "floor" in cols else 0.0}

    out_file.write("\nTotals:")
    for k, v in totals.items():
        out_file.write(f"  {k}: {v:.2f}\n")


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
    cap: float,
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
    prob += (
        pulp.lpSum(
            df.loc[i, "salary"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )
        <= cap
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
    cap: float,
    cols: list[str],
    k: int = 10,
    maximize_fpts: bool = False,
):
    prob, y = build_ilp_with_slots(df, cap, maximize_fpts=maximize_fpts)

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
    parser.add_argument("--input", required=True, help="Path to DK salary CSV")
    parser.add_argument("--cap", type=float, required=True, help="Salary cap")
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
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    df, slate_games, cols = load_players(args.input)

    # Generate maximum minutes lineup
    prob, y = build_ilp_with_slots(df, args.cap, maximize_fpts=False)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    max_minutes_lineup = extract_lineup(df, y, cols)

    # Generate maximum minutes lineup
    prob, y = build_ilp_with_slots(df, args.cap, maximize_fpts=True)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    max_fpts_lineup = extract_lineup(df, y, cols)

    # Generate top k lineups
    top_k_lineups = []
    if args.k_lineups > 1:
        # Disallow any player under minutes floor.
        df = df[df["proj_minutes"] >= 22].reset_index(drop=True)

        top_k_lineups = solve_top_k_lineups(
            df,
            cap=args.cap,
            cols=cols,
            k=args.k_lineups,
            maximize_fpts=True if args.maximize_fpts else False,
        )

        # Soft, slate-aware fragility penalty for ranking only (does not change feasibility).
        def alpha_from_slate(games: int) -> float:
            if games >= 10:
                return 1.0
            if games >= 7:
                return 0.7
            if games >= 5:
                return 0.4
            return 0.2

        # Soft, fixed, slate-independent minutes deficit penalty for ranking only (does not change feasibility).
        beta = 0.05

        # Slate-aware minutes floor
        def minutes_floor_from_slate(games: int) -> float:
            if games >= 10:
                return 258.0
            if games >= 7:
                return 255.0
            if games >= 5:
                return 250.0
            return 245.0

        def total_fragile_minutes(lineup_df: pd.DataFrame) -> float:
            return float(sum(max(0, 30 - m) for m in lineup_df["proj_minutes"]))

        alpha = alpha_from_slate(slate_games)
        minutes_floor = minutes_floor_from_slate(slate_games)
        for lu in top_k_lineups:
            tfm = total_fragile_minutes(lu["lineup"])
            lu["total_fragile_minutes"] = tfm
            # Adjusted score penalizes fragile minutes; acts as a tie-breaker favoring safer minutes on larger slates.
            lu["adjusted_score"] = (
                lu["lineup"]["proj_fpts"].sum()
                - alpha * math.sqrt(tfm)
                - beta * max(0, minutes_floor - lu["lineup"]["proj_minutes"].sum())
            )

        top_k_lineups = sorted(
            top_k_lineups, key=lambda x: x["adjusted_score"], reverse=True
        )

    max_fpts = 0
    max_minutes = 0
    max_ceil = 0
    max_floor = 0

    with open(args.output, "w") as out_file:
        # Print max minute lineup info to stdout
        print("\nMax Minutes Lineup")
        print_lineup(max_minutes_lineup, cols)

        # Write max minute lineup info to output file
        out_file.write("\nMax Minutes Lineup")
        write_lineup(max_minutes_lineup, cols, out_file)

        # Print max fpts lineup info to stdout
        print("\nMax FPTS Lineup")
        print_lineup(max_fpts_lineup, cols)

        # Write max fpts lineup info to output file
        out_file.write("\nMax FPTS Lineup")
        write_lineup(max_fpts_lineup, cols, out_file)

        # Print top k lineups to stdout and write them to the output file
        for i, lineup in enumerate(top_k_lineups):
            print(f"\nLineup #{i}")
            print_lineup(lineup["lineup"], cols)

            out_file.write(f"\nLineup #{i}")
            write_lineup(lineup["lineup"], cols, out_file)

            max_fpts = max(max_fpts, lineup["lineup"]["proj_fpts"].sum())
            max_minutes = max(max_minutes, lineup["lineup"]["proj_minutes"].sum())
            max_ceil = max(max_ceil, lineup["lineup"]["ceiling"].sum())

            if "floor" in cols:
                max_floor = max(max_floor, lineup["lineup"]["floor"].sum())

    print(f"{max_fpts=}")
    print(f"{max_minutes=}")
    print(f"{max_ceil=}")
    print(f"{max_floor=}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
