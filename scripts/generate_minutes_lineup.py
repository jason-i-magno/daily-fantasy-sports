#!/usr/bin/env python3
"""
Build a DraftKings NBA lineup that provably maximizes total projected minutes
using Integer Linear Programming (PuLP).

This is a diagnostic / baseline tool, NOT a fantasy-point optimizer.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
import pulp
from utils import coerce_numeric, normalize_columns, print_table

# ----------------------------
# Helpers
# ----------------------------


def format_lineup(df, assignments):
    rows = []
    for i, s in assignments:
        rows.append(
            {
                "slot": SLOTS[s]["name"],
                "player_name": df.loc[i, "player_name"],
                "position": df.loc[i, "position"],
                "salary": df.loc[i, "salary"],
                "proj_minutes": df.loc[i, "proj_minutes"],
            }
        )
    return pd.DataFrame(rows)


def parse_positions(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    return {p.strip().upper() for p in raw.split("/") if p.strip()}


def load_players(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = normalize_columns(
        df,
        required=[
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "ceiling",
        ],
    )
    df = df[
        ["player_name", "salary", "proj_minutes", "proj_fpts", "position", "ceiling"]
    ].copy()
    df["salary"] = coerce_numeric(df["salary"])
    df["proj_minutes"] = coerce_numeric(df["proj_minutes"])
    df["proj_fpts"] = coerce_numeric(df["proj_fpts"])
    df["positions"] = df["position"].map(parse_positions)
    df["ceiling"] = coerce_numeric(df["ceiling"])

    # Fragility scoring is based solely on projected minutes; drop any extreme low-minute players.
    def fragility_score(minutes: float) -> int:
        if minutes >= 34:
            return 0
        if minutes >= 30:
            return 1
        if minutes >= 26:
            return 2
        if minutes >= 22:
            return 3
        return 4

    df["fragility"] = df["proj_minutes"].apply(fragility_score)

    df = df.dropna(
        subset=["player_name", "salary", "proj_minutes", "proj_fpts", "ceiling"]
    )
    df = df[df["positions"].map(bool)]

    return df.reset_index(drop=True)


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
    reject_fragility: bool = False,
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

    # Fragility cap: sum of per-player fragility scores across the lineup.
    if reject_fragility:
        prob += (
            pulp.lpSum(
                df.loc[i, "fragility"] * y[(i, s)]
                for i in range(n_players)
                for s in range(n_slots)
            )
            <= 6
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


def extract_lineup(df: pd.DataFrame, y) -> pd.DataFrame:
    rows = []
    for (i, s), var in y.items():
        if var.value() == 1:
            rows.append(
                {
                    "slot": SLOTS[s]["name"],
                    "player_name": df.loc[i, "player_name"],
                    "position": df.loc[i, "position"],
                    "salary": df.loc[i, "salary"],
                    "proj_minutes": df.loc[i, "proj_minutes"],
                    "proj_fpts": df.loc[i, "proj_fpts"],
                    "ceiling": df.loc[i, "ceiling"],
                    "player_index": i,
                    "slot_index": s,
                }
            )

    if len(rows) != len(SLOTS):
        raise RuntimeError("Invalid solution extracted")

    return pd.DataFrame(rows)


def is_eligible(player_positions: Set[str], slot_allowed: Optional[Set[str]]) -> bool:
    if slot_allowed is None:
        return True
    return bool(player_positions & slot_allowed)


def solve_top_k_lineups(
    df: pd.DataFrame, cap: float, k: int = 10, maximize_fpts: bool = False
):
    prob, y = build_ilp_with_slots(
        df, cap, maximize_fpts=maximize_fpts, reject_fragility=True
    )

    lineups = []
    chosen_players = set()

    for n in range(k):
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            break

        lineup_df = extract_lineup(df, y)
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


def print_lineup(df: pd.DataFrame) -> None:
    print_table(
        df,
        cols=[
            "slot",
            "player_name",
            "position",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "ceiling",
        ],
        empty_msg="No valid lineup found under the given cap and slot constraints.",
    )
    totals = {
        "total_salary": df["salary"].sum(),
        "total_proj_minutes": df["proj_minutes"].sum(),
        "total_proj_fpts": df["proj_fpts"].sum(),
        "total_ceiling": df["ceiling"].sum(),
    }
    print("\nTotals:")
    for k, v in totals.items():
        print(f"  {k}: {v:.2f}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    df = load_players(args.input)

    # Generate maximum minutes lineup
    prob, y = build_ilp_with_slots(
        df, args.cap, maximize_fpts=False, reject_fragility=False
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    max_minutes_lineup = extract_lineup(df, y)

    # Generate maximum minutes lineup
    prob, y = build_ilp_with_slots(
        df, args.cap, maximize_fpts=True, reject_fragility=False
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    max_fpts_lineup = extract_lineup(df, y)

    # Generate top k lineups
    top_k_lineups = []
    if args.k_lineups > 1:
        # Disallow any player under minutes floor.
        df = df[df["proj_minutes"] >= 22].reset_index(drop=True)

        top_k_lineups = solve_top_k_lineups(
            df,
            cap=args.cap,
            k=args.k_lineups,
            maximize_fpts=True if args.maximize_fpts else False,
        )

    max_fpts = 0
    max_minutes = 0

    with open(args.output, "w") as out_file:
        # Print max minute lineup info to stdout
        print("\nMax Minutes Lineup")
        print_lineup(max_minutes_lineup)

        # Write max minute lineup info to output file
        out_file.write("\nMax Minutes Lineup")
        out_file.write(
            str(
                max_minutes_lineup[
                    [
                        "slot",
                        "player_name",
                        "position",
                        "salary",
                        "proj_minutes",
                        "proj_fpts",
                        "ceiling",
                    ]
                ]
            )
        )
        totals = {
            "total_salary": max_minutes_lineup["salary"].sum(),
            "total_proj_minutes": max_minutes_lineup["proj_minutes"].sum(),
            "total_proj_fpts": max_minutes_lineup["proj_fpts"].sum(),
            "total_ceiling": max_minutes_lineup["ceiling"].sum(),
        }

        out_file.write("\nTotals:")
        for k, v in totals.items():
            out_file.write(f"  {k}: {v:.2f}\n")

        # Print max fpts lineup info to stdout
        print("\nMax FPTS Lineup")
        print_lineup(max_fpts_lineup)

        # Write max fpts lineup info to output file
        out_file.write("\nMax FPTS Lineup")
        out_file.write(
            str(
                max_fpts_lineup[
                    [
                        "slot",
                        "player_name",
                        "position",
                        "salary",
                        "proj_minutes",
                        "proj_fpts",
                        "ceiling",
                    ]
                ]
            )
        )
        totals = {
            "total_salary": max_fpts_lineup["salary"].sum(),
            "total_proj_minutes": max_fpts_lineup["proj_minutes"].sum(),
            "total_proj_fpts": max_fpts_lineup["proj_fpts"].sum(),
            "total_ceiling": max_fpts_lineup["ceiling"].sum(),
        }

        out_file.write("\nTotals:")
        for k, v in totals.items():
            out_file.write(f"  {k}: {v:.2f}\n")

        # Print top k lineups to stdout and write them to the output file
        for i, lineup in enumerate(top_k_lineups):
            print(f"\nLineup #{i}")
            print_lineup(lineup["lineup"])

            out_file.write(f"\nLineup #{i}")
            out_file.write(
                str(
                    lineup["lineup"][
                        [
                            "slot",
                            "player_name",
                            "position",
                            "salary",
                            "proj_minutes",
                            "proj_fpts",
                            "ceiling",
                        ]
                    ]
                )
            )
            totals = {
                "total_salary": lineup["lineup"]["salary"].sum(),
                "total_proj_minutes": lineup["lineup"]["proj_minutes"].sum(),
                "total_proj_fpts": lineup["lineup"]["proj_fpts"].sum(),
                "total_ceiling": lineup["lineup"]["ceiling"].sum(),
            }

            out_file.write("\nTotals:")
            for k, v in totals.items():
                out_file.write(f"  {k}: {v:.2f}\n")

            max_fpts = max(max_fpts, lineup["lineup"]["proj_fpts"].sum())
            max_minutes = max(max_minutes, lineup["lineup"]["proj_minutes"].sum())

    print(f"{max_fpts=}")
    print(f"{max_minutes=}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
