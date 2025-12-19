#!/usr/bin/env python3
"""
Build a DraftKings NBA lineup that provably maximizes total projected minutes
using Integer Linear Programming (PuLP).

This is a diagnostic / baseline tool, NOT a fantasy-point optimizer.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pulp
from utils import coerce_numeric, normalize_columns, print_table

# ----------------------------
# Helpers
# ----------------------------


def parse_positions(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    return {p.strip().upper() for p in raw.split("/") if p.strip()}


def load_players(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = normalize_columns(
        df, required=["player_name", "salary", "proj_minutes", "position"]
    )
    df = df[["player_name", "salary", "proj_minutes", "position"]].copy()
    df["salary"] = coerce_numeric(df["salary"])
    df["proj_minutes"] = coerce_numeric(df["proj_minutes"])
    df["positions"] = df["position"].map(parse_positions)

    df = df.dropna(subset=["player_name", "salary", "proj_minutes"])
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


def is_eligible(player_positions: Set[str], slot_allowed: Optional[Set[str]]) -> bool:
    if slot_allowed is None:
        return True
    return bool(player_positions & slot_allowed)


def solve_max_minutes_with_slots(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    # Build ILP with assignment variables y[i,s]
    prob = pulp.LpProblem("dk_max_minutes_with_slot_assignment", pulp.LpMaximize)

    n_players = len(df)
    n_slots = len(SLOTS)

    # y[(i,s)] = 1 if player i assigned to slot s
    y: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for i in range(n_players):
        for s in range(n_slots):
            y[(i, s)] = pulp.LpVariable(f"y_{i}_{s}", cat="Binary")

    # Objective: maximize total projected minutes of chosen assignments
    prob += pulp.lpSum(
        df.loc[i, "proj_minutes"] * y[(i, s)]
        for i in range(n_players)
        for s in range(n_slots)
    )

    # Salary cap: each chosen player counts once (via slot assignment)
    prob += (
        pulp.lpSum(
            df.loc[i, "salary"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )
        <= cap
    )

    # Each slot filled by exactly one player
    for s in range(n_slots):
        prob += pulp.lpSum(y[(i, s)] for i in range(n_players)) == 1

    # Each player used at most once across all slots
    for i in range(n_players):
        prob += pulp.lpSum(y[(i, s)] for s in range(n_slots)) <= 1

    # Eligibility constraints: disallow assigning player i to slot s if not eligible
    for i in range(n_players):
        ppos = df.loc[i, "positions"]
        for s in range(n_slots):
            allowed = SLOTS[s]["allowed"]
            if not is_eligible(ppos, allowed):
                prob += y[(i, s)] == 0

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"No optimal solution found (status={pulp.LpStatus[status]})"
        )

    # Extract lineup (one row per slot)
    rows = []
    for s in range(n_slots):
        chosen_i = None
        for i in range(n_players):
            if y[(i, s)].value() == 1:
                chosen_i = i
                break
        if chosen_i is None:
            raise RuntimeError(
                f"Slot {SLOTS[s]['name']} not assigned in solution (unexpected)."
            )

        rows.append(
            {
                "slot": SLOTS[s]["name"],
                "player_name": df.loc[chosen_i, "player_name"],
                "position": df.loc[chosen_i, "position"],
                "salary": float(df.loc[chosen_i, "salary"]),
                "proj_minutes": float(df.loc[chosen_i, "proj_minutes"]),
            }
        )

    lineup = pd.DataFrame(rows)
    # Optional: stable presentation order by slot list
    lineup["slot_order"] = lineup["slot"].map(
        {SLOTS[i]["name"]: i for i in range(n_slots)}
    )
    lineup = (
        lineup.sort_values("slot_order")
        .drop(columns=["slot_order"])
        .reset_index(drop=True)
    )
    return lineup


# ----------------------------
# CLI
# ----------------------------


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to DK salary CSV")
    parser.add_argument("--cap", type=float, required=True, help="Salary cap")
    parser.add_argument(
        "--output",
        default="data/processed/dk_max_minutes_lineup.csv",
        help="Output CSV path",
    )
    return parser.parse_args(argv)


def print_lineup(df: pd.DataFrame) -> None:
    print_table(
        df,
        cols=["slot", "player_name", "position", "salary", "proj_minutes"],
        empty_msg="No valid lineup found under the given cap and slot constraints.",
    )
    totals = {
        "total_salary": df["salary"].sum(),
        "total_proj_minutes": df["proj_minutes"].sum(),
    }
    print("\nTotals:")
    for k, v in totals.items():
        print(f"  {k}: {v:.2f}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    df = load_players(args.input)
    lineup = solve_max_minutes_with_slots(df, cap=args.cap)
    print(lineup)
    print_lineup(lineup)

    lineup.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
