#!/usr/bin/env python3
"""
Build a DraftKings NBA lineup that provably maximizes total projected minutes
using Integer Linear Programming (PuLP).

This is a diagnostic / baseline tool, NOT a fantasy-point optimizer.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import pulp
from sqlalchemy import select

from db.models import Contest, Player, Projection, Salary, Source
from db.session import SessionLocal
from scripts.utils import (
    ET,
    MT,
    SLOT_ORDER,
    SLOTS,
    adjusted_score,
    blend_projections,
    get_proj_cols,
    load_dk_salaries_csv,
    load_projections,
    parse_positions,
    parse_slate_id,
    total_fragile_minutes,
)


# ----------------------------
# Helpers
# ----------------------------
def add_game_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Example game_info:  "LAL@SAC 01/12/2026 10:00PM ET"
    df = df.copy()
    df["game_id"] = df["game_info"].apply(lambda x: x.split()[0])
    return df


def build_player_index_map(
    df: pd.DataFrame, id_col: str = "player_name"
) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(df[id_col])}


def load_from_db(meta, projection_source: str):
    """Load DK salaries and projections from the database for the given slate."""
    session = SessionLocal()
    contest_date = datetime.fromisoformat(meta.datetime.split("T")[0])
    contest = (
        session.execute(
            select(Contest).where(
                Contest.date == contest_date, Contest.slate_type == meta.slate
            )
        )
        .scalars()
        .first()
    )
    if not contest:
        raise ValueError(
            f"No contest found in DB for {meta.slate} on {contest_date.date()}"
        )

    # Salaries
    sal_rows = session.execute(
        select(Salary, Player)
        .join(Player, Salary.player_id == Player.id)
        .where(Salary.contest_id == contest.id)
    ).all()

    salary_records = []
    for row in sal_rows:
        salary = row[0]
        player = row[1]
        salary_records.append(
            {
                "player_name": player.key,
                "player_key": player.key,
                "salary": salary.salary,
                "position": salary.position,
                "positions": parse_positions(salary.position),
                "team": salary.team,
                "game_info": salary.game or "",
                "game_id": salary.game or "",
                "game_time_local": pd.NaT,
            }
        )
    dk_df = pd.DataFrame(salary_records)

    def projection_df(source_name: str) -> pd.DataFrame:
        source = session.execute(
            select(Source).where(Source.name == source_name)
        ).scalar_one_or_none()
        if not source:
            return pd.DataFrame()
        rows = session.execute(
            select(Projection, Player)
            .join(Player, Projection.player_id == Player.id)
            .where(
                Projection.contest_id == contest.id,
                Projection.source_id == source.id,
            )
            .order_by(Projection.snapshot_time.desc())
        ).all()
        if not rows:
            return pd.DataFrame()
        latest_time = rows[0][0].snapshot_time
        rows = [r for r in rows if r[0].snapshot_time == latest_time]
        recs = []
        for proj, player in rows:
            recs.append(
                {
                    "player_name": player.key,
                    "player_key": player.key,
                    "proj_minutes": proj.proj_minutes,
                    "proj_fpts": proj.proj_fpts,
                    "game_time_local": proj.snapshot_time.replace(tzinfo=ET).astimezone(
                        MT
                    ),
                }
            )
        return pd.DataFrame(recs)

    rg_df = projection_df("rg")
    etr_df = projection_df("etr")

    def attach_meta(proj_df: pd.DataFrame) -> pd.DataFrame:
        if proj_df.empty:
            return proj_df
        lookup = dk_df.set_index("player_key")
        proj_df = proj_df.copy()
        proj_df["salary"] = proj_df["player_key"].map(lookup["salary"])
        proj_df["positions"] = proj_df["player_key"].map(lookup["positions"])
        proj_df["team"] = proj_df["player_key"].map(lookup["team"])
        if "ceiling" not in proj_df:
            proj_df["ceiling"] = 0.0
        if "floor" not in proj_df:
            proj_df["floor"] = 0.0
        return proj_df

    rg_df = attach_meta(rg_df)
    etr_df = attach_meta(etr_df)

    session.close()

    blend_df = pd.DataFrame()
    if not rg_df.empty and not etr_df.empty:
        # To blend, add salary/positions from DK set
        lookup = dk_df.set_index("player_key")
        for df in (rg_df, etr_df):
            df["salary"] = df["player_key"].map(lookup["salary"])
            df["position"] = df["player_key"].map(lookup["positions"])
            df["positions"] = df["player_key"].map(lookup["positions"])
            df["team"] = df["player_key"].map(lookup["team"])
        blend_df = blend_projections(rg_df, etr_df)

    return dk_df, rg_df, etr_df, blend_df


def write_lineup(
    lineup: pd.DataFrame, proj_source: str, out_file: io.TextIOWrapper
) -> None:
    cols = get_proj_cols(proj_source)

    lineup = lineup.copy()
    lineup["slot"] = pd.Categorical(lineup["slot"], categories=SLOT_ORDER, ordered=True)
    sorted_lineup = lineup.sort_values("slot")

    out_file.write(sorted_lineup[["slot"] + cols].to_string(index=False))
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
slot_index = {
    "PG": 0,
    "SG": 1,
    "SF": 2,
    "PF": 3,
    "C": 4,
    "G": 5,
    "F": 6,
    "UTIL": 7,
}


def build_ilp_with_slots(
    df: pd.DataFrame,
    strategy: str = "max_fpts",
    locked_assignments: dict[int, int] | None = None,
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
    if strategy == "max_fpts":
        # Maximize fpts
        prob += pulp.lpSum(
            df.loc[i, "proj_fpts"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )
    elif strategy == "max_minutes":
        # Maximize minutes
        prob += pulp.lpSum(
            df.loc[i, "proj_minutes"] * y[(i, s)]
            for i in range(n_players)
            for s in range(n_slots)
        )
    else:
        raise ValueError(f"Invalid strategy {strategy}")

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

    # -------------------------------------------
    # Game diversity: require players from >= 2 games
    # -------------------------------------------
    player_game_idx = df["game_idx"].tolist()
    unique_games = sorted(df["game_idx"].unique())
    n_slots = len(SLOTS)

    # Map each game -> list of player indices in that game
    game_to_players: dict[int, list[int]] = {g: [] for g in unique_games}
    for i, gi in enumerate(player_game_idx):
        game_to_players[gi].append(i)

    # Binary indicator: game_used[g] == 1 if any player from game g is in the lineup
    g_used = {g: pulp.LpVariable(f"game_used_{g}", cat="Binary") for g in unique_games}

    # Link players <-> game_used:
    # If any player from game g is used, game_used[g] must be 1.
    # If game_used[g] is 1, at least one player from game g must be used.
    MAX_FROM_ONE_GAME = len(SLOTS)  # at most 8 roster spots total

    for g, players_in_game in game_to_players.items():
        if not players_in_game:
            continue

        total_from_game = pulp.lpSum(
            y[(i, s)] for i in players_in_game for s in range(n_slots)
        )

        # Upper bound: cannot use players from game g unless game_used[g] == 1
        prob += total_from_game <= MAX_FROM_ONE_GAME * g_used[g]

        # Lower bound: if game_used[g] == 1, must take at least one player
        prob += total_from_game >= g_used[g]

    # Require at least 2 distinct games to have players
    prob += pulp.lpSum(g_used[g] for g in unique_games) >= 2

    # ------------------------------
    # Lock specific players into slots
    # ------------------------------
    if locked_assignments:
        for player_idx, slot_idx in locked_assignments.items():
            # Force the chosen player into the chosen slot
            prob += y[(player_idx, slot_idx)] == 1

            # Prevent that same player from appearing in any other slot
            for other_s in range(n_slots):
                if other_s != slot_idx:
                    prob += y[(player_idx, other_s)] == 0

    # Force top projected scorer(s) into the lineup
    # top_fpts_ids = df.sort_values("proj_fpts", ascending=False).head(0).index.tolist()
    # for top_fpts_id in top_fpts_ids:
    #     prob += pulp.lpSum(y[(top_fpts_id, s)] for s in range(len(SLOTS))) == 1

    return prob, y


def extract_lineup(df: pd.DataFrame, y, proj_source: str) -> pd.DataFrame:
    cols = get_proj_cols(proj_source)

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
    proj_source: str,
    k: int = 10,
    strategy: str = "max_fpts",
    locked_assignments: dict[int, int] | None = None,
):
    prob, y = build_ilp_with_slots(
        df, strategy=strategy, locked_assignments=locked_assignments
    )

    lineups = []
    chosen_players = set()

    for n in range(k):
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            break

        lineup_df = extract_lineup(df, y, proj_source)
        total_minutes = lineup_df["proj_minutes"].sum()
        lineup = lineup_df.copy()
        lineup["slot"] = pd.Categorical(
            lineup["slot"], categories=SLOT_ORDER, ordered=True
        )
        sorted_lineup = lineup.sort_values("slot")

        lineups.append(
            {
                "rank": n + 1,
                "total_minutes": total_minutes,
                "lineup": sorted_lineup.copy(),
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
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Load salaries/projections from the database instead of CSVs.",
    )
    parser.add_argument(
        "-f",
        "--filter-game_time",
        action="store_true",
        help="Load salaries/projections from the database instead of CSVs.",
    )
    return parser.parse_args(argv)


def generate_lineups(
    slate_id: str,
    projection_source: str,
    k_lineups: int,
    strategy: str = "max_fpts",
    locked_players: dict = {},
    game_time_filter: datetime = datetime.now(ZoneInfo("America/Denver")),
    use_db: bool = False,
):
    meta = parse_slate_id(slate_id)
    if use_db:
        dk_df, rg_df, etr_df, blend_df = load_from_db(meta, projection_source)
        slate_games = len(set(dk_df["team"])) // 2 if not dk_df.empty else 0
    else:
        dk_df = load_dk_salaries_csv(meta)
        etr_df, rg_df, slate_games = load_projections(
            meta, dk_df=dk_df, remove_nan=True, locked_players=locked_players
        )
        blend_df = blend_projections(rg_df, etr_df)

    working_df = None
    if projection_source == "rg":
        working_df = rg_df.copy()
    elif projection_source == "etr":
        working_df = etr_df.copy()
    elif projection_source == "blend":
        working_df = blend_df.copy()
    else:
        raise ValueError("Invalid projection source")

    # Filter out players not in the current slate
    working_df = working_df[working_df["player_key"].isin(set(dk_df["player_key"]))]

    lookup = dk_df.set_index("player_key")

    # Replace columns with DK columns
    working_df["positions"] = working_df["player_key"].map(lookup["positions"])
    working_df["salary"] = working_df["player_key"].map(lookup["salary"])
    working_df["game_info"] = working_df["player_key"].map(lookup["game_info"])

    # Merge in local game time column
    if not use_db:
        working_df["game_time_local"] = working_df["player_key"].map(
            lookup["game_time_local"]
        )
    elif projection_source == "blend":
        working_df["game_time_local"] = working_df["player_key"].map(
            rg_df.set_index("player_key")["game_time_local"]
        )

    working_df = add_game_id_columns(working_df)

    unique_games = sorted(working_df["game_id"].unique())
    game_index = {g: j for j, g in enumerate(unique_games)}

    working_df["game_idx"] = working_df["game_id"].map(game_index)

    working_df = working_df.reset_index(drop=True)
    locked_keys = set(locked_players.keys())

    if game_time_filter:
        working_df = working_df[
            (working_df["player_key"].isin(locked_keys))
            | (working_df["game_time_local"] > game_time_filter)
        ].reset_index(drop=True)

    locked_assignments = {}
    for player_key, position in locked_players.items():
        locked_assignments[
            working_df.index[working_df["player_key"] == player_key][0]
        ] = slot_index[position]

    if "minutes_floor" in strategy:
        # Disallow any player under minutes floor.
        working_df = working_df[
            (working_df["proj_minutes"] >= 22)
            | (working_df["player_key"].isin(locked_keys))
        ].reset_index(drop=True)

        locked_assignments = {}
        for player_key, position in locked_players.items():
            locked_assignments[
                working_df.index[working_df["player_key"] == player_key][0]
            ] = slot_index[position]

    # Generate top k lineups
    lineups = solve_top_k_lineups(
        working_df,
        proj_source=projection_source,
        k=k_lineups,
        strategy="max_fpts" if "max_fpts" in strategy else "max_minutes",
        locked_assignments=locked_assignments,
    )

    if "adjusted" in strategy:
        for lu in lineups:
            tfm = total_fragile_minutes(lu["lineup"])
            lu["total_fragile_minutes"] = tfm
            # Adjusted score penalizes fragile minutes; acts as a tie-breaker favoring safer minutes on larger slates.
            lu["adjusted_score"] = adjusted_score(
                lu["lineup"]["proj_fpts"].sum(),
                lu["lineup"]["proj_minutes"].sum(),
                tfm,
                slate_games,
            )

        lineups = sorted(lineups, key=lambda x: x["adjusted_score"], reverse=True)

    return lineups


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    locked_players = {}
    locked_players = {
        # "collingillespie": "PG",
        # "devincarter": "SG",
        # "niqueclifford": "SF",
        # "dylancardwell": "PF",
        "donovanclingan": "C",
        "vincewilliamsjr": "G",
        # "micahpotter": "F",
        "jrueholiday": "UTIL",
    }
    game_time_filter = datetime.now(MT) if args.filter_game_time else None

    if "T" in args.slate_id:
        game_time_filter = datetime.fromisoformat(
            args.slate_id.split("_")[3]
        ).astimezone(MT) - timedelta(minutes=1)

    top_minutes_lineup = generate_lineups(
        slate_id=args.slate_id,
        projection_source=args.projection_source,
        k_lineups=1,
        strategy="max_minutes",
        locked_players=locked_players,
        game_time_filter=game_time_filter,
        use_db=args.use_db,
    )[0]

    top_fpts_lineup = generate_lineups(
        slate_id=args.slate_id,
        projection_source=args.projection_source,
        k_lineups=1,
        strategy="max_fpts",
        locked_players=locked_players,
        game_time_filter=game_time_filter,
        use_db=args.use_db,
    )[0]

    adjusted_lineups = generate_lineups(
        slate_id=args.slate_id,
        projection_source=args.projection_source,
        k_lineups=args.k_lineups,
        strategy="adjusted_max_fpts_minutes_floor",
        locked_players=locked_players,
        game_time_filter=game_time_filter,
        use_db=args.use_db,
    )

    with open(args.output, "w") as out_file:
        # Write max minute lineup info to output file
        out_file.write("\nMax Minutes Lineup ")
        write_lineup(top_minutes_lineup["lineup"], args.projection_source, out_file)

        # Write max fpts lineup info to output file
        out_file.write("\nMax FPTS Lineup ")
        write_lineup(top_fpts_lineup["lineup"], args.projection_source, out_file)

        # Print top k lineups to stdout and write them to the output file
        for i, lineup in enumerate(adjusted_lineups):
            out_file.write(f"\nLineup #{i} ")
            write_lineup(lineup["lineup"], args.projection_source, out_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
