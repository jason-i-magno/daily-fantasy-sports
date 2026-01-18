#!/usr/bin/env python3
"""
Minimal, extensible backtesting for NBA DK H2H lineup selection.

Focus: validate deterministic decision rules (not GPP sims or parameter tuning).
Assumptions:
- Lineups are represented by player sets (slot order ignored).
- Inputs are pre-lock projections and post-slate results CSVs.
- No opponent or ownership modeling; H2H win = higher total actual DK points.

TODO: extend with ownership-aware baselines, entry-fee buckets, and per-position eligibility checks if needed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import List, Set

import pandas as pd
from utils import (
    adjusted_score,
    blend_projections,
    load_projection_csv,
    normalize_name,
    parse_filename,
    total_fragile_minutes,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Data structures
# ----------------------------
@dataclasses.dataclass(frozen=True)
class Lineup:
    players: frozenset
    projected_fpts: float
    projected_minutes: float
    rg_floor: float
    fragile_count: int
    total_fragile_minutes: float
    actual_fpts: float


@dataclasses.dataclass
class SlateResult:
    slate_id: str
    slate_games: int
    strategy: str
    my_actual: float
    baseline_proj: float
    win: float
    margin: float
    is_mirror: bool
    win_median: float
    margin_median: float
    winnings: float
    opponent: str


@dataclasses.dataclass
class OpponentResult:
    slate_id: str
    opponent_score: float
    my_score: float
    win: float
    margin: float
    is_mirror: bool
    opponent: str


# ----------------------------
# Helpers
# ----------------------------


def join_proj_results(
    proj: pd.DataFrame, results: pd.DataFrame, id_col: str = "player_key"
) -> pd.DataFrame:
    merged = proj.merge(
        results[[id_col, "FPTS"]], on=id_col, how="left", indicator=True
    )
    missing_actuals = merged[merged["_merge"] == "left_only"][id_col].tolist()
    if missing_actuals:
        logger.debug("Missing actuals for players: %s", missing_actuals)
    merged = merged.drop(columns=["_merge"])
    merged["FPTS"] = merged["FPTS"].fillna(0.0)
    return merged


def load_h2h_results(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)

    required = {"fee_1", "fee_2", "fee_3"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Candidate lineup file missing keys: {missing}")

    return payload


def load_candidate_lineups(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)

    required = {"slate_id", "top_fpts", "top_minutes"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Candidate lineup file missing keys: {missing}")

    return payload["top_fpts"], payload["top_minutes"]


def load_results_csv(path: str) -> pd.DataFrame:
    """
    Load post-slate results. Expected columns: id_col, actual_fpts.
    """
    raw = pd.read_csv(path, dtype=str)

    entries = []
    players = []

    for _, row in raw.iterrows():
        rank = row["Rank"]

        if pd.notna(rank) and rank.isdigit():
            entries.append(
                {
                    "Rank": int(row["Rank"]),
                    "EntryId": row["EntryId"],
                    "EntryName": row["EntryName"],
                    "TimeRemaining": row["TimeRemaining"],
                    "Points": float(row["Points"]),
                    "Lineup": row["Lineup"],
                }
            )

        if pd.notna(row["Player"]):
            players.append(
                {
                    "player_key": normalize_name(row["Player"]),
                    "Player": row["Player"],
                    "RosterPosition": row["Roster Position"],
                    "DraftedPct": row["%Drafted"],
                    "FPTS": float(row["FPTS"]),
                }
            )

    entries_df = pd.DataFrame(entries)
    players_df = pd.DataFrame(players)

    return entries_df, players_df


def lineup_from_player_keys(
    df: pd.DataFrame, player_keys: Set[str], id_col: str
) -> Lineup:
    player_keys = set(player_keys)
    df = df.copy()
    df[id_col] = df[id_col].map(normalize_name)

    lineup_df = df[df[id_col].isin(player_keys)]

    missing = player_keys - set(lineup_df[id_col])
    if missing:
        df.to_csv("test.csv")
        raise ValueError(f"Players not found: {missing}")

    players = frozenset(lineup_df[id_col].tolist())
    projected_fpts = float(lineup_df["proj_fpts"].sum())
    projected_minutes = float(lineup_df["proj_minutes"].sum())
    # rg_floor = float(lineup_df["floor"].sum())
    rg_floor = 0
    fragile_count = int((lineup_df["proj_minutes"] < 30).sum())
    actual_fpts = float(lineup_df["FPTS"].sum())
    return Lineup(
        players=players,
        projected_fpts=projected_fpts,
        projected_minutes=projected_minutes,
        rg_floor=rg_floor,
        fragile_count=fragile_count,
        total_fragile_minutes=total_fragile_minutes(lineup_df),
        actual_fpts=actual_fpts,
    )


# ----------------------------
# Candidate pool
# ----------------------------
def dedupe_lineups(lineups: List[Lineup]) -> List[Lineup]:
    seen = set()
    deduped = []
    for lu in lineups:
        if lu.players in seen:
            continue
        seen.add(lu.players)
        deduped.append(lu)
    return deduped


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_proj_lineups(
    slate_id: str,
    slate_games: int,
    top_fpts_lineups: List[Lineup],
    top_minutes_lineups: List[Lineup],
    baseline_proj: Lineup,
    proj_source: str,
) -> List[SlateResult]:
    results: List[SlateResult] = []

    top_10_fpts_lineups = top_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    # Max FPTS
    max_fpts_lineup = top_fpts_lineups[0]

    slate_result, _ = get_slate_result(
        baseline_proj,
        max_fpts_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max FPTs",
    )
    results.append(slate_result)

    # Max Minutes
    max_minutes_lineup = top_minutes_lineups[0]

    slate_result, _ = get_slate_result(
        baseline_proj,
        max_minutes_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max Mins",
    )
    results.append(slate_result)

    # Max Minutes within Top FPTs
    max_minutes_within_top_fpts_lineup = sorted(
        top_fpts_lineups,
        key=lambda lu: (lu.projected_minutes, lu.projected_fpts),
        reverse=True,
    )[0]

    slate_result, _ = get_slate_result(
        baseline_proj,
        max_minutes_within_top_fpts_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max Mins in Top FPTs",
    )
    results.append(slate_result)

    # Adjusted Fragile
    adjusted_fragile_lineup = sorted(
        top_fpts_lineups,
        key=lambda lu: (
            adjusted_score(
                lu.projected_fpts,
                lu.projected_minutes,
                lu.total_fragile_minutes,
                slate_games,
            )
        ),
        reverse=True,
    )[0]

    slate_result, _ = get_slate_result(
        baseline_proj,
        adjusted_fragile_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Adj Frag",
    )
    results.append(slate_result)

    # Top 10 median vs RG max FPTs
    win_vs_proj = 0.0

    if top_10_median_actual > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif top_10_median_actual < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy=f"{proj_source} median",
            my_actual=top_10_median_actual,
            baseline_proj=baseline_proj.actual_fpts,
            win=win_vs_proj,
            margin=top_10_median_actual - baseline_proj.actual_fpts,
            is_mirror=False,
            win_median=0.5,
            margin_median=0.0,
            winnings=0,
            opponent="",
        )
    )

    return results


def evaluate_h2h_lineups(
    slate_id: str,
    slate_games: int,
    top_fpts_lineups: List[Lineup],
    baseline_proj: Lineup,
    h2h_lineup_1: Lineup,
    h2h_lineup_2: Lineup,
    h2h_lineup_3: Lineup,
    h2h_opponent_1: str,
    h2h_opponent_2: str,
    h2h_opponent_3: str,
    proj_source: str,
) -> List[SlateResult]:
    results: List[SlateResult] = []
    opponent_results = []
    adjusted_fragile_wins = 0
    max_fpts_wins = 0

    top_10_fpts_lineups = top_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    # Adjusted Fragile
    adjusted_fragile_lineup = sorted(
        top_fpts_lineups,
        key=lambda lu: (
            adjusted_score(
                lu.projected_fpts,
                lu.projected_minutes,
                lu.total_fragile_minutes,
                slate_games,
            )
        ),
        reverse=True,
    )[0]

    # Opponent H2H $1 vs Adjusted Fragile
    slate_result, opponent_result = get_slate_result(
        adjusted_fragile_lineup,
        h2h_lineup_1,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $1 Adj Frag",
        fee=1,
        opponent=h2h_opponent_1,
    )
    results.append(slate_result)
    opponent_results.append(opponent_result)
    adjusted_fragile_wins += 1 - slate_result.win

    # Opponent H2H $2 vs Adjusted Fragile
    slate_result, opponent_result = get_slate_result(
        adjusted_fragile_lineup,
        h2h_lineup_2,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $2 Adj Frag",
        fee=2,
        opponent=h2h_opponent_2,
    )
    results.append(slate_result)
    if h2h_opponent_2 != h2h_opponent_1:
        opponent_results.append(opponent_result)
    adjusted_fragile_wins += 1 - slate_result.win

    # Opponent H2H $3 vs Adjusted Fragile
    slate_result, opponent_result = get_slate_result(
        adjusted_fragile_lineup,
        h2h_lineup_3,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $3 Adj Frag",
        fee=3,
        opponent=h2h_opponent_3,
    )
    results.append(slate_result)
    if h2h_opponent_3 != h2h_opponent_2 and h2h_opponent_3 != h2h_opponent_1:
        opponent_results.append(opponent_result)
    adjusted_fragile_wins += 1 - slate_result.win

    # Opponent H2H $1 vs Max FPTs
    slate_result, _ = get_slate_result(
        baseline_proj,
        h2h_lineup_1,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $1 Max FPTs",
        fee=1,
        opponent=h2h_opponent_1,
    )
    results.append(slate_result)
    max_fpts_wins += 1 - slate_result.win

    # Opponent H2H $2 vs Max FPTs
    slate_result, _ = get_slate_result(
        baseline_proj,
        h2h_lineup_2,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $2 Max FPTs",
        fee=2,
        opponent=h2h_opponent_2,
    )
    results.append(slate_result)
    max_fpts_wins += 1 - slate_result.win

    # Opponent H2H $3 vs Max FPTs
    slate_result, _ = get_slate_result(
        baseline_proj,
        h2h_lineup_3,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} H2H $3 Max FPTs",
        fee=3,
        opponent=h2h_opponent_3,
    )
    results.append(slate_result)
    max_fpts_wins += 1 - slate_result.win

    adjusted_fragile_win_rate = adjusted_fragile_wins / 3
    max_fpts_win_rate = max_fpts_wins / 3

    return results, opponent_results, adjusted_fragile_win_rate, max_fpts_win_rate


def get_slate_result(
    baseline_proj,
    lineup,
    top_10_median_actual,
    slate_id,
    slate_games,
    strategy,
    fee=1,
    opponent="",
):
    win_vs_proj = 0.0
    is_mirror = lineup.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif lineup.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif lineup.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if lineup.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif lineup.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    winnings = -fee

    if is_mirror:
        winnings = 0
    elif lineup.actual_fpts < baseline_proj.actual_fpts:
        winnings = fee * 0.8
    elif lineup.actual_fpts == baseline_proj.actual_fpts:
        winnings = -fee * 0.2

    opponent_result = OpponentResult(
        slate_id=slate_id,
        opponent_score=lineup.actual_fpts,
        my_score=baseline_proj.actual_fpts,
        win=win_vs_proj,
        margin=lineup.actual_fpts - baseline_proj.actual_fpts,
        is_mirror=is_mirror,
        opponent=opponent,
    )

    slate_result = SlateResult(
        slate_id=slate_id,
        slate_games=slate_games,
        strategy=strategy,
        my_actual=lineup.actual_fpts,
        baseline_proj=baseline_proj.actual_fpts,
        win=win_vs_proj,
        margin=lineup.actual_fpts - baseline_proj.actual_fpts,
        is_mirror=is_mirror,
        win_median=win_vs_top_10_median,
        margin_median=lineup.actual_fpts - top_10_median_actual,
        winnings=winnings,
        opponent=opponent,
    )

    return slate_result, opponent_result


def aggregate_results(results: List[SlateResult], opponent_results) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(dataclasses.asdict(r) for r in results)

    def bucket(games: int) -> str:
        if games <= 4:
            return "2-4"
        if games <= 8:
            return "5-8"
        return "9+"

    df["slate_bucket"] = df["slate_games"].map(bucket)
    df["non_mirror"] = 1 - df["is_mirror"]
    # Only count wins on non-mirrors; NaN for mirrors so they don't affect mean
    df["win_no_mirror"] = df.apply(
        lambda r: r["win"] if r["is_mirror"] == 0 else None,
        axis=1,
    )
    agg = (
        df.groupby(["strategy", "slate_bucket"])
        .agg(
            # Overall score vs RG (mirrors count as 0.5)
            win=("win", "mean"),
            # Edge win rate: only when you deviated
            win_no_mirror=("win_no_mirror", "mean"),
            # How often you mirrored RG
            mirror_rate=("is_mirror", "mean"),
            # Margins (still informative overall)
            margin=("margin", "mean"),
            # Overall score vs RG (mirrors count as 0.5)
            win_median=("win_median", "mean"),
            # Margins (still informative overall)
            margin_median=("margin_median", "mean"),
            slates=("slate_id", "nunique"),
            winnings=("winnings", "sum"),
        )
        .round(3)
        .reset_index()
    )
    opponents_df = pd.DataFrame(dataclasses.asdict(r) for r in opponent_results)
    opponents_df["non_mirror"] = 1 - opponents_df["is_mirror"]
    # Only count wins on non-mirrors; NaN for mirrors so they don't affect mean
    opponents_df["win_no_mirror"] = opponents_df.apply(
        lambda r: r["win"] if r["is_mirror"] == 0 else None,
        axis=1,
    )
    opponents = (
        opponents_df.groupby(["opponent"])
        .agg(
            # Overall score vs RG (mirrors count as 0.5)
            win=("win", "mean"),
            # Edge win rate: only when you deviated
            win_no_mirror=("win_no_mirror", "mean"),
            # How often you mirrored RG
            mirror_rate=("is_mirror", "mean"),
            # Margins (still informative overall)
            margin=("margin", "mean"),
            slates=("slate_id", "nunique"),
        )
        .round(3)
        .sort_values(by="win")
        .reset_index()
    )
    return agg, opponents


def evaluate_slate(
    blend_candidate_lineups_path: Path,
    etr_candidate_lineups_path: Path,
    etr_proj_path: Path,
    h2h_path: Path,
    results_path: Path,
    rg_candidate_lineups_path: Path,
    rg_proj_path: Path,
    id_col: str = "player_key",
) -> pd.DataFrame:
    """
    Example backtest workflow. `top_fpts_indices` and `top_minutes_indices` are lists of
    lineups represented by player indices into the projection dataframe.
    """
    rg_proj, slate_games, _ = load_projection_csv(rg_proj_path, remove_nan=False)
    etr_proj, _, _ = load_projection_csv(etr_proj_path, remove_nan=False)
    blend_proj = blend_projections(rg_proj, etr_proj)
    _, results_players_df = load_results_csv(results_path)
    rg_merged = join_proj_results(rg_proj, results_players_df, id_col=id_col)
    etr_merged = join_proj_results(etr_proj, results_players_df, id_col=id_col)
    blend_merged = join_proj_results(blend_proj, results_players_df, id_col=id_col)
    rg_top_fpts_player_keys, rg_top_minutes_player_keys = load_candidate_lineups(
        rg_candidate_lineups_path
    )
    etr_top_fpts_player_keys, etr_top_minutes_player_keys = load_candidate_lineups(
        etr_candidate_lineups_path
    )
    blend_top_fpts_player_keys, blend_top_minutes_player_keys = load_candidate_lineups(
        blend_candidate_lineups_path
    )
    h2h_results = load_h2h_results(h2h_path)

    # Candidate pool: top N by fpts and minutes, then dedupe by player set.
    blend_top_fpts_lineups = [
        lineup_from_player_keys(blend_merged, player_keys, id_col=id_col)
        for player_keys in blend_top_fpts_player_keys
    ]
    blend_top_minutes_lineups = [
        lineup_from_player_keys(blend_merged, player_keys, id_col=id_col)
        for player_keys in blend_top_minutes_player_keys
    ]
    etr_top_fpts_lineups = [
        lineup_from_player_keys(etr_merged, player_keys, id_col=id_col)
        for player_keys in etr_top_fpts_player_keys
    ]
    etr_top_minutes_lineups = [
        lineup_from_player_keys(etr_merged, player_keys, id_col=id_col)
        for player_keys in etr_top_minutes_player_keys
    ]
    rg_top_fpts_lineups = [
        lineup_from_player_keys(rg_merged, player_keys, id_col=id_col)
        for player_keys in rg_top_fpts_player_keys
    ]
    rg_top_minutes_lineups = [
        lineup_from_player_keys(rg_merged, player_keys, id_col=id_col)
        for player_keys in rg_top_minutes_player_keys
    ]

    h2h_lineup_1 = lineup_from_player_keys(
        rg_merged,
        [normalize_name(name) for name in h2h_results["fee_1"]["lineup"].values()],
        id_col=id_col,
    )
    h2h_opponent_1 = h2h_results["fee_1"]["entry_name"]

    if h2h_lineup_1.actual_fpts != h2h_results["fee_1"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_1.actual_fpts}, does not equal recorded score {h2h_results['fee_1']['points']}"
        )

    h2h_lineup_2 = lineup_from_player_keys(
        rg_merged,
        [normalize_name(name) for name in h2h_results["fee_2"]["lineup"].values()],
        id_col=id_col,
    )
    h2h_opponent_2 = h2h_results["fee_2"]["entry_name"]

    if h2h_lineup_2.actual_fpts != h2h_results["fee_2"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_2.actual_fpts}, does not equal recorded score {h2h_results['fee_2']['points']}"
        )

    h2h_lineup_3 = lineup_from_player_keys(
        rg_merged,
        [normalize_name(name) for name in h2h_results["fee_3"]["lineup"].values()],
        id_col=id_col,
    )
    h2h_opponent_3 = h2h_results["fee_3"]["entry_name"]

    if h2h_lineup_3.actual_fpts != h2h_results["fee_3"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_3.actual_fpts}, does not equal recorded score {h2h_results['fee_3']['points']}"
        )

    # Baseline: RG max projection lineup (first from top_fpts_indices).
    baseline_proj = rg_top_fpts_lineups[0]

    slate_results = evaluate_proj_lineups(
        slate_id=Path(rg_proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=rg_top_fpts_lineups,
        top_minutes_lineups=rg_top_minutes_lineups,
        baseline_proj=baseline_proj,
        proj_source="RG",
    )

    slate_results += evaluate_proj_lineups(
        slate_id=Path(rg_proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=etr_top_fpts_lineups,
        top_minutes_lineups=etr_top_minutes_lineups,
        baseline_proj=baseline_proj,
        proj_source="ETR",
    )

    slate_results += evaluate_proj_lineups(
        slate_id=Path(rg_proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=blend_top_fpts_lineups,
        top_minutes_lineups=blend_top_minutes_lineups,
        baseline_proj=baseline_proj,
        proj_source="BLEND",
    )

    win_rates = {}

    results, opponent_results, adjusted_fragile_win_rate, max_fpts_win_rate = (
        evaluate_h2h_lineups(
            slate_id=Path(rg_proj_path).stem,
            slate_games=slate_games,
            top_fpts_lineups=rg_top_fpts_lineups,
            baseline_proj=baseline_proj,
            h2h_lineup_1=h2h_lineup_1,
            h2h_lineup_2=h2h_lineup_2,
            h2h_lineup_3=h2h_lineup_3,
            h2h_opponent_1=h2h_opponent_1,
            h2h_opponent_2=h2h_opponent_2,
            h2h_opponent_3=h2h_opponent_3,
            proj_source="RG",
        )
    )
    slate_results += results
    win_rates["rg_adjusted_fragile"] = adjusted_fragile_win_rate
    win_rates["rg_max_fpts"] = max_fpts_win_rate

    results, opp_results, adjusted_fragile_win_rate, max_fpts_win_rate = (
        evaluate_h2h_lineups(
            slate_id=Path(rg_proj_path).stem,
            slate_games=slate_games,
            top_fpts_lineups=etr_top_fpts_lineups,
            baseline_proj=etr_top_fpts_lineups[0],
            h2h_lineup_1=h2h_lineup_1,
            h2h_lineup_2=h2h_lineup_2,
            h2h_lineup_3=h2h_lineup_3,
            h2h_opponent_1=h2h_opponent_1,
            h2h_opponent_2=h2h_opponent_2,
            h2h_opponent_3=h2h_opponent_3,
            proj_source="ETR",
        )
    )
    slate_results += results
    opponent_results += opp_results
    win_rates["etr_adjusted_fragile"] = adjusted_fragile_win_rate
    win_rates["etr_max_fpts"] = max_fpts_win_rate

    results, _, adjusted_fragile_win_rate, max_fpts_win_rate = evaluate_h2h_lineups(
        slate_id=Path(rg_proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=blend_top_fpts_lineups,
        baseline_proj=blend_top_fpts_lineups[0],
        h2h_lineup_1=h2h_lineup_1,
        h2h_lineup_2=h2h_lineup_2,
        h2h_lineup_3=h2h_lineup_3,
        h2h_opponent_1=h2h_opponent_1,
        h2h_opponent_2=h2h_opponent_2,
        h2h_opponent_3=h2h_opponent_3,
        proj_source="BLEND",
    )
    slate_results += results
    win_rates["blend_adjusted_fragile"] = adjusted_fragile_win_rate
    win_rates["blend_max_fpts"] = max_fpts_win_rate

    return slate_results, opponent_results, win_rates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")

    blend_candidate_dir = data_dir / "candidate_lineups" / "blend"
    etr_candidate_dir = data_dir / "candidate_lineups" / "etr"
    etr_proj_dir = data_dir / "raw" / "etr"
    h2h_dir = data_dir / "processed" / "h2h"
    results_dir = data_dir / "raw" / "history"
    rg_candidate_dir = data_dir / "candidate_lineups" / "rotogrinders"
    rg_proj_dir = data_dir / "raw" / "rotogrinders"

    slates = []

    for results_file in results_dir.iterdir():
        if not results_file.is_file():
            continue

        meta = parse_filename(results_file.stem)

        if meta.sport != "nba":
            continue

        slates.append(meta)

    n_slates = len(slates)

    slates.sort(key=lambda x: x.date)

    slate_results = []
    opponent_results = []
    win_rates = {
        "blend_adjusted_fragile": 0,
        "blend_max_fpts": 0,
        "etr_adjusted_fragile": 0,
        "etr_max_fpts": 0,
        "rg_adjusted_fragile": 0,
        "rg_max_fpts": 0,
    }

    for i, slate in enumerate(slates):
        logging.info(f"[{i + 1}/{n_slates}] {slate.slate_id}")

        blend_candidate_lineups_path = (
            blend_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )
        etr_candidate_lineups_path = (
            etr_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )
        etr_proj_path = (
            etr_proj_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.date}.csv"
        )
        h2h_path = (
            h2h_dir / f"{slate.sport}_{slate.slate}_{slate.site}_h2h_{slate.date}.json"
        )
        rg_proj_path = (
            rg_proj_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.date}.csv"
        )
        rg_candidate_lineups_path = (
            rg_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )
        results_path = (
            results_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_results_{slate.date}.csv"
        )

        results, opp_results, slate_win_rates = evaluate_slate(
            blend_candidate_lineups_path=blend_candidate_lineups_path,
            etr_candidate_lineups_path=etr_candidate_lineups_path,
            etr_proj_path=etr_proj_path,
            h2h_path=h2h_path,
            results_path=results_path,
            rg_candidate_lineups_path=rg_candidate_lineups_path,
            rg_proj_path=rg_proj_path,
        )

        slate_results += results
        opponent_results += opp_results
        win_rates["blend_adjusted_fragile"] += slate_win_rates["blend_adjusted_fragile"]
        win_rates["blend_max_fpts"] += slate_win_rates["blend_max_fpts"]
        win_rates["etr_adjusted_fragile"] += slate_win_rates["etr_adjusted_fragile"]
        win_rates["etr_max_fpts"] += slate_win_rates["etr_max_fpts"]
        win_rates["rg_adjusted_fragile"] += slate_win_rates["rg_adjusted_fragile"]
        win_rates["rg_max_fpts"] += slate_win_rates["rg_max_fpts"]

    win_rates["blend_adjusted_fragile"] = round(
        win_rates["blend_adjusted_fragile"] / n_slates * 100, 3
    )
    win_rates["blend_max_fpts"] = round(win_rates["blend_max_fpts"] / n_slates * 100, 3)
    win_rates["etr_adjusted_fragile"] = round(
        win_rates["etr_adjusted_fragile"] / n_slates * 100, 3
    )
    win_rates["etr_max_fpts"] = round(win_rates["etr_max_fpts"] / n_slates * 100, 3)
    win_rates["rg_adjusted_fragile"] = round(
        win_rates["rg_adjusted_fragile"] / n_slates * 100, 3
    )
    win_rates["rg_max_fpts"] = round(win_rates["rg_max_fpts"] / n_slates * 100, 3)

    print("=======================================================================")
    print("Win Rates")
    print("-----------------------------------------------------------------------")
    print(f"BLEND Adjusted Fragile: {win_rates['blend_adjusted_fragile']}%")
    print(f"BLEND Max FPTs: {win_rates['blend_max_fpts']}%")
    print(f"ETR Adjusted Fragile: {win_rates['etr_adjusted_fragile']}%")
    print(f"ETR Max FPTs: {win_rates['etr_max_fpts']}%")
    print(f"RG Adjusted Fragile: {win_rates['rg_adjusted_fragile']}%")
    print(f"RG Max FPTs: {win_rates['rg_max_fpts']}%")
    print("=======================================================================")

    summary_df, opponents_df = aggregate_results(slate_results, opponent_results)
    summary_df.to_csv("data/processed/backtest.csv")
    opponents_df.to_csv("data/processed/opponents.csv")
