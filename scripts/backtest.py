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
    win_vs_proj: float
    margin_vs_proj: float
    is_mirror: bool
    win_vs_top_10_median: float
    margin_vs_top_10_median: float


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
    df = pd.read_csv(path)
    required = {"Player", "FPTS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results file missing columns: {missing}")

    df["player_key"] = df["Player"].map(normalize_name)

    return df


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
    rg_floor = float(lineup_df["floor"].sum())
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
def evaluate_lineups(
    slate_id: str,
    slate_games: int,
    top_fpts_lineups: List[Lineup],
    top_minutes_lineups: List[Lineup],
    baseline_proj: Lineup,
    h2h_lineup_1: Lineup,
    h2h_lineup_2: Lineup,
    h2h_lineup_3: Lineup,
) -> List[SlateResult]:
    results: List[SlateResult] = []

    top_10_fpts_lineups = top_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    max_minutes_lineup = top_minutes_lineups[0]
    win_vs_proj = 0.0
    is_mirror = max_minutes_lineup.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif max_minutes_lineup.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif max_minutes_lineup.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if max_minutes_lineup.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif max_minutes_lineup.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    # for i, minutes_lineup in enumerate(top_minutes_lineups):
    #     results.append(
    #         SlateResult(
    #             slate_id=f"{slate_id}_{i}",
    #             slate_games=slate_games,
    #             strategy="max_minutes",
    #             my_actual=minutes_lineup.actual_fpts,
    #             baseline_proj=baseline_proj.actual_fpts,
    #             win_vs_proj=int(
    #                 minutes_lineup.actual_fpts >= baseline_proj.actual_fpts
    #             ),
    #             margin_vs_proj=minutes_lineup.actual_fpts - baseline_proj.actual_fpts,
    #         )
    #     )

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="max_minutes",
            my_actual=max_minutes_lineup.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=max_minutes_lineup.actual_fpts - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=max_minutes_lineup.actual_fpts
            - top_10_median_actual,
        )
    )

    max_minutes_within_top_fpts_lineup = sorted(
        top_fpts_lineups,
        key=lambda lu: (lu.projected_minutes, lu.projected_fpts),
        reverse=True,
    )[0]
    win_vs_proj = 0.0
    is_mirror = max_minutes_within_top_fpts_lineup.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif max_minutes_within_top_fpts_lineup.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif max_minutes_within_top_fpts_lineup.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if max_minutes_within_top_fpts_lineup.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif max_minutes_within_top_fpts_lineup.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    # for i in range(10):
    #     lineup = max_minutes_within_top_fpts_lineups[i]
    #     results.append(
    #         SlateResult(
    #             slate_id=f"{slate_id}_{i}",
    #             slate_games=slate_games,
    #             strategy="max_minutes_within_top_fpts",
    #             my_actual=lineup.actual_fpts,
    #             baseline_proj=baseline_proj.actual_fpts,
    #             win_vs_proj=int(lineup.actual_fpts >= baseline_proj.actual_fpts),
    #             margin_vs_proj=lineup.actual_fpts - baseline_proj.actual_fpts,
    #         )
    #     )

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="max_minutes_within_top_fpts",
            my_actual=max_minutes_within_top_fpts_lineup.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=max_minutes_within_top_fpts_lineup.actual_fpts
            - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=max_minutes_within_top_fpts_lineup.actual_fpts
            - top_10_median_actual,
        )
    )

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
    win_vs_proj = 0.0
    is_mirror = adjusted_fragile_lineup.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif adjusted_fragile_lineup.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif adjusted_fragile_lineup.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if adjusted_fragile_lineup.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif adjusted_fragile_lineup.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    # for i in range(10):
    #     lineup = adjusted_fragile_lineups[i]
    #     results.append(
    #         SlateResult(
    #             slate_id=f"{slate_id}_{i}",
    #             slate_games=slate_games,
    #             strategy="adjusted_fragile",
    #             my_actual=lineup.actual_fpts,
    #             baseline_proj=baseline_proj.actual_fpts,
    #             win_vs_proj=int(lineup.actual_fpts >= baseline_proj.actual_fpts),
    #             margin_vs_proj=lineup.actual_fpts - baseline_proj.actual_fpts,
    #         )
    #     )

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="adjusted_fragile",
            my_actual=adjusted_fragile_lineup.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=adjusted_fragile_lineup.actual_fpts
            - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=adjusted_fragile_lineup.actual_fpts
            - top_10_median_actual,
        )
    )

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
            strategy="top_10_median (Uses actual scores)",
            my_actual=top_10_median_actual,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=top_10_median_actual - baseline_proj.actual_fpts,
            is_mirror=False,
            win_vs_top_10_median=0.5,
            margin_vs_top_10_median=0.0,
        )
    )

    # Opponent H2H $1
    win_vs_proj = 0.0
    is_mirror = h2h_lineup_1.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif h2h_lineup_1.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif h2h_lineup_1.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if h2h_lineup_1.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif h2h_lineup_1.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="Opponent H2H $1",
            my_actual=h2h_lineup_1.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=h2h_lineup_1.actual_fpts - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=h2h_lineup_1.actual_fpts - top_10_median_actual,
        )
    )

    # Opponent H2H $2
    win_vs_proj = 0.0
    is_mirror = h2h_lineup_2.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif h2h_lineup_2.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif h2h_lineup_2.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if h2h_lineup_2.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif h2h_lineup_2.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="Opponent H2H $2",
            my_actual=h2h_lineup_2.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=h2h_lineup_2.actual_fpts - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=h2h_lineup_2.actual_fpts - top_10_median_actual,
        )
    )

    # Opponent H2H $3
    win_vs_proj = 0.0
    is_mirror = h2h_lineup_3.players == baseline_proj.players

    if is_mirror:
        win_vs_proj = 0.5  # tie
    elif h2h_lineup_3.actual_fpts > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif h2h_lineup_3.actual_fpts < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if h2h_lineup_3.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif h2h_lineup_3.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    results.append(
        SlateResult(
            slate_id=slate_id,
            slate_games=slate_games,
            strategy="Opponent H2H $3",
            my_actual=h2h_lineup_3.actual_fpts,
            baseline_proj=baseline_proj.actual_fpts,
            win_vs_proj=win_vs_proj,
            margin_vs_proj=h2h_lineup_3.actual_fpts - baseline_proj.actual_fpts,
            is_mirror=is_mirror,
            win_vs_top_10_median=win_vs_top_10_median,
            margin_vs_top_10_median=h2h_lineup_3.actual_fpts - top_10_median_actual,
        )
    )

    return results


def aggregate_results(results: List[SlateResult]) -> pd.DataFrame:
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
    df["win_vs_proj_non_mirror"] = df.apply(
        lambda r: r["win_vs_proj"] if r["is_mirror"] == 0 else None,
        axis=1,
    )
    agg = (
        df.groupby(["strategy", "slate_bucket"])
        .agg(
            # Overall score vs RG (mirrors count as 0.5)
            win_rate_vs_proj=("win_vs_proj", "mean"),
            # Edge win rate: only when you deviated
            win_rate_vs_proj_non_mirror=("win_vs_proj_non_mirror", "mean"),
            # How often you mirrored RG
            mirror_rate=("is_mirror", "mean"),
            # Margins (still informative overall)
            avg_margin_vs_proj=("margin_vs_proj", "mean"),
            # Overall score vs RG (mirrors count as 0.5)
            win_rate_vs_top_10_median=("win_vs_top_10_median", "mean"),
            # Margins (still informative overall)
            avg_margin_vs_top_10_median=("margin_vs_top_10_median", "mean"),
            slates=("slate_id", "nunique"),
        )
        .reset_index()
    )
    return agg


def evaluate_slate(
    candidate_lineups_path: Path,
    h2h_path: Path,
    proj_path: Path,
    results_path: Path,
    id_col: str = "player_key",
) -> pd.DataFrame:
    """
    Example backtest workflow. `top_fpts_indices` and `top_minutes_indices` are lists of
    lineups represented by player indices into the projection dataframe.
    """
    proj, slate_games, _ = load_projection_csv(proj_path, remove_nan=False)
    results = load_results_csv(results_path)
    merged = join_proj_results(proj, results, id_col=id_col)
    top_fpts_player_keys, top_minutes_player_keys = load_candidate_lineups(
        candidate_lineups_path
    )
    h2h_results = load_h2h_results(h2h_path)

    # Candidate pool: top N by fpts and minutes, then dedupe by player set.
    top_fpts_lineups = [
        lineup_from_player_keys(merged, player_keys, id_col=id_col)
        for player_keys in top_fpts_player_keys
    ]
    top_minutes_lineups = [
        lineup_from_player_keys(merged, player_keys, id_col=id_col)
        for player_keys in top_minutes_player_keys
    ]

    h2h_lineup_1 = lineup_from_player_keys(
        merged,
        [normalize_name(name) for name in h2h_results["fee_1"]["lineup"].values()],
        id_col=id_col,
    )

    if h2h_lineup_1.actual_fpts != h2h_results["fee_1"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_1.actual_fpts}, does not equal recorded score {h2h_results['fee_1']['points']}"
        )

    h2h_lineup_2 = lineup_from_player_keys(
        merged,
        [normalize_name(name) for name in h2h_results["fee_2"]["lineup"].values()],
        id_col=id_col,
    )

    if h2h_lineup_2.actual_fpts != h2h_results["fee_2"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_2.actual_fpts}, does not equal recorded score {h2h_results['fee_2']['points']}"
        )

    h2h_lineup_3 = lineup_from_player_keys(
        merged,
        [normalize_name(name) for name in h2h_results["fee_3"]["lineup"].values()],
        id_col=id_col,
    )

    if h2h_lineup_3.actual_fpts != h2h_results["fee_3"]["points"]:
        raise ValueError(
            f"Calculated H2H lineup score, {h2h_lineup_3.actual_fpts}, does not equal recorded score {h2h_results['fee_3']['points']}"
        )

    # Baseline: RG max projection lineup (first from top_fpts_indices).
    baseline_proj = top_fpts_lineups[0]

    slate_results = evaluate_lineups(
        slate_id=Path(proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=top_fpts_lineups,
        top_minutes_lineups=top_minutes_lineups,
        baseline_proj=baseline_proj,
        h2h_lineup_1=h2h_lineup_1,
        h2h_lineup_2=h2h_lineup_2,
        h2h_lineup_3=h2h_lineup_3,
    )

    return slate_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    proj_dir = data_dir / "raw" / "rotogrinders"
    results_dir = data_dir / "raw" / "history"
    candidate_dir = data_dir / "candidate_lineups" / "rotogrinders"
    h2h_dir = data_dir / "processed" / "h2h"

    slate_results = []

    for results_file in results_dir.iterdir():
        meta = parse_filename(results_file.stem)
        print(meta.slate_id)

        if meta.sport != "nba":
            continue
        proj_path = (
            proj_dir
            / f"{meta.sport}_{meta.slate}_{meta.site}_rg_projections_{meta.date}.csv"
        )
        candidate_lineups_path = (
            candidate_dir
            / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.date}.json"
        )
        h2h_path = (
            h2h_dir / f"{meta.sport}_{meta.slate}_{meta.site}_h2h_{meta.date}.json"
        )

        slate_results += evaluate_slate(
            candidate_lineups_path=candidate_lineups_path,
            h2h_path=h2h_path,
            proj_path=proj_path,
            results_path=results_file,
        )
    summary_df = aggregate_results(slate_results)
    summary_df.to_csv("data/processed/backtest.csv")
