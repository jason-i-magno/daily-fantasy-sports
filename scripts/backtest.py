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
from typing import Iterable, List

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
        logger.warning("Missing actuals for players: %s", missing_actuals)
    merged = merged.drop(columns=["_merge"])
    merged["FPTS"] = merged["FPTS"].fillna(0.0)
    return merged


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


def lineup_from_indices(
    df: pd.DataFrame, indices: Iterable[int], id_col: str
) -> Lineup:
    sub = df.loc[list(indices)]
    players = frozenset(sub[id_col].tolist())
    projected_fpts = float(sub["proj_fpts"].sum())
    projected_minutes = float(sub["proj_minutes"].sum())
    rg_floor = float(sub["floor"].sum())
    fragile_count = int((sub["proj_minutes"] < 28).sum())
    actual_fpts = float(sub["FPTS"].sum())
    return Lineup(
        players=players,
        projected_fpts=projected_fpts,
        projected_minutes=projected_minutes,
        rg_floor=rg_floor,
        fragile_count=fragile_count,
        total_fragile_minutes=total_fragile_minutes(sub),
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
    proj_path: str,
    results_path: str,
    candidate_lineups_path: str,
    id_col: str = "player_key",
) -> pd.DataFrame:
    """
    Example backtest workflow. `top_fpts_indices` and `top_minutes_indices` are lists of
    lineups represented by player indices into the projection dataframe.
    """
    proj, slate_games, _ = load_projection_csv(proj_path)
    results = load_results_csv(results_path)
    merged = join_proj_results(proj, results, id_col=id_col)
    top_fpts_indices, top_minutes_indices = load_candidate_lineups(
        candidate_lineups_path
    )

    # Candidate pool: top N by fpts and minutes, then dedupe by player set.
    top_fpts_lineups = [
        lineup_from_indices(merged, idxs, id_col=id_col) for idxs in top_fpts_indices
    ]
    top_minutes_lineups = [
        lineup_from_indices(merged, idxs, id_col=id_col) for idxs in top_minutes_indices
    ]

    # Baseline: RG max projection lineup (first from top_fpts_indices).
    baseline_proj = top_fpts_lineups[0]

    slate_results = evaluate_lineups(
        slate_id=Path(proj_path).stem,
        slate_games=slate_games,
        top_fpts_lineups=top_fpts_lineups,
        top_minutes_lineups=top_minutes_lineups,
        baseline_proj=baseline_proj,
    )

    return slate_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    proj_dir = data_dir / "raw" / "rotogrinders"
    results_dir = data_dir / "raw" / "history"
    candidate_dir = data_dir / "candidate_lineups"

    slate_results = []

    for results_file in results_dir.iterdir():
        meta = parse_filename(results_file.stem)

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

        slate_results += evaluate_slate(
            proj_path=proj_path,
            results_path=results_file,
            candidate_lineups_path=candidate_lineups_path,
        )
    summary_df = aggregate_results(slate_results)
    print(summary_df)
    summary_df.to_csv("data/processed/backtest.csv")
