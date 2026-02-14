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
import hashlib
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta
from utils import (
    CANDIDATE_DIR_MAP,
    ETR_PROJ_DIR,
    H2H_DIR,
    RG_PROJ_DIR,
    SLOT_ORDER,
    SLOTS,
    ResultsFileMeta,
    SlateMeta,
    adjusted_score,
    blend_projections,
    get_slates,
    load_dk_salaries_csv,
    load_projections,
    load_results_csv,
    normalize_name,
    total_fragile_minutes,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Constants
# ----------------------------
MODELS = [
    "blend",
    "etr",
    "rg",
]

FILTER_TYPES = [
    "all",
    "no_late_swaps",
    "missing_late_swaps",
    "late_swaps",
    "fee_1",
    "fee_2",
    "fee_3",
]


STRATEGIES = [
    "adjusted_fragile_minutes_floor",
    "adjusted_fragile",
    "max_fpts",
]


# ----------------------------
# Data structures
# ----------------------------
@dataclasses.dataclass
class AggregateStrategyMetrics:
    win_rate: float = 0.0
    win_rate_no_mirror: float = 0.0
    winnings: float = 0.0
    winnings_no_mirror: float = 0.0
    n_slates: int = 0
    n_slates_no_mirror: int = 0


@dataclasses.dataclass
class AggregateFilterMetrics:
    all: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    no_late_swaps: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    missing_late_swaps: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    late_swaps: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    fee_1: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    fee_2: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    fee_3: AggregateStrategyMetrics = dataclasses.field(
        default_factory=AggregateStrategyMetrics
    )
    by_slate: dict[str, AggregateStrategyMetrics] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class AggregateLineupMetrics:
    adjusted_fragile_minutes_floor: AggregateFilterMetrics = dataclasses.field(
        default_factory=AggregateFilterMetrics
    )
    adjusted_fragile: AggregateFilterMetrics = dataclasses.field(
        default_factory=AggregateFilterMetrics
    )
    max_fpts: AggregateFilterMetrics = dataclasses.field(
        default_factory=AggregateFilterMetrics
    )


@dataclasses.dataclass
class AllSlatesEvaluation:
    blend: AggregateLineupMetrics = dataclasses.field(
        default_factory=AggregateLineupMetrics
    )
    etr: AggregateLineupMetrics = dataclasses.field(
        default_factory=AggregateLineupMetrics
    )
    rg: AggregateLineupMetrics = dataclasses.field(
        default_factory=AggregateLineupMetrics
    )


@dataclasses.dataclass
class ContestResult:
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
    opponent: str


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
class H2HResults:
    lineup_1: Lineup = None
    lineup_2: Lineup = None
    lineup_3: Lineup = None
    opponent_1: str = ""
    opponent_2: str = ""
    opponent_3: str = ""


@dataclasses.dataclass
class StrategyResult:
    win_rate: float = 0.0
    win_rate_no_mirror: float | None = None
    winnings: float = 0.0
    winnings_no_mirror: float | None = None
    win_rate_by_fee: dict[int, float] = dataclasses.field(default_factory=dict)
    win_rate_no_mirror_by_fee: dict[int, float] = dataclasses.field(
        default_factory=dict
    )
    winnings_by_fee: dict[int, float] = dataclasses.field(default_factory=dict)
    winnings_no_mirror_by_fee: dict[int, float] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class LineupResult:
    adjusted_fragile_minutes_floor: StrategyResult = dataclasses.field(
        default_factory=StrategyResult
    )
    adjusted_fragile: StrategyResult = dataclasses.field(default_factory=StrategyResult)
    max_fpts: StrategyResult = dataclasses.field(default_factory=StrategyResult)


@dataclasses.dataclass
class OpponentResult:
    slate_id: str
    opponent_score: float
    my_score: float
    win: float
    margin: float
    is_mirror: bool
    opponent: str


@dataclasses.dataclass
class SlateResult:
    blend: LineupResult = dataclasses.field(default_factory=LineupResult)
    etr: LineupResult = dataclasses.field(default_factory=LineupResult)
    rg: LineupResult = dataclasses.field(default_factory=LineupResult)
    has_late_swaps: bool = True
    missing_late_swap_proj: bool = False
    slate_games: int = 0
    slate: str = ""


@dataclasses.dataclass
class TopLineups:
    blend_fpts: List[Lineup] = dataclasses.field(default_factory=list)
    blend_minutes: List[Lineup] = dataclasses.field(default_factory=list)
    blend_fpts_minutes_floor: List[Lineup] = dataclasses.field(default_factory=list)
    etr_fpts: List[Lineup] = dataclasses.field(default_factory=list)
    etr_minutes: List[Lineup] = dataclasses.field(default_factory=list)
    etr_fpts_minutes_floor: List[Lineup] = dataclasses.field(default_factory=list)
    rg_fpts: List[Lineup] = dataclasses.field(default_factory=list)
    rg_minutes: List[Lineup] = dataclasses.field(default_factory=list)
    rg_fpts_minutes_floor: List[Lineup] = dataclasses.field(default_factory=list)


# ----------------------------
# Helpers
# ----------------------------
def calculate_winnings(fee, win, is_mirror):
    winnings = -fee

    if is_mirror:
        winnings = 0
    elif win == 1:
        winnings = fee * 0.8
    elif win == 0.5:
        winnings = -fee * 0.2

    return winnings


def extract_game_id(game_info: str) -> str:
    if not isinstance(game_info, str) or not game_info.strip():
        return ""
    return game_info.split()[0]


def join_proj_results(proj: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    lookup = results.set_index("player_key")

    # Merge in FPTScolumn
    proj["FPTS"] = proj["player_key"].map(lookup["FPTS"])

    return proj


def lineup_from_keys(player_keys: list[str], dk_salaries: pd.DataFrame):
    """
    Convert 8 player_keys into a DataFrame compatible with validate_lineup().
    Requires full_df to contain player metadata.
    """
    if len(player_keys) != 8:
        raise ValueError("Lineup must contain exactly 8 players.")

    # Map each player_key → slot
    lookup = dk_salaries.set_index("player_key")
    df_lineup = pd.DataFrame({"player_key": player_keys, "slot": SLOT_ORDER}).assign(
        salary=lambda df: df["player_key"].map(lookup["salary"]),
        team=lambda df: df["player_key"].map(lookup["team"]),
        game_info=lambda df: df["player_key"].map(lookup["game_info"]),
        positions=lambda df: df["player_key"].map(lookup["positions"]),
    )

    missing = df_lineup[df_lineup.isna().any(axis=1)]
    if not missing.empty:
        print(df_lineup.to_string())
        raise ValueError(f"Unknown players in lineup: {missing['player_key'].tolist()}")

    return df_lineup


def load_h2h_results(slate: ResultsFileMeta, results_df: pd.DataFrame) -> H2HResults:
    h2h_path = (
        H2H_DIR / f"{slate.sport}_{slate.slate}_{slate.site}_h2h_{slate.datetime}.json"
    )

    with open(h2h_path, "r") as f:
        payload = json.load(f)

    required_fee_keys = {"fee_1", "fee_2", "fee_3"}
    missing = required_fee_keys - payload.keys()
    if missing:
        raise ValueError(f"Candidate lineup file missing keys: {missing}")

    h2h_results = H2HResults()

    for fee in range(1, 4):
        fee_payload = payload[f"fee_{fee}"]
        lineup = lineup_from_player_keys(
            results_df,
            [normalize_name(name) for name in fee_payload["lineup"].values()],
        )
        setattr(
            h2h_results,
            f"lineup_{fee}",
            lineup,
        )
        setattr(h2h_results, f"opponent_{fee}", fee_payload["entry_name"])

        if lineup.actual_fpts != fee_payload["points"]:
            raise ValueError(
                f"Calculated H2H lineup score, {lineup.actual_fpts}, does not equal recorded score {fee_payload['points']}"
            )

    return h2h_results


def load_candidate_lineups(
    meta: SlateMeta,
    proj_source: str,
    proj: pd.DataFrame,
    dk_salaries: pd.DataFrame,
    use_late_swaps: bool,
) -> Tuple[List[Lineup], List[Lineup], List[Lineup]]:
    candidate_lineups_path = (
        CANDIDATE_DIR_MAP[proj_source]
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
    )

    if use_late_swaps:
        game_times = sorted(
            dk_salaries.set_index("player_key")["game_time_local"].unique()
        )
        candidate_lineups_path = (
            CANDIDATE_DIR_MAP[proj_source]
            / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}T{game_times[-1].strftime('%H%M')}.json"
        )

    with open(candidate_lineups_path, "r") as f:
        payload = json.load(f)

    required = {"slate_id", "max_fpts", "max_minutes", "max_fpts_minutes_floor"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Candidate lineup file missing keys: {missing}")

    n_lineups = 1

    max_fpts_keys = payload["max_fpts"][:n_lineups]
    max_minutes_keys = payload["max_minutes"][:n_lineups]
    max_fpts_minutes_floor_keys = payload["max_fpts_minutes_floor"][:n_lineups]

    for i, lineups in enumerate(
        [max_fpts_keys, max_minutes_keys, max_fpts_minutes_floor_keys]
    ):
        for lineup in lineups:
            lineup_df = lineup_from_keys(lineup, dk_salaries)
            errors = validate_lineup(lineup_df, dk_salaries)

            if errors:
                print(i, proj_source)
                print(lineup)
                for error in errors:
                    logging.error(error)

    max_fpts_lineups = [
        lineup_from_player_keys(proj, player_keys) for player_keys in max_fpts_keys
    ]
    max_minutes_lineups = [
        lineup_from_player_keys(proj, player_keys) for player_keys in max_minutes_keys
    ]
    max_fpts_minutes_floor_lineups = [
        lineup_from_player_keys(proj, player_keys)
        for player_keys in max_fpts_minutes_floor_keys
    ]

    return (
        max_fpts_lineups,
        max_minutes_lineups,
        max_fpts_minutes_floor_lineups,
    )


def load_candidates(
    meta: SlateMeta, blend_proj, etr_proj, rg_proj, dk_salaries, use_late_swaps
) -> TopLineups:
    lineups = TopLineups()
    (
        lineups.rg_fpts,
        lineups.rg_minutes,
        lineups.rg_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "rg", rg_proj, dk_salaries, use_late_swaps)
    (
        lineups.etr_fpts,
        lineups.etr_minutes,
        lineups.etr_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "etr", etr_proj, dk_salaries, use_late_swaps)
    (
        lineups.blend_fpts,
        lineups.blend_minutes,
        lineups.blend_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "blend", blend_proj, dk_salaries, use_late_swaps)

    return lineups


def lineup_from_player_keys(
    df: pd.DataFrame, player_keys: Set[str], id_col: str = "player_key"
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

    if lineup_df["FPTS"].isna().any():
        print(lineup_df.to_string())
        raise ValueError("Lineup contains NaN FPTS values")
        # logging.warning("Lineup contains NaN FPTS values")

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


def print_evaluation(
    eval_by_bucket: dict[str, AllSlatesEvaluation],
    slate_values: dict[tuple[str, str, str, str, bool], list[float]],
    breakeven: float = 0.56,
    n_boot: int = 20000,
):
    for bucket, eval in sorted(eval_by_bucket.items()):
        print("\n==============================")
        print(f"Slate bucket: {bucket}")
        print("==============================")

        for model in MODELS:
            model_metrics = getattr(eval, model)
            print(f"=== {model.upper()} ===")
            for strategy in STRATEGIES:
                strategy_metrics = getattr(model_metrics, strategy)
                filter_types = list(FILTER_TYPES) + sorted(
                    strategy_metrics.by_slate.keys()
                )
                for filter_type in filter_types:
                    if filter_type in strategy_metrics.by_slate:
                        filter_metrics = strategy_metrics.by_slate[filter_type]
                    else:
                        filter_metrics = getattr(strategy_metrics, filter_type)
                    for allow_mirrors in [False, True]:
                        print(
                            f"\n==={model.upper()} {strategy.upper()} {filter_type.upper()}{'' if allow_mirrors else ' NO MIRROR'}==="
                        )
                        key = (bucket, model, strategy, filter_type, allow_mirrors)
                        vals = slate_values.get(key, [])
                        if not vals:
                            print("0 Slates. Skipping")
                            continue

                        seed = stable_seed(
                            bucket, model, strategy, filter_type, allow_mirrors
                        )
                        point, lo, hi = bootstrap_mean_ci(
                            vals, n_boot=n_boot, seed=seed
                        )
                        p_gt = bootstrap_prob_mean_gt(
                            vals, breakeven, n_boot=n_boot, seed=seed
                        )
                        wins = point * len(vals)
                        n_slates = len(vals)

                        print(
                            f"Observed slate-avg win rate: {point:.2%} ({wins:.2f}/{n_slates})"
                        )
                        print(f"Bootstrap 95% CI: [{lo:.2%}, {hi:.2%}]")
                        print(f"Bootstrap P(mean > {breakeven:.0%}) = {p_gt:.1%}")
                        print(
                            f"Winnings: ${filter_metrics.winnings if allow_mirrors else filter_metrics.winnings_no_mirror:.2f}"
                        )


def evaluation_to_df(
    eval_by_bucket: dict[str, AllSlatesEvaluation],
    slate_values: dict[tuple[str, str, str, str, bool], list[float]],
    breakeven: float = 0.56,
    n_boot: int = 20000,
) -> pd.DataFrame:
    """Flatten bucketed evaluation into a dataframe for writing to disk."""
    rows = []

    for bucket, agg in sorted(eval_by_bucket.items()):
        for model in MODELS:
            model_metrics = getattr(agg, model, None)
            if model_metrics is None:
                continue
            for strategy in STRATEGIES:
                strategy_metrics = getattr(model_metrics, strategy)
                filter_types = list(FILTER_TYPES) + sorted(
                    strategy_metrics.by_slate.keys()
                )
                for filter_type in filter_types:
                    if filter_type in strategy_metrics.by_slate:
                        filter_metrics = strategy_metrics.by_slate[filter_type]
                    else:
                        filter_metrics = getattr(strategy_metrics, filter_type)
                    for allow_mirrors in [True, False]:
                        key = (bucket, model, strategy, filter_type, allow_mirrors)
                        vals = slate_values.get(key, [])
                        n_slates = len(vals)

                        if n_slates == 0:
                            rows.append(
                                {
                                    "bucket": bucket,
                                    "model": model,
                                    "strategy": strategy,
                                    "filter": filter_type,
                                    "allow_mirrors": allow_mirrors,
                                    "n_slates": 0,
                                    "wins": float("nan"),
                                    "win_rate": float("nan"),
                                    "winnings": float("nan"),
                                    "boot_95_low": float("nan"),
                                    "boot_95_high": float("nan"),
                                    "boot_p_breakeven": float("nan"),
                                }
                            )
                            continue

                        seed = stable_seed(
                            bucket, model, strategy, filter_type, allow_mirrors
                        )
                        point, lo, hi = bootstrap_mean_ci(
                            vals, n_boot=n_boot, seed=seed
                        )
                        p_gt = bootstrap_prob_mean_gt(
                            vals, breakeven, n_boot=n_boot, seed=seed
                        )
                        wins = point * n_slates
                        winnings = getattr(
                            filter_metrics,
                            "winnings" if allow_mirrors else "winnings_no_mirror",
                            None,
                        )
                        rows.append(
                            {
                                "bucket": bucket,
                                "model": model,
                                "strategy": strategy,
                                "filter": filter_type,
                                "allow_mirrors": allow_mirrors,
                                "n_slates": n_slates,
                                "wins": round(wins, 2),
                                "win_rate": round(point * 100, 2),
                                "winnings": round(winnings, 2)
                                if winnings is not None
                                else float("nan"),
                                "boot_95_low": round(lo * 100, 2),
                                "boot_95_high": round(hi * 100, 2),
                                "boot_p_breakeven": round(p_gt * 100, 2),
                            }
                        )
    return pd.DataFrame(rows)


def stable_seed(
    bucket: str, model: str, strategy: str, filter_type: str, allow_mirrors: bool
) -> int:
    """
    Deterministic 32-bit seed derived from the group key.
    Stable across Python runs/machines.
    """
    key_str = f"{bucket}|{model}|{strategy}|{filter_type}|{int(allow_mirrors)}"
    digest = hashlib.blake2b(key_str.encode("utf-8"), digest_size=8).digest()
    # Use lower 32 bits for a NumPy seed
    return int.from_bytes(digest, byteorder="little") & 0xFFFFFFFF


def validate_lineup(
    lineup_df,
    dk_salaries,  # full projection DataFrame (with salary, positions, game_time)
    salary_cap=50000,
    slots_definition=SLOTS,
    minutes_floor=None,
    locked=None,  # dict {player_key: required_slot_name}
    now_mt=None,  # optional; for game-time validation
):
    """
    Validate a DraftKings NBA lineup against all structural + DK-specific constraints.

    lineup_df: DataFrame with columns ["player_key", "slot"]
    df: full projections DF with columns: player_key, salary, positions, game_time_local, proj_minutes
    """

    errors = []

    # ---------------------
    # Basic lineup structure
    # ---------------------
    if len(lineup_df) != 8:
        errors.append(f"Lineup must contain exactly 8 players, found {len(lineup_df)}.")

    # Check valid slot names
    allowed_slots = [s["name"] for s in slots_definition]
    for slot in lineup_df["slot"]:
        if slot not in allowed_slots:
            errors.append(f"Invalid slot '{slot}' not in DK NBA slots.")

    # Check no duplicate slots
    if lineup_df["slot"].nunique() != 8:
        errors.append(
            "Lineup must contain exactly one player per slot (duplicate slot found)."
        )

    # Check no duplicate players
    if lineup_df["player_key"].nunique() != 8:
        errors.append("Duplicate players found in lineup.")

    # ---------------------
    # Salary cap
    # ---------------------
    merged = lineup_df.merge(
        dk_salaries,
        on="player_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_dk"),
    )
    merged["salary"] = merged["salary_dk"]
    merged = merged.drop(columns=["salary_dk"])
    total_salary = merged["salary"].sum()

    if total_salary > salary_cap:
        errors.append(f"Salary cap exceeded: {total_salary} > {salary_cap}.")

    # ---------------------
    # Positional eligibility
    # ---------------------
    slot_allowed = {s["name"]: s["allowed"] for s in slots_definition}

    for _, row in merged.iterrows():
        allowed = slot_allowed[row["slot"]]
        # player positions assumed to be a set or list of DK positions
        player_positions = row["positions"]

        if isinstance(player_positions, str):
            # e.g. "PG/SG"
            player_positions = set(player_positions.split("/"))

        if allowed is not None and not (set(player_positions) & allowed):
            errors.append(
                f"Illegal assignment: {row['player_key']} plays {player_positions} "
                f"but placed in {row['slot']} which allows {allowed}."
            )

    # ---------------------
    # Must include players from at least two different games
    # ---------------------
    game_ids = lineup_df["game_info"].apply(extract_game_id)

    unique_games = set(game_ids)

    if len(unique_games) < 2:
        print(lineup_df.to_string())
        errors.append(
            f"Lineup uses players from only one game: {unique_games}. Must use at least two."
        )

    # ---------------------
    # Locked player constraints
    # ---------------------
    if locked:
        for key, required_slot in locked.items():
            rows = merged[merged["player_key"] == key]
            if rows.empty:
                errors.append(f"Locked player {key} not found in lineup.")
            else:
                actual_slot = rows.iloc[0]["slot"]
                if actual_slot != required_slot:
                    errors.append(
                        f"Locked player {key} must be in {required_slot} but is in {actual_slot}."
                    )

    # ---------------------
    # Game-time filtering
    # ---------------------
    if now_mt is not None:
        for _, row in merged.iterrows():
            gt = row.get("game_time_local")
            if gt is not None and isinstance(gt, datetime):
                if gt < now_mt:
                    errors.append(
                        f"Player {row['player_key']} assigned after game start: game_time={gt}, now={now_mt}."
                    )

    # ---------------------
    # Minutes floor (optional)
    # ---------------------
    if minutes_floor is not None:
        for _, row in merged.iterrows():
            if row["proj_minutes"] < minutes_floor:
                errors.append(
                    f"Player {row['player_key']} below minutes floor: {row['proj_minutes']} < {minutes_floor}"
                )

    return errors


# ----------------------------
# Uncertainty Computation
# ----------------------------
def jeffreys_interval(
    wins: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Jeffreys credible interval for a Bernoulli proportion.
    Uses Beta(0.5, 0.5) prior.

    Returns (lower, upper) as floats in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)

    alpha = wins + 0.5
    beta_param = n - wins + 0.5

    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower

    return (
        beta.ppf(lower, alpha, beta_param),
        beta.ppf(upper, alpha, beta_param),
    )


def prob_beating_breakeven(
    wins: int,
    n: int,
    breakeven: float = 0.56,
) -> float:
    """
    P(true win rate > breakeven) under Jeffreys posterior.
    """
    alpha = wins + 0.5
    beta_param = n - wins + 0.5
    return 1.0 - beta.cdf(breakeven, alpha, beta_param)


def wilson_interval(
    wins: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score interval for a Bernoulli proportion.

    Returns (lower, upper) as floats in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)

    z = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }.get(confidence)

    if z is None:
        raise ValueError("Unsupported confidence level")

    p_hat = wins / n
    denom = 1.0 + (z**2) / n

    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n) / denom

    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of slate-level values. Returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        val = float(x[0])
        return (val, val, val)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    mean = float(x.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (mean, lo, hi)


def bootstrap_prob_mean_gt(
    values: list[float],
    threshold: float,
    n_boot: int = 20000,
    seed: int = 123,
) -> float:
    """Bootstrap probability that mean(values) > threshold (frequency under bootstrap resampling)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(x[0] > threshold)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    return float((boot_means > threshold).mean())


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_h2h_lineups(
    slate_id: str,
    slate_games: int,
    top_lineups: TopLineups,
    baseline_proj: Lineup,
    h2h_results: H2HResults,
    proj_source: str,
) -> List[ContestResult]:
    lineup_result = LineupResult()
    adjusted_fragile_minutes_floor_wins = 0
    adjusted_fragile_minutes_floor_wins_no_mirror = 0
    adjusted_fragile_wins = 0
    adjusted_fragile_wins_no_mirror = 0
    max_fpts_wins = 0
    max_fpts_wins_no_mirror = 0

    if proj_source.lower() == "blend":
        max_fpts_lineups = top_lineups.blend_fpts
        max_fpts_lineups_minutes_floor = top_lineups.blend_fpts_minutes_floor
    elif proj_source.lower() == "etr":
        max_fpts_lineups = top_lineups.etr_fpts
        max_fpts_lineups_minutes_floor = top_lineups.etr_fpts_minutes_floor
    elif proj_source.lower() == "rg":
        max_fpts_lineups = top_lineups.rg_fpts
        max_fpts_lineups_minutes_floor = top_lineups.rg_fpts_minutes_floor
    else:
        raise ValueError(f"Invalid projection source {proj_source}")

    top_10_fpts_lineups = max_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    adjusted_fragile_minutes_floor_lineup = max_fpts_lineups_minutes_floor[0]
    adjusted_fragile_lineup = sorted(
        max_fpts_lineups,
        key=lambda lu: adjusted_score(
            lu.projected_fpts,
            lu.projected_minutes,
            lu.total_fragile_minutes,
            slate_games,
        ),
        reverse=True,
    )[0]

    for strategy in STRATEGIES:
        my_lineup = None
        no_mirror_count = 0

        if strategy == "max_fpts":
            my_lineup = baseline_proj
        elif strategy == "adjusted_fragile":
            my_lineup = adjusted_fragile_lineup
        elif strategy == "adjusted_fragile_minutes_floor":
            my_lineup = adjusted_fragile_minutes_floor_lineup
        else:
            raise ValueError("Invalid strategy")

        for fee in range(1, 4):
            contest_result, _ = evaluate_contest(
                my_lineup,
                getattr(h2h_results, f"lineup_{fee}"),
                top_10_median_actual,
                slate_id,
                slate_games,
                f"{proj_source} H2H ${fee} {strategy}",
                fee=fee,
                opponent=getattr(h2h_results, f"opponent_{fee}"),
            )

            winnings = calculate_winnings(
                fee, contest_result.win, contest_result.is_mirror
            )

            strategy_result = getattr(lineup_result, strategy)
            strategy_result.win_rate_by_fee[fee] = contest_result.win
            strategy_result.winnings_by_fee[fee] = winnings

            if not contest_result.is_mirror:
                strategy_result.win_rate_no_mirror_by_fee[fee] = contest_result.win
                strategy_result.winnings_no_mirror_by_fee[fee] = winnings

            if strategy == "max_fpts":
                max_fpts_wins += contest_result.win
                lineup_result.max_fpts.winnings += winnings

                if not contest_result.is_mirror:
                    if lineup_result.max_fpts.winnings_no_mirror is not None:
                        lineup_result.max_fpts.winnings_no_mirror += winnings
                    else:
                        lineup_result.max_fpts.winnings_no_mirror = winnings
                    max_fpts_wins_no_mirror += contest_result.win
                    no_mirror_count += 1
            elif strategy == "adjusted_fragile":
                adjusted_fragile_wins += contest_result.win
                lineup_result.adjusted_fragile.winnings += winnings

                if not contest_result.is_mirror:
                    if lineup_result.adjusted_fragile.winnings_no_mirror is not None:
                        lineup_result.adjusted_fragile.winnings_no_mirror += winnings
                    else:
                        lineup_result.adjusted_fragile.winnings_no_mirror = winnings
                    adjusted_fragile_wins_no_mirror += contest_result.win
                    no_mirror_count += 1
            elif strategy == "adjusted_fragile_minutes_floor":
                adjusted_fragile_minutes_floor_wins += contest_result.win
                lineup_result.adjusted_fragile_minutes_floor.winnings += winnings

                if not contest_result.is_mirror:
                    if (
                        lineup_result.adjusted_fragile_minutes_floor.winnings_no_mirror
                        is not None
                    ):
                        lineup_result.adjusted_fragile_minutes_floor.winnings_no_mirror += winnings
                    else:
                        lineup_result.adjusted_fragile_minutes_floor.winnings_no_mirror = winnings
                    adjusted_fragile_minutes_floor_wins_no_mirror += contest_result.win
                    no_mirror_count += 1

        if no_mirror_count > 0:
            if strategy == "max_fpts":
                lineup_result.max_fpts.win_rate_no_mirror = (
                    max_fpts_wins_no_mirror / no_mirror_count
                )
            elif strategy == "adjusted_fragile":
                lineup_result.adjusted_fragile.win_rate_no_mirror = (
                    adjusted_fragile_wins_no_mirror / no_mirror_count
                )
            elif strategy == "adjusted_fragile_minutes_floor":
                lineup_result.adjusted_fragile_minutes_floor.win_rate_no_mirror = (
                    adjusted_fragile_minutes_floor_wins_no_mirror / no_mirror_count
                )

    lineup_result.adjusted_fragile_minutes_floor.win_rate = (
        adjusted_fragile_minutes_floor_wins / 3
    )
    lineup_result.adjusted_fragile.win_rate = adjusted_fragile_wins / 3
    lineup_result.max_fpts.win_rate = max_fpts_wins / 3

    return lineup_result


def evaluate_contest(
    my_lineup,
    opponent_lineup,
    top_10_median_actual,
    slate_id,
    slate_games,
    strategy,
    fee=1,
    opponent="",
):
    win_vs_opponent = 0.0
    is_mirror = opponent_lineup.players == my_lineup.players

    if is_mirror:
        win_vs_opponent = 0.5  # tie
    elif opponent_lineup.actual_fpts < my_lineup.actual_fpts:
        win_vs_opponent = 1.0  # win
    elif opponent_lineup.actual_fpts > my_lineup.actual_fpts:
        win_vs_opponent = 0.0  # loss
    else:
        win_vs_opponent = 0.5  # tie (rare with different lineups)

    win_vs_top_10_median = 0.0

    if my_lineup.actual_fpts > top_10_median_actual:
        win_vs_top_10_median = 1.0  # win
    elif my_lineup.actual_fpts < top_10_median_actual:
        win_vs_top_10_median = 0.0  # loss
    else:
        win_vs_top_10_median = 0.5  # tie (rare with different lineups)

    opponent_result = OpponentResult(
        slate_id=slate_id,
        opponent_score=opponent_lineup.actual_fpts,
        my_score=my_lineup.actual_fpts,
        win=win_vs_opponent,
        margin=opponent_lineup.actual_fpts - my_lineup.actual_fpts,
        is_mirror=is_mirror,
        opponent=opponent,
    )

    contest_result = ContestResult(
        slate_id=slate_id,
        slate_games=slate_games,
        strategy=strategy,
        my_actual=my_lineup.actual_fpts,
        baseline_proj=my_lineup.actual_fpts,
        win=win_vs_opponent,
        margin=my_lineup.actual_fpts - opponent_lineup.actual_fpts,
        is_mirror=is_mirror,
        win_median=win_vs_top_10_median,
        margin_median=my_lineup.actual_fpts - top_10_median_actual,
        opponent=opponent,
    )

    return contest_result, opponent_result


def slate_bucket(games: int) -> str:
    if games <= 4:
        return "2-4"
    if games <= 8:
        return "5-8"
    return "9+"


def aggregate_slate_results(
    results: list[SlateResult],
) -> dict[str, AllSlatesEvaluation]:
    bucketed: dict[str, AllSlatesEvaluation] = defaultdict(AllSlatesEvaluation)
    overall = AllSlatesEvaluation()

    def finalize(target: AllSlatesEvaluation):
        for model in MODELS:
            model_dest = getattr(target, model)
            for strategy in STRATEGIES:
                strategy_dest = getattr(model_dest, strategy)
                filter_types = list(FILTER_TYPES) + list(strategy_dest.by_slate.keys())
                for filter_type in filter_types:
                    if filter_type in strategy_dest.by_slate:
                        filter_dest = strategy_dest.by_slate[filter_type]
                    else:
                        filter_dest = getattr(strategy_dest, filter_type)
                    denom = getattr(filter_dest, "n_slates")
                    if denom > 0:
                        setattr(
                            filter_dest,
                            "win_rate",
                            getattr(filter_dest, "win_rate") / denom,
                        )

                    denom_no_mirror = getattr(filter_dest, "n_slates_no_mirror")
                    if denom_no_mirror > 0:
                        setattr(
                            filter_dest,
                            "win_rate_no_mirror",
                            getattr(filter_dest, "win_rate_no_mirror")
                            / denom_no_mirror,
                        )

    for slate in results:
        bucket = slate_bucket(slate.slate_games)
        agg = bucketed[bucket]
        agg_all = overall

        for model in MODELS:
            update_aggregate_linuep_metrics(model, slate, agg)
            update_aggregate_linuep_metrics(model, slate, agg_all)

    for agg in bucketed.values():
        finalize(agg)

    finalize(overall)

    result = dict(bucketed)
    # result = {}
    result["overall"] = overall
    return result


def collect_slate_values(
    slate_results: list[SlateResult],
) -> dict[tuple[str, str, str, str, bool], list[float]]:
    """
    Map (bucket, model, strategy, filter_type, allow_mirrors) -> list of per-slate win_rate values.
    Each list contains one value per slate, gated by filter type.
    """
    values: dict[tuple[str, str, str, str, bool], list[float]] = defaultdict(list)
    for slate in slate_results:
        bucket = slate_bucket(slate.slate_games)
        for model in MODELS:
            model_result = getattr(slate, model)
            for strategy in STRATEGIES:
                strategy_result = getattr(model_result, strategy)
                slate_filter = getattr(slate, "slate", "")
                if slate_filter:
                    for allow_mirrors in [True, False]:
                        if allow_mirrors:
                            val = strategy_result.win_rate
                        else:
                            val = strategy_result.win_rate_no_mirror
                        if val is None:
                            continue
                        values[
                            (bucket, model, strategy, slate_filter, allow_mirrors)
                        ].append(val)
                        values[
                            ("overall", model, strategy, slate_filter, allow_mirrors)
                        ].append(val)
                for filter_type in FILTER_TYPES:
                    if filter_type in {"fee_1", "fee_2", "fee_3"}:
                        fee = int(filter_type.split("_")[1])
                        for allow_mirrors in [True, False]:
                            if allow_mirrors:
                                val = strategy_result.win_rate_by_fee.get(fee)
                            else:
                                val = strategy_result.win_rate_no_mirror_by_fee.get(fee)
                            if val is None:
                                continue
                            values[
                                (bucket, model, strategy, filter_type, allow_mirrors)
                            ].append(val)
                            values[
                                ("overall", model, strategy, filter_type, allow_mirrors)
                            ].append(val)
                        continue
                    if filter_type == "no_late_swaps" and slate.has_late_swaps:
                        continue
                    if filter_type == "missing_late_swaps" and not (
                        slate.has_late_swaps and slate.missing_late_swap_proj
                    ):
                        continue
                    if filter_type == "late_swaps" and not (
                        slate.has_late_swaps and not slate.missing_late_swap_proj
                    ):
                        continue

                    for allow_mirrors in [True, False]:
                        if allow_mirrors:
                            val = strategy_result.win_rate
                            values[
                                (bucket, model, strategy, filter_type, allow_mirrors)
                            ].append(val)
                            values[
                                ("overall", model, strategy, filter_type, allow_mirrors)
                            ].append(val)
                        else:
                            val = strategy_result.win_rate_no_mirror
                            if val is None:
                                continue
                            values[
                                (bucket, model, strategy, filter_type, allow_mirrors)
                            ].append(val)
                            values[
                                ("overall", model, strategy, filter_type, allow_mirrors)
                            ].append(val)
    return dict(values)


def update_aggregate_linuep_metrics(model_name, slate, agg):
    model_src = getattr(slate, model_name.split("_")[0])  # LineupResult
    model_dest = getattr(agg, model_name)  # AggregateLineupMetrics

    fee_filters = {"fee_1", "fee_2", "fee_3"}
    slate_filter = slate.slate

    # Update win-rate and winnings
    for strategy in STRATEGIES:
        strategy_dest = getattr(model_dest, strategy)
        strategy_src = getattr(model_src, strategy)

        if slate_filter:
            slate_dest = strategy_dest.by_slate.setdefault(
                slate_filter, AggregateStrategyMetrics()
            )
            for allow_mirrors in [True, False]:
                filter_win_rate = "win_rate" if allow_mirrors else "win_rate_no_mirror"
                val_win_rate = getattr(strategy_src, filter_win_rate)
                if val_win_rate is not None:
                    setattr(
                        slate_dest,
                        filter_win_rate,
                        getattr(slate_dest, filter_win_rate) + val_win_rate,
                    )
                    n_slates = "n_slates" if allow_mirrors else "n_slates_no_mirror"
                    setattr(slate_dest, n_slates, getattr(slate_dest, n_slates) + 1)

                filter_winnings = "winnings" if allow_mirrors else "winnings_no_mirror"
                val_winnings = getattr(strategy_src, filter_winnings)
                if val_winnings is not None:
                    setattr(
                        slate_dest,
                        filter_winnings,
                        getattr(slate_dest, filter_winnings) + val_winnings,
                    )
        for filter_type in FILTER_TYPES:
            filter_dest = getattr(strategy_dest, filter_type)

            if filter_type not in fee_filters:
                if filter_type == "no_late_swaps" and slate.has_late_swaps:
                    continue

                if filter_type == "missing_late_swaps" and not (
                    slate.has_late_swaps and slate.missing_late_swap_proj
                ):
                    continue

                if filter_type == "late_swaps" and not (
                    slate.has_late_swaps and not slate.missing_late_swap_proj
                ):
                    continue

            for allow_mirrors in [True, False]:
                filter_win_rate = "win_rate" if allow_mirrors else "win_rate_no_mirror"
                old_win_rate = getattr(filter_dest, filter_win_rate)

                if filter_type in fee_filters:
                    fee = int(filter_type.split("_")[1])
                    if allow_mirrors:
                        val_win_rate = strategy_src.win_rate_by_fee.get(fee)
                        val_winnings = strategy_src.winnings_by_fee.get(fee)
                    else:
                        val_win_rate = strategy_src.win_rate_no_mirror_by_fee.get(fee)
                        val_winnings = strategy_src.winnings_no_mirror_by_fee.get(fee)
                else:
                    val_win_rate = getattr(strategy_src, filter_win_rate)
                    val_winnings = getattr(
                        strategy_src,
                        "winnings" if allow_mirrors else "winnings_no_mirror",
                    )

                if val_win_rate is not None:
                    setattr(filter_dest, filter_win_rate, old_win_rate + val_win_rate)
                    n_slates = "n_slates" if allow_mirrors else "n_slates_no_mirror"
                    old_n_slates = getattr(filter_dest, n_slates)
                    setattr(filter_dest, n_slates, old_n_slates + 1)

                filter_winnings = "winnings" if allow_mirrors else "winnings_no_mirror"
                if val_winnings is not None:
                    old_winnings = getattr(filter_dest, filter_winnings)
                    setattr(filter_dest, filter_winnings, old_winnings + val_winnings)


def evaluate_slate(slate: ResultsFileMeta) -> pd.DataFrame:
    """
    Example backtest workflow. `max_fpts_indices` and `max_minutes_indices` are lists of
    lineups represented by player indices into the projection dataframe.
    """
    meta = SlateMeta(
        sport=slate.sport,
        slate=slate.slate,
        site=slate.site,
        datetime=slate.datetime,
        id=slate.slate_id,
    )

    dk_salaries = load_dk_salaries_csv(meta)

    game_times = sorted(dk_salaries.set_index("player_key")["game_time_local"].unique())

    n_game_times = len(game_times)
    slate_result = SlateResult()
    slate_result.slate = meta.slate

    if n_game_times == 1:
        slate_result.has_late_swaps = False
        logging.info("No late swaps")

    for i in range(1, len(game_times)):
        lock_time = game_times[i].strftime("%H%M")
        etr_proj_path = (
            ETR_PROJ_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.datetime}T{lock_time}.csv"
        )

        if not etr_proj_path.is_file():
            slate_result.missing_late_swap_proj = True

            logging.info("Missing ETR late swap projections.")
            break

        rg_proj_path = (
            RG_PROJ_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.datetime}T{lock_time}.csv"
        )

        if not rg_proj_path.is_file():
            slate_result.missing_late_swap_proj = True

            logging.info("Missing RG late swap projections.")
            break

    etr_proj, rg_proj, slate_games = load_projections(meta, remove_nan=False)
    slate_result.slate_games = slate_games
    blend_proj = blend_projections(rg_proj, etr_proj)

    _, results_players_df = load_results_csv(slate)
    rg_merged = join_proj_results(rg_proj, results_players_df)
    etr_merged = join_proj_results(etr_proj, results_players_df)
    blend_merged = join_proj_results(blend_proj, results_players_df)
    top_lineups = load_candidates(
        meta,
        blend_merged,
        etr_merged,
        rg_merged,
        dk_salaries,
        slate_result.has_late_swaps and not slate_result.missing_late_swap_proj,
        # False,
    )

    # Evaluate strategies against H2H contests
    h2h_results = load_h2h_results(slate, rg_merged)

    slate_result.rg = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.rg_fpts[0],
        h2h_results=h2h_results,
        proj_source="RG",
    )

    slate_result.etr = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.etr_fpts[0],
        h2h_results=h2h_results,
        proj_source="ETR",
    )

    slate_result.blend = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.blend_fpts[0],
        h2h_results=h2h_results,
        proj_source="BLEND",
    )

    return slate_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    slates = get_slates()
    n_slates = len(slates)

    slates.sort(key=lambda x: x.datetime)

    slate_results: List[SlateResult] = []

    for i, slate in enumerate(slates):
        print(f"[{i + 1}/{n_slates}] {slate.slate_id}")

        slate_result = evaluate_slate(slate=slate)

        slate_results.append(slate_result)

    aggregate = aggregate_slate_results(slate_results)
    slate_values = collect_slate_values(slate_results)
    print_evaluation(aggregate, slate_values)
    eval_df = evaluation_to_df(aggregate, slate_values)
    out_path = Path("data/processed/backtest_eval.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)
