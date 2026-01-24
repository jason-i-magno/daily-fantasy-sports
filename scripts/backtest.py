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
from datetime import datetime
from typing import List, Set, Tuple

import pandas as pd
from utils import (
    BLEND_CANDIDATE_DIR,
    ETR_CANDIDATE_DIR,
    H2H_DIR,
    RESULTS_DIR,
    RG_CANDIDATE_DIR,
    SLOTS,
    ResultsFileMeta,
    SlateMeta,
    adjusted_score,
    blend_projections,
    load_dk_salaries_csv,
    load_projections,
    normalize_name,
    parse_filename,
    total_fragile_minutes,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Data structures
# ----------------------------
@dataclasses.dataclass
class AggregateLineupMetrics:
    adjusted_fragile_minutes_floor_win_rate: float = 0.0
    adjusted_fragile_minutes_floor_winnings: float = 0.0
    adjusted_fragile_win_rate: float = 0.0
    adjusted_fragile_winnings: float = 0.0
    max_fpts_win_rate: float = 0.0
    max_fpts_winnings: float = 0.0


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
    num_slates: int = 0


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
    winnings: float
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
class LineupResult:
    adjusted_fragile_minutes_floor_win_rate: float = 0.0
    adjusted_fragile_minutes_floor_winnings: float = 0.0
    adjusted_fragile_win_rate: float = 0.0
    adjusted_fragile_winnings: float = 0.0
    max_fpts_win_rate: float = 0.0
    max_fpts_winnings: float = 0.0


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


def lineup_from_keys(player_keys: list[str], dk_salaries: pd.DataFrame):
    """
    Convert 8 player_keys into a DataFrame compatible with validate_lineup().
    Requires full_df to contain player metadata.
    """
    if len(player_keys) != 8:
        raise ValueError("Lineup must contain exactly 8 players.")

    SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

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
        H2H_DIR / f"{slate.sport}_{slate.slate}_{slate.site}_h2h_{slate.date}.json"
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
    meta: SlateMeta, proj_source: str, proj: pd.DataFrame, dk_salaries: pd.DataFrame
) -> Tuple[List[Lineup], List[Lineup], List[Lineup]]:
    candidate_dirs = {
        "blend": BLEND_CANDIDATE_DIR,
        "etr": ETR_CANDIDATE_DIR,
        "rg": RG_CANDIDATE_DIR,
    }
    candidate_lineups_path = (
        candidate_dirs[proj_source]
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.date}.json"
    )

    with open(candidate_lineups_path, "r") as f:
        payload = json.load(f)

    required = {"slate_id", "top_fpts", "top_minutes", "top_adjusted"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Candidate lineup file missing keys: {missing}")

    n_lineups = 10

    top_fpts_keys = payload["top_fpts"][:n_lineups]
    top_minutes_keys = payload["top_minutes"][:n_lineups]
    top_fpts_minutes_floor_keys = payload["top_adjusted"][:n_lineups]

    for i, lineups in enumerate(
        [top_fpts_keys, top_minutes_keys, top_fpts_minutes_floor_keys]
    ):
        for lineup in lineups:
            lineup_df = lineup_from_keys(lineup, dk_salaries)
            errors = validate_lineup(lineup_df, dk_salaries)

            if errors:
                print(i, proj_source)
                print(lineup)
                for error in errors:
                    logging.error(error)

    top_fpts_lineups = [
        lineup_from_player_keys(proj, player_keys) for player_keys in top_fpts_keys
    ]
    top_minutes_lineups = [
        lineup_from_player_keys(proj, player_keys) for player_keys in top_minutes_keys
    ]
    top_fpts_minutes_floor_lineups = [
        lineup_from_player_keys(proj, player_keys)
        for player_keys in top_fpts_minutes_floor_keys
    ]

    return (
        top_fpts_lineups,
        top_minutes_lineups,
        top_fpts_minutes_floor_lineups,
    )


def load_candidates(
    meta: SlateMeta, blend_proj, etr_proj, rg_proj, dk_salaries
) -> TopLineups:
    lineups = TopLineups()
    (
        lineups.rg_fpts,
        lineups.rg_minutes,
        lineups.rg_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "rg", rg_proj, dk_salaries)
    (
        lineups.etr_fpts,
        lineups.etr_minutes,
        lineups.etr_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "etr", etr_proj, dk_salaries)
    (
        lineups.blend_fpts,
        lineups.blend_minutes,
        lineups.blend_fpts_minutes_floor,
    ) = load_candidate_lineups(meta, "blend", blend_proj, dk_salaries)

    return lineups


def load_results_csv(slate: ResultsFileMeta) -> pd.DataFrame:
    """
    Load post-slate results. Expected columns: id_col, actual_fpts.
    """
    results_path = (
        RESULTS_DIR
        / f"{slate.sport}_{slate.slate}_{slate.site}_results_{slate.date}.csv"
    )

    raw = pd.read_csv(results_path, dtype=str)

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


def print_evaluation(eval: AllSlatesEvaluation):
    models = ["blend", "etr", "rg"]
    win_rate_fields = [
        "adjusted_fragile_minutes_floor_win_rate",
        "adjusted_fragile_win_rate",
        "max_fpts_win_rate",
    ]

    winnings_fields = [
        "adjusted_fragile_minutes_floor_winnings",
        "adjusted_fragile_winnings",
        "max_fpts_winnings",
    ]

    print(f"\nEvaluated {eval.num_slates} slates\n")

    for model in models:
        metrics = getattr(eval, model)
        print(f"=== {model.upper()} ===")
        for field_name in winnings_fields:
            print(f"  {field_name}: ${getattr(metrics, field_name):.2f}")

        for field_name in win_rate_fields:
            print(f"  {field_name}: {getattr(metrics, field_name) * 100:.2f}%")
        print()


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
    top_lineups: TopLineups,
    proj_source: str,
) -> List[ContestResult]:
    results: List[ContestResult] = []
    baseline_proj = top_lineups.etr_fpts[0]

    if proj_source.lower() == "blend":
        top_fpts_lineups = top_lineups.blend_fpts
        top_minutes_lineups = top_lineups.blend_minutes
    elif proj_source.lower() == "etr":
        top_fpts_lineups = top_lineups.etr_fpts
        top_minutes_lineups = top_lineups.etr_minutes
    elif proj_source.lower() == "rg":
        top_fpts_lineups = top_lineups.rg_fpts
        top_minutes_lineups = top_lineups.rg_minutes
    else:
        raise ValueError(f"Invalid projection source {proj_source}")

    top_10_fpts_lineups = top_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    # Max FPTS
    max_fpts_lineup = top_fpts_lineups[0]

    contest_result, _ = evaluate_contest(
        baseline_proj,
        max_fpts_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max FPTs",
    )
    results.append(contest_result)

    # Max Minutes
    max_minutes_lineup = top_minutes_lineups[0]

    contest_result, _ = evaluate_contest(
        baseline_proj,
        max_minutes_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max Mins",
    )
    results.append(contest_result)

    # Max Minutes within Top FPTs
    max_minutes_within_top_fpts_lineup = sorted(
        top_fpts_lineups,
        key=lambda lu: (lu.projected_minutes, lu.projected_fpts),
        reverse=True,
    )[0]

    contest_result, _ = evaluate_contest(
        baseline_proj,
        max_minutes_within_top_fpts_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Max Mins in Top FPTs",
    )
    results.append(contest_result)

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

    contest_result, _ = evaluate_contest(
        baseline_proj,
        adjusted_fragile_lineup,
        top_10_median_actual,
        slate_id,
        slate_games,
        f"{proj_source} Adj Frag",
    )
    results.append(contest_result)

    # Top 10 median vs RG max FPTs
    win_vs_proj = 0.0

    if top_10_median_actual > baseline_proj.actual_fpts:
        win_vs_proj = 1.0  # win
    elif top_10_median_actual < baseline_proj.actual_fpts:
        win_vs_proj = 0.0  # loss
    else:
        win_vs_proj = 0.5  # tie (rare with different lineups)

    results.append(
        ContestResult(
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
    top_lineups: TopLineups,
    baseline_proj: Lineup,
    h2h_results: H2HResults,
    proj_source: str,
) -> List[ContestResult]:
    contest_results: List[ContestResult] = []
    opponent_results = []
    lineup_result = LineupResult()
    adjusted_fragile_minutes_floor_wins = 0
    adjusted_fragile_wins = 0
    max_fpts_wins = 0

    if proj_source.lower() == "blend":
        top_fpts_lineups = top_lineups.blend_fpts
        top_fpts_lineups_minutes_floor = top_lineups.blend_fpts_minutes_floor
    elif proj_source.lower() == "etr":
        top_fpts_lineups = top_lineups.etr_fpts
        top_fpts_lineups_minutes_floor = top_lineups.etr_fpts_minutes_floor
    elif proj_source.lower() == "rg":
        top_fpts_lineups = top_lineups.rg_fpts
        top_fpts_lineups_minutes_floor = top_lineups.rg_fpts_minutes_floor
    else:
        raise ValueError(f"Invalid projection source {proj_source}")

    top_10_fpts_lineups = top_fpts_lineups[:10]
    actual_scores = sorted(lu.actual_fpts for lu in top_10_fpts_lineups)
    top_10_median_actual = actual_scores[len(actual_scores) // 2]

    adjusted_fragile_minutes_floor_lineup = top_fpts_lineups_minutes_floor[0]
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

    for strategy in ["max_fpts", "adj_frag", "adj_frag_minutes_floor"]:
        lineup = None

        if strategy == "max_fpts":
            lineup = baseline_proj
        elif strategy == "adj_frag":
            lineup = adjusted_fragile_lineup
        elif strategy == "adj_frag_minutes_floor":
            lineup = adjusted_fragile_minutes_floor_lineup

        for fee in range(1, 4):
            contest_result, opponent_result = evaluate_contest(
                lineup,
                getattr(h2h_results, f"lineup_{fee}"),
                top_10_median_actual,
                slate_id,
                slate_games,
                f"{proj_source} H2H ${fee} {strategy}",
                fee=fee,
                opponent=getattr(h2h_results, f"opponent_{fee}"),
            )

            if strategy == "max_fpts" and (
                fee == 1
                or (fee == 2 and h2h_results.opponent_2 != h2h_results.opponent_1)
                or (
                    fee == 3
                    and h2h_results.opponent_3 != h2h_results.opponent_2
                    and h2h_results.opponent_3 != h2h_results.opponent_1
                )
            ):
                opponent_results.append(opponent_result)

            contest_results.append(contest_result)

            win = 1 - contest_result.win
            winnings = calculate_winnings(
                fee, 1 - contest_result.win, contest_result.is_mirror
            )

            if strategy == "max_fpts":
                max_fpts_wins += win
                lineup_result.max_fpts_winnings += winnings
            elif strategy == "adj_frag":
                adjusted_fragile_wins += win
                lineup_result.adjusted_fragile_winnings += winnings
            elif strategy == "adj_frag_minutes_floor":
                adjusted_fragile_minutes_floor_wins += win
                lineup_result.adjusted_fragile_minutes_floor_winnings += winnings

    lineup_result.adjusted_fragile_minutes_floor_win_rate = (
        adjusted_fragile_minutes_floor_wins / 3
    )
    lineup_result.adjusted_fragile_win_rate = adjusted_fragile_wins / 3
    lineup_result.max_fpts_win_rate = max_fpts_wins / 3

    return (
        contest_results,
        opponent_results,
        lineup_result,
    )


def evaluate_contest(
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

    contest_result = ContestResult(
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

    return contest_result, opponent_result


def aggregate_results(results: List[ContestResult], opponent_results) -> pd.DataFrame:
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


def aggregate_slate_results(results: list[SlateResult]) -> AllSlatesEvaluation:
    agg = AllSlatesEvaluation()

    win_rate_fields = [
        "adjusted_fragile_minutes_floor_win_rate",
        "adjusted_fragile_win_rate",
        "max_fpts_win_rate",
    ]

    winnings_fields = [
        "adjusted_fragile_minutes_floor_winnings",
        "adjusted_fragile_winnings",
        "max_fpts_winnings",
    ]

    for index, slate in enumerate(results, start=1):
        agg.num_slates = index

        for model_name in ("blend", "etr", "rg"):
            src = getattr(slate, model_name)  # LineupResult
            dest = getattr(agg, model_name)  # AggregateLineupMetrics

            if src is None:
                continue

            # Update win-rate
            for f in win_rate_fields:
                old = getattr(dest, f)
                val = getattr(src, f)

                setattr(dest, f, old + (val - old) / index)

            # Update winnings
            for f in winnings_fields:
                setattr(dest, f, getattr(dest, f) + getattr(src, f))

    agg.num_slates = len(results)
    return agg


def evaluate_slate(
    slate: ResultsFileMeta,
    id_col: str = "player_key",
) -> pd.DataFrame:
    """
    Example backtest workflow. `top_fpts_indices` and `top_minutes_indices` are lists of
    lineups represented by player indices into the projection dataframe.
    """
    meta = SlateMeta(
        sport=slate.sport,
        slate=slate.slate,
        site=slate.site,
        date=slate.date,
        id=slate.slate_id,
    )

    etr_proj, rg_proj, slate_games = load_projections(meta, remove_nan=False)
    dk_salaries = load_dk_salaries_csv(meta)
    blend_proj = blend_projections(rg_proj, etr_proj)

    _, results_players_df = load_results_csv(slate)
    rg_merged = join_proj_results(rg_proj, results_players_df, id_col=id_col)
    etr_merged = join_proj_results(etr_proj, results_players_df, id_col=id_col)
    blend_merged = join_proj_results(blend_proj, results_players_df, id_col=id_col)
    top_lineups = load_candidates(
        meta, blend_merged, etr_merged, rg_merged, dk_salaries
    )

    contest_results: List[ContestResult] = []

    # Evaluate strategies against projection strategies
    for proj_source in ["BLEND", "ETR", "RG"]:
        contest_results += evaluate_proj_lineups(
            slate_id=slate.slate_id,
            slate_games=slate_games,
            top_lineups=top_lineups,
            proj_source=proj_source,
        )

    # Evaluate strategies against H2H contests
    h2h_results = load_h2h_results(slate, rg_merged)
    slate_result = SlateResult()

    (
        results,
        _,
        slate_result.rg,
    ) = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.rg_fpts[0],
        h2h_results=h2h_results,
        proj_source="RG",
    )
    contest_results += results

    (
        results,
        opponent_results,
        slate_result.etr,
    ) = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.etr_fpts[0],
        h2h_results=h2h_results,
        proj_source="ETR",
    )
    contest_results += results

    (
        results,
        _,
        slate_result.blend,
    ) = evaluate_h2h_lineups(
        slate_id=slate.slate_id,
        slate_games=slate_games,
        top_lineups=top_lineups,
        baseline_proj=top_lineups.blend_fpts[0],
        h2h_results=h2h_results,
        proj_source="BLEND",
    )
    contest_results += results

    return contest_results, opponent_results, slate_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    slates = []

    for results_file in RESULTS_DIR.iterdir():
        if not results_file.is_file():
            continue

        meta = parse_filename(results_file.stem)

        if meta.sport != "nba":
            continue

        slates.append(meta)

    n_slates = len(slates)

    slates.sort(key=lambda x: x.date)

    contest_results = []
    opponent_results = []

    slate_results: List[SlateResult] = []

    for i, slate in enumerate(slates):
        print(f"[{i + 1}/{n_slates}] {slate.slate_id}")

        results, opp_results, slate_result = evaluate_slate(slate=slate)

        contest_results += results
        opponent_results += opp_results
        slate_results.append(slate_result)

    aggregate = aggregate_slate_results(slate_results)
    print_evaluation(aggregate)

    summary_df, opponents_df = aggregate_results(contest_results, opponent_results)
    summary_df.to_csv("data/processed/backtest.csv")
    opponents_df.to_csv("data/processed/opponents.csv")
