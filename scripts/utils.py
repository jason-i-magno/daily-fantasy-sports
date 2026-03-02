"""
Shared helpers for DFS projection scripts.
"""

from __future__ import annotations

import dataclasses
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

# Canonical column names and common aliases found in provider exports.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "player_name": ("player_name", "player", "name", "Name", "PLAYER", "DKName"),
    "salary": ("salary", "Salary", "SALARY"),
    "proj_minutes": ("projected_minutes", "minutes", "Minutes", "MINUTES"),
    "proj_fpts": ("projected_fpts", "FPTS", "fpts"),
    "position": (
        "position",
        "positions",
        "pos",
        "Pos",
        "POS",
        "Position",
    ),
    "proj_ceil": ("ceiling", "Ceiling", "CEILING", "ceil", "Ceil", "CEIL"),
    "proj_floor": ("floor", "Floor", "FLOOR"),
    "team": ("team", "Team", "TEAM", "TeamAbbrev"),
    "game_info": {"Game Info"},
}

_FILENAME_RE = re.compile(
    r"""
    ^
    (?P<sport>[a-z]+)_
    (?P<slate>[a-z0-9\-]+)_
    (?P<site>[a-z]+)
    (?:_(?P<source>[a-z]+))?
    _
    (?P<datatype>[a-z\-]+)_
    (?P<datetime>
        \d{4}-\d{2}-\d{2}
        (?:T\d{4})?
        (?:[+-]\d{4})?
    )
    $
    """,
    re.VERBOSE,
)

_SLATE_ID_RE = re.compile(
    r"""
    ^
    (?P<sport>[a-z]+)_
    (?P<slate>[a-z0-9\-]+)_
    (?P<site>[a-z]+)_
    (?P<datetime>
        \d{4}-\d{2}-\d{2}
        (?:T\d{4})?
        (?:[+-]\d{4})?
    )
    $
    """,
    re.VERBOSE,
)

_SUFFIX_RE = re.compile(
    r"""
    (?:,)?\s*                    # optional comma/space
    (?:jr|sr|ii|iii|iv|v|vi|vii|viii|ix|x)  # suffixes
    \.?                          # optional period
    \s*$                         # end of string
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Path Constants
DATA_DIR = Path("data")

CANDIDATE_LINEUPS_DIR = DATA_DIR / "candidate_lineups"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_LINEUPS_DIR = DATA_DIR / "raw"

BLEND_AVG_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "blend_avg"
BLEND_MIN_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "blend_min"
DK_HISTORY_DIR = RAW_LINEUPS_DIR / "dk_history"
DK_SALARIES_DIR = RAW_LINEUPS_DIR / "draftkings"
ETR_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "etr"
ETR_PROJ_DIR = RAW_LINEUPS_DIR / "etr"
H2H_DIR = PROCESSED_DIR / "h2h"
NBA_BOX_SCORES_DIR = RAW_LINEUPS_DIR / "nba_box_scores"
RG_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "rotogrinders"
RG_PROJ_DIR = RAW_LINEUPS_DIR / "rotogrinders"

MODELS = [
    "blend_avg",
    "blend_min",
    "etr",
    "rg",
]

CANDIDATE_DIR_MAP = {
    "blend_avg": BLEND_AVG_CANDIDATE_DIR,
    "blend_min": BLEND_MIN_CANDIDATE_DIR,
    "etr": ETR_CANDIDATE_DIR,
    "rg": RG_CANDIDATE_DIR,
}

# Lineup Structure
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

SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# Projection Columns
PROJ_COLS = [
    "player_name",
    "salary",
    "proj_minutes",
    "proj_fpts",
    "position",
    "proj_ceil",
    "proj_floor",
    "team",
]

# STRATEGIES
STRATEGIES = [
    "max_ceil",
    "max_ceil_force_sal_50000",
    "max_ceil_1_from_top_team",
    "max_ceil_2_from_top_team",
    # "max_minutes",
    "max_floor",
    "max_floor_adjusted_fragile",
    "max_floor_minutes_floor",
    "max_floor_force_sal_50000",
    "max_floor_1_from_top_game",
    "max_floor_2_from_top_game",
    # "max_floor_3_from_top_game",
    # "max_floor_4_from_top_game",
    # "max_floor_5_from_top_game",
    "max_floor_1_from_top_team",
    "max_floor_2_from_top_team",
    # "max_floor_3_from_top_team",
    # "max_floor_4_from_top_team",
    # "max_floor_5_from_top_team",
    "max_fpts",
    "max_fpts_adjusted_fragile",
    "max_fpts_minutes_floor",
    "max_fpts_force_top_proj_1",
    "max_fpts_force_top_proj_2",
    "max_fpts_force_top_proj_3",
    "max_fpts_force_sal_50000",
    "max_fpts_1_from_top_game",
    "max_fpts_2_from_top_game",
    # "max_fpts_3_from_top_game",
    # "max_fpts_4_from_top_game",
    # "max_fpts_5_from_top_game",
    "max_fpts_1_from_top_team",
    "max_fpts_2_from_top_team",
    # "max_fpts_3_from_top_team",
    # "max_fpts_4_from_top_team",
    # "max_fpts_5_from_top_team",
    "max_fpts_1_from_top_team_force_sal_50000",
    "max_fpts_2_from_top_team_force_sal_50000",
]

# Aliases
ALIASES = {
    "Jimmy Butler III": "Jimmy Butler",
    "GG Jackson": "Gregory Jackson",
    "Alex Sarr": "Alexandre Sarr",
    "Robert Williams III": "Robert Williams",
}


# ----------------------------
# Data structures
# ----------------------------
@dataclasses.dataclass(frozen=True)
class ResultsFileMeta:
    sport: str
    slate: str
    site: str
    source: str
    datatype: str
    datetime: str
    slate_id: str


@dataclasses.dataclass()
class SlateMeta:
    sport: str
    slate: str
    site: str
    datetime: str
    id: str


# ----------------------------
# Helpers
# ----------------------------
def adjusted_score(proj_fpts, proj_minutes, tfm, games):
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

    return (
        proj_fpts
        - alpha_from_slate(games) * math.sqrt(tfm)
        - beta * max(0, minutes_floor_from_slate(games) - proj_minutes)
    )


def blend_projections(
    rg_df: pd.DataFrame,
    etr_df: pd.DataFrame,
    model: str,
    weight_rg: float = 0.5,
    weight_etr: float = 0.5,
) -> pd.DataFrame:
    rg = rg_df.rename(
        columns={
            "proj_fpts": "rg_fpts",
            "proj_minutes": "rg_minutes",
            "proj_ceil": "rg_ceil",
            "proj_floor": "rg_floor",
        }
    )

    etr = etr_df.rename(
        columns={
            "proj_fpts": "etr_fpts",
            "proj_minutes": "etr_minutes",
            "proj_ceil": "etr_ceil",
            "proj_floor": "etr_floor",
        }
    )

    # Merge on player_name (inner join ensures both sites project the player)
    blend = rg.merge(etr, on="player_key", how="inner")

    # Create blended + min columns
    blend["proj_minutes"] = blend[["rg_minutes", "etr_minutes"]].min(axis=1)
    if model == "blend_avg":
        blend["proj_fpts"] = (
            weight_rg * blend["rg_fpts"] + weight_etr * blend["etr_fpts"]
        )
        blend["proj_ceil"] = (
            weight_rg * blend["rg_ceil"] + weight_etr * blend["etr_ceil"]
        )
        blend["proj_floor"] = (
            weight_rg * blend["rg_floor"] + weight_etr * blend["etr_floor"]
        )
    else:
        blend["proj_fpts"] = blend[["rg_fpts", "etr_fpts"]].min(axis=1)
        blend["proj_ceil"] = blend[["rg_ceil", "etr_ceil"]].min(axis=1)
        blend["proj_floor"] = blend[["rg_floor", "etr_floor"]].min(axis=1)
    blend["player_name"] = blend["player_name_x"]
    blend["position"] = blend["position_x"]
    blend["positions"] = blend["positions_x"]
    blend["salary"] = blend["salary_x"]
    blend["team"] = blend["team_x"]
    cols_to_drop = [
        "etr_fpts",
        "etr_minutes",
        "etr_ceil",
        "etr_floor",
        "rg_fpts",
        "rg_minutes",
        "rg_ceil",
        "rg_floor",
        "player_name_x",
        "player_name_y",
        "salary_x",
        "salary_y",
        "position_x",
        "position_y",
        "positions_x",
        "positions_y",
        "team_x",
        "team_y",
    ]
    blend = blend.drop(columns=cols_to_drop)

    return blend


def calcFloorFromMeanCeil(
    mean: pd.Series, ceil: pd.Series, *, clamp_min: float = 0.0
) -> pd.Series:
    """
    Simple symmetric 'p10-ish' floor proxy: floor = 2*mean - ceil.
    """
    floor = 2.0 * mean - ceil
    floor = floor.clip(lower=clamp_min)

    # Optional: keep floor <= mean when both are present
    floor = pd.concat([floor, mean], axis=1).min(axis=1)

    return floor


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, stripping currency/commas where present."""
    cleaned = series.replace(r"[\\$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_output_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_slates():
    slates = []

    for dk_history_file in DK_HISTORY_DIR.iterdir():
        if not dk_history_file.is_file():
            continue

        meta = parse_filename(dk_history_file.stem)

        if meta.sport != "nba":
            continue

        slates.append(meta)

    return slates


def lineup_df_to_player_keys(
    lineup_df: pd.DataFrame,
    id_col: str = "player_name",
) -> list[int]:
    player_keys = []
    for name in lineup_df[id_col]:
        player_keys.append(normalize_name(name))
    return player_keys


def load_dk_salaries_csv(meta: SlateMeta) -> tuple[pd.DataFrame, int, list[str]]:
    cols = [
        "player_name",
        "salary",
        "position",
        "team",
        "game_info",
    ]
    dk_salaries_path = (
        DK_SALARIES_DIR
        / f"{meta.sport}_{meta.slate}_{meta.site}_salaries_{meta.datetime.split('T')[0]}.csv"
    )

    if not dk_salaries_path.is_file():
        raise FileNotFoundError(f"DK salaries file not found '{dk_salaries_path}'")

    df = pd.read_csv(dk_salaries_path)

    df = normalize_columns(
        df,
        required=["player_name", "salary", "position", "team", "game_info"],
    )
    df = df[cols].copy()
    df["salary"] = coerce_numeric(df["salary"])
    df["positions"] = df["position"].map(parse_positions)
    df["player_key"] = df["player_name"].map(normalize_name)
    parsed_game = df["game_info"].apply(parse_game_time)
    df["game_id"] = parsed_game.apply(lambda x: x[0] if x else None)
    df["game_time_local"] = parsed_game.apply(lambda x: x[1] if x else None)

    df = df[df["positions"].map(bool)]

    return df.reset_index(drop=True)


def load_projection_csv(
    path: Path,
    dk_df: pd.DataFrame = None,
    remove_nan: bool = True,
    locked_players: dict = {},
) -> tuple[pd.DataFrame, int]:
    source = None

    if "rotogrinders" in str(path):
        source = "rg"
    elif "etr" in str(path):
        source = "etr"
    else:
        raise ValueError("Invalid projection source.")

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
    df["salary"] = coerce_numeric(df["salary"])
    df["proj_minutes"] = coerce_numeric(df["proj_minutes"])
    df["proj_fpts"] = coerce_numeric(df["proj_fpts"])
    df["positions"] = df["position"].map(parse_positions)
    df["proj_ceil"] = coerce_numeric(df["proj_ceil"])
    df["player_key"] = df["player_name"].map(normalize_name)

    if source == "rg":
        df["proj_floor"] = coerce_numeric(df["proj_floor"])
    elif source == "etr":
        # Create a proj_floor proxy from mean + ceiling
        # (Assumes ceiling acts like a high-quantile; symmetric lower quantile)
        df["proj_floor"] = calcFloorFromMeanCeil(df["proj_fpts"], df["proj_ceil"])
        bad = (
            df["proj_ceil"].notna()
            & df["proj_fpts"].notna()
            & (df["proj_ceil"] < df["proj_fpts"])
        )
        if bad.any():
            # If ceiling is below mean, fall back to something conservative: proj_floor = 0 or proj_floor = mean
            df.loc[bad, "proj_floor"] = 0.0

    # ----------------------------------------
    # OVERRIDE missing projections for locked players
    # ----------------------------------------
    if locked_players:
        locked_keys = set(locked_players.keys())

        for key in locked_keys:
            mask = df["player_key"] == key

            # Only override if missing (NaN)
            if df.loc[mask, "proj_minutes"].isna().any():
                df.loc[mask, "proj_minutes"] = 0.0

            if df.loc[mask, "proj_fpts"].isna().any():
                df.loc[mask, "proj_fpts"] = 0.0

            if df.loc[mask, "proj_ceil"].isna().any():
                df.loc[mask, "proj_ceil"] = 0.0

            if df.loc[mask, "proj_floor"].isna().any():
                df.loc[mask, "proj_floor"] = 0.0

        # Players missing entirely from projection file
        missing_keys = locked_keys - set(df["player_key"])

        if missing_keys:
            # You MUST have DK salary dataframe accessible here
            # dk_df must contain: player_key, salary, positions, team, game_info, etc.
            # Add dk_df as a function argument if needed.
            for key in missing_keys:
                dk_row = dk_df[dk_df["player_key"] == key]
                if dk_row.empty:
                    raise ValueError(f"Locked player {key} missing from DK salaries.")

                # Build a fallback projection row
                new_row = {
                    "player_name": dk_row["player_name"].iloc[0],
                    "player_key": key,
                    "salary": dk_row["salary"].iloc[0],
                    "proj_minutes": 0.0,  # median or default
                    "proj_fpts": 0.0,
                    "positions": dk_row["positions"].iloc[0],
                    "team": dk_row["team"].iloc[0],
                    "proj_ceil": 0.0,
                    "proj_floor": 0.0,
                }

                df.loc[len(df)] = new_row

    if remove_nan:
        remove_nan_cols = [
            "player_name",
            "proj_minutes",
            "proj_fpts",
        ]
        df = df.dropna(subset=remove_nan_cols)
    else:
        df["projection_missing"] = df["proj_fpts"].isna() | df["proj_minutes"].isna()
        df["proj_fpts_filled"] = df["proj_fpts"].fillna(0.0)
        df["proj_minutes_filled"] = df["proj_minutes"].fillna(0.0)

        if source == "rg":
            df["floor_filled"] = df["proj_floor"].fillna(0.0)

    df = df[df["positions"].map(bool)]
    df = set_team_ranks(df, dk_df=dk_df)

    def infer_slate_games(frame: pd.DataFrame) -> int:
        # Infer slate size from unique teams (approx games = teams/2). Fallback to 8 if missing.
        teams = set(frame["team"].dropna().unique())

        if len(teams) % 2 != 0:
            raise ValueError("Number of teams must be even.")

        return len(teams) / 2

    slate_games = infer_slate_games(df)

    return df.reset_index(drop=True), slate_games


def load_projections(
    slate: SlateMeta,
    dk_df: pd.DataFrame = None,
    remove_nan: bool = True,
    locked_players: dict = {},
):
    etr_proj_path = (
        ETR_PROJ_DIR
        / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.datetime}.csv"
    )

    if not etr_proj_path.is_file():
        raise FileNotFoundError(f"ETR projection file not found '{etr_proj_path}'")

    rg_proj_path = (
        RG_PROJ_DIR
        / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.datetime}.csv"
    )

    if not rg_proj_path.is_file():
        raise FileNotFoundError(f"RG projection file not found '{rg_proj_path}'")

    rg_proj, slate_games = load_projection_csv(
        rg_proj_path, dk_df=dk_df, remove_nan=remove_nan, locked_players=locked_players
    )
    etr_proj, _ = load_projection_csv(
        etr_proj_path, dk_df=dk_df, remove_nan=remove_nan, locked_players=locked_players
    )

    return etr_proj, rg_proj, slate_games


def load_dk_history_csv(slate: ResultsFileMeta) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load post-slate history. Expected columns: id_col, actual_fpts.
    """
    dk_history_path = (
        DK_HISTORY_DIR
        / f"{slate.sport}_{slate.slate}_{slate.site}_history_{slate.datetime}.csv"
    )

    raw = pd.read_csv(dk_history_path, dtype=str)

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


def load_nba_box_scores_csv(date: str) -> pd.DataFrame:
    nba_box_scores_path = NBA_BOX_SCORES_DIR / f"{date}_nba_box_scores.csv"
    return pd.read_csv(nba_box_scores_path)


def normalize_columns(
    df: pd.DataFrame, required: Iterable[str] | None = None
) -> pd.DataFrame:
    """Rename columns to canonical names using aliases; optionally enforce required columns."""
    df = df.copy()
    renamed = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            renamed[canonical] = canonical
            continue
        match = next((col for col in df.columns if col in aliases), None)
        if match:
            renamed[match] = canonical
    df = df.rename(columns=renamed)

    if required:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def normalize_name(name: str) -> str:
    """Lowercase, strip, and remove punctuation/accents for simple matching."""
    if not isinstance(name, str):
        return ""

    normalized = ALIASES.get(name, name)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    cleaned = "".join(ch for ch in normalized if ch.isalnum())
    return cleaned.strip()


def parse_filename(path: str | Path) -> ResultsFileMeta:
    filename = Path(path).name
    match = _FILENAME_RE.match(filename)

    if not match:
        raise ValueError(
            f"Filename does not match expected format: {filename}\n"
            "Expected: {sport}_{slate}_{site}_{source}?_{datatype}_YYYY-MM-DD"
        )

    parts = match.groupdict()

    slate_id = f"{parts['sport']}_{parts['slate']}_{parts['site']}_{parts['datetime']}"

    return ResultsFileMeta(
        sport=parts["sport"],
        slate=parts["slate"],
        site=parts["site"],
        source=parts["source"],
        datatype=parts["datatype"],
        datetime=parts["datetime"],
        slate_id=slate_id,
    )


ET = ZoneInfo("America/New_York")
MT = ZoneInfo("America/Denver")


def parse_game_time(game_info: str):
    """
    Extract game_id and timezone-aware Eastern Time datetime from DK Game Info.
    Example: 'LAL@SAC 01/12/2026 10:00PM ET' -> ('LAL@SAC', datetime(..., tzinfo=ET))
    Returns None if parsing fails.
    """
    if not isinstance(game_info, str):
        return None
    parts = game_info.split()
    if len(parts) < 2:
        return None
    game_id = parts[0]
    m = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}[AP]M)", game_info)
    if not m:
        return None

    date_str, time_str = m.group(1), m.group(2)
    combined = f"{date_str} {time_str}"
    dt_naive = datetime.strptime(combined, "%m/%d/%Y %I:%M%p")
    dt_et = dt_naive.replace(tzinfo=ET)
    return game_id, dt_et.astimezone(MT)


def parse_positions(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    return {p.strip().upper() for p in raw.split("/") if p.strip()}


def parse_slate_id(slate_id: str) -> SlateMeta:
    match = _SLATE_ID_RE.match(slate_id)

    if not match:
        raise ValueError(
            f"Slate ID does not match expected format: {slate_id}\n"
            "Expected: {sport}_{slate}_{site}_YYYY-MM-DD"
        )

    parts = match.groupdict()

    return SlateMeta(
        sport=parts["sport"],
        slate=parts["slate"],
        site=parts["site"],
        datetime=parts["datetime"],
        id=slate_id,
    )


def print_table(
    df: pd.DataFrame,
    cols: Iterable[str],
    empty_msg: str = "No rows to display.",
    limit: int | None = None,
) -> None:
    if df.empty:
        print(empty_msg)
        return
    cols = list(cols)
    view = df[cols]
    if limit is not None:
        view = view.head(limit)
    print(view.to_string(index=False))


def set_team_ranks(df, team_top_n: int = 8, dk_df: pd.DataFrame | None = None):
    # Team projected totals: sum of top N projected players per team.
    if team_top_n and "team" in df.columns:
        top_n = max(int(team_top_n), 1)
        team_totals = (
            df.sort_values(["team", "proj_fpts"], ascending=[True, False])
            .groupby("team", as_index=True)
            .head(top_n)
            .groupby("team")["proj_fpts"]
            .sum()
            .sort_values(ascending=False)
        )
        team_rank = {team: rank for rank, team in enumerate(team_totals.index, start=1)}
        df["team_proj_rank"] = df["team"].map(team_rank)

        # Game projected totals: sum of team totals for teams in the same game.
        df["game_proj_rank"] = float("nan")
        team_game = None
        if "game_id" in df.columns:
            team_game = (
                df[["team", "game_id"]]
                .dropna()
                .groupby("team")["game_id"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            )
        elif (
            dk_df is not None and "team" in dk_df.columns and "game_id" in dk_df.columns
        ):
            team_game = (
                dk_df[["team", "game_id"]]
                .dropna()
                .groupby("team")["game_id"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            )

        if team_game is not None:
            team_game = team_game.dropna()
            if not team_game.empty:
                game_totals = (
                    team_totals.to_frame("team_total")
                    .join(team_game.rename("game_id"), how="inner")
                    .groupby("game_id")["team_total"]
                    .sum()
                    .sort_values(ascending=False)
                )
                game_rank = {
                    game_id: rank
                    for rank, game_id in enumerate(game_totals.index, start=1)
                }
                df["game_proj_rank"] = df["team"].map(team_game).map(game_rank)

    return df


def total_fragile_minutes(lineup_df: pd.DataFrame) -> float:
    return float(sum(max(0, 30 - m) for m in lineup_df["proj_minutes"]))
