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
    "ceiling": ("ceiling", "Ceiling", "CEILING", "ceil", "Ceil", "CEIL"),
    "floor": ("floor", "Floor", "FLOOR"),
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

# Path Constants
DATA_DIR = Path("data")

CANDIDATE_LINEUPS_DIR = DATA_DIR / "candidate_lineups"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_LINEUPS_DIR = DATA_DIR / "raw"

BLEND_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "blend"
BLEND_OUTPUT_DIR = PROCESSED_DIR / "blend"
DK_SALARIES_DIR = RAW_LINEUPS_DIR / "draftkings"
ETR_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "etr"
ETR_OUTPUT_DIR = PROCESSED_DIR / "etr"
ETR_PROJ_DIR = RAW_LINEUPS_DIR / "etr"
H2H_DIR = PROCESSED_DIR / "h2h"
RESULTS_DIR = RAW_LINEUPS_DIR / "history"
RG_CANDIDATE_DIR = CANDIDATE_LINEUPS_DIR / "rotogrinders"
RG_OUTPUT_DIR = PROCESSED_DIR / "rotogrinders"
RG_PROJ_DIR = RAW_LINEUPS_DIR / "rotogrinders"

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
RG_PROJ_COLS = [
    "player_name",
    "salary",
    "proj_minutes",
    "proj_fpts",
    "position",
    "ceiling",
    "floor",
    "team",
]
ETR_PROJ_COLS = [
    "player_name",
    "salary",
    "proj_minutes",
    "proj_fpts",
    "position",
    "ceiling",
    "team",
]


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


@dataclasses.dataclass(frozen=True)
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
    weight_rg: float = 0.4,
    weight_etr: float = 0.6,
) -> pd.DataFrame:
    rg = rg_df.rename(
        columns={
            "proj_fpts": "rg_fpts",
            "proj_minutes": "rg_minutes",
        }
    )

    etr = etr_df.rename(
        columns={
            "proj_fpts": "etr_fpts",
            "proj_minutes": "etr_minutes",
        }
    )

    # Merge on player_name (inner join ensures both sites project the player)
    blend = rg.merge(etr, on="player_key", how="inner")

    # Create blended + min columns
    blend["proj_fpts"] = weight_rg * blend["rg_fpts"] + weight_etr * blend["etr_fpts"]

    blend["proj_minutes"] = blend[["rg_minutes", "etr_minutes"]].min(axis=1)
    blend["ceiling"] = blend["ceiling_x"]
    blend["player_name"] = blend["player_name_x"]
    blend["position"] = blend["position_x"]
    blend["positions"] = blend["positions_x"]
    blend["salary"] = blend["salary_x"]
    blend["team"] = blend["team_x"]
    cols_to_drop = [
        "etr_fpts",
        "etr_minutes",
        "rg_fpts",
        "rg_minutes",
        "player_name_x",
        "player_name_y",
        "salary_x",
        "salary_y",
        "position_x",
        "position_y",
        "positions_x",
        "positions_y",
        "ceiling_x",
        "ceiling_y",
        "team_x",
        "team_y",
    ]
    blend = blend.drop(columns=cols_to_drop)

    return blend


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, stripping currency/commas where present."""
    cleaned = series.replace(r"[\\$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_output_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_proj_cols(proj_source: str) -> List[str]:
    if proj_source == "rg":
        return RG_PROJ_COLS
    elif proj_source == "etr" or proj_source == "blend":
        return ETR_PROJ_COLS
    else:
        raise ValueError("Invalid projection source.")


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

    cols = get_proj_cols(source)
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
    df["player_key"] = df["player_name"].map(normalize_name)

    if source == "rg":
        df["floor"] = coerce_numeric(df["floor"])

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

            if df.loc[mask, "ceiling"].isna().any():
                df.loc[mask, "ceiling"] = 0.0

            if source == "rg":
                if df.loc[mask, "floor"].isna().any():
                    df.loc[mask, "floor"] = 0.0

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
                    "ceiling": 0.0,
                }

                # RG-only column, optional
                if "floor" in df.columns:
                    new_row["floor"] = 0.0

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
            df["floor_filled"] = df["floor"].fillna(0.0)

    df = df[df["positions"].map(bool)]

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


def load_results_csv(slate: ResultsFileMeta) -> pd.DataFrame:
    """
    Load post-slate results. Expected columns: id_col, actual_fpts.
    """
    results_path = (
        RESULTS_DIR
        / f"{slate.sport}_{slate.slate}_{slate.site}_results_{slate.datetime}.csv"
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
    normalized = unicodedata.normalize("NFKD", name)
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


def total_fragile_minutes(lineup_df: pd.DataFrame) -> float:
    return float(sum(max(0, 30 - m) for m in lineup_df["proj_minutes"]))
