"""
Shared helpers for DFS projection scripts.
"""

from __future__ import annotations

import dataclasses
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd

# Canonical column names and common aliases found in provider exports.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "player_name": ("player_name", "player", "name", "PLAYER", "DKName"),
    "salary": ("salary", "Salary", "SALARY"),
    "proj_minutes": ("projected_minutes", "minutes", "Minutes", "MINUTES"),
    "proj_fpts": ("projected_fpts", "FPTS", "fpts"),
    "position": (
        "position",
        "positions",
        "pos",
        "Pos",
        "POS",
        "ROSTER POSITION",
        "Roster Position",
    ),
    "ceiling": ("ceiling", "Ceiling", "CEILING", "ceil", "Ceil", "CEIL"),
    "floor": ("floor", "Floor", "FLOOR"),
    "team": ("team", "Team", "TEAM"),
}

_FILENAME_RE = re.compile(
    r"""
    ^
    (?P<sport>[a-z]+)_
    (?P<slate>[a-z\-]+)_
    (?P<site>[a-z]+)
    (?:_(?P<source>[a-z]+))?
    _
    (?P<datatype>[a-z\-]+)_
    (?P<date>\d{4}-\d{2}-\d{2})
    $
    """,
    re.VERBOSE,
)


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
    date: str  # ISO yyyy-mm-dd
    slate_id: str


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


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, stripping currency/commas where present."""
    cleaned = series.replace(r"[\\$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_output_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_projection_csv(path: Path) -> tuple[pd.DataFrame, int, list[str]]:
    cols = []
    if "rotogrinders" in str(path):
        cols = [
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "ceiling",
            "floor",
            "team",
        ]
    elif "etr" in str(path):
        cols = [
            "player_name",
            "salary",
            "proj_minutes",
            "proj_fpts",
            "position",
            "ceiling",
            "team",
        ]
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

    if "rotogrinders" in str(path):
        df["floor"] = coerce_numeric(df["floor"])

    df = df.dropna(subset=cols)
    df = df[df["positions"].map(bool)]

    def infer_slate_games(frame: pd.DataFrame) -> int:
        # Infer slate size from unique teams (approx games = teams/2). Fallback to 8 if missing.
        teams = set(frame["team"].dropna().unique())

        if len(teams) % 2 != 0:
            raise ValueError("Number of teams must be even.")

        return len(teams) / 2

    slate_games = infer_slate_games(df)

    return df.reset_index(drop=True), slate_games, cols


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

    slate_id = f"{parts['sport']}_{parts['slate']}_{parts['site']}_{parts['date']}"

    return ResultsFileMeta(
        sport=parts["sport"],
        slate=parts["slate"],
        site=parts["site"],
        source=parts["source"],
        datatype=parts["datatype"],
        date=parts["date"],
        slate_id=slate_id,
    )


def parse_positions(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    return {p.strip().upper() for p in raw.split("/") if p.strip()}


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
