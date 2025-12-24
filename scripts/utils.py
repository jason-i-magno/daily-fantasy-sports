"""
Shared helpers for DFS projection scripts.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

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


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric, stripping currency/commas where present."""
    cleaned = series.replace(r"[\\$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_output_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
