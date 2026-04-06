"""Provider interface for projection ingestion."""

from __future__ import annotations

import abc
import datetime as dt
from typing import Any

import pandas as pd


class ProjectionProvider(abc.ABC):
    """Provider-agnostic contract for fetching and normalizing projections."""

    source: str  # short identifier, e.g., "etr" or "rg"

    @abc.abstractmethod
    def fetch_raw(self, slate_id: str) -> pd.DataFrame:
        """Retrieve raw data for a slate. May read from CSV or authenticated JSON."""

    @abc.abstractmethod
    def normalize(self, df: pd.DataFrame, slate_id: str) -> pd.DataFrame:
        """Return DataFrame with the common projection schema."""

    def _pulled_at(self) -> dt.datetime:
        return dt.datetime.utcnow()

    def _normalize_columns(
        self, df: pd.DataFrame, mapping: dict[str, str], required: set[str]
    ) -> pd.DataFrame:
        """Rename columns using mapping if present and ensure required columns exist."""
        renamed = {col: mapping.get(col, col) for col in df.columns}
        df = df.rename(columns=renamed)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after normalization: {missing}")
        return df

    def _coerce_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _coerce_str(self, series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip()
