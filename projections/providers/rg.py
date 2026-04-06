"""RotoGrinders projections provider."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests

from projections.providers.base import ProjectionProvider


class RGProvider(ProjectionProvider):
    source = "rg"

    def __init__(self, csv_path: str | None = None, url: str | None = None, cookie: str | None = None):
        self.csv_path = csv_path
        self.url = url
        self.cookie = cookie

    def fetch_raw(self, slate_id: str) -> pd.DataFrame:
        if self.csv_path:
            return pd.read_csv(self.csv_path)
        if self.url:
            headers = {"Cookie": self.cookie} if self.cookie else {}
            resp = requests.get(self.url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return pd.DataFrame(data)
        raise ValueError("RGProvider needs a csv_path or url.")

    def normalize(self, df: pd.DataFrame, slate_id: str) -> pd.DataFrame:
        mapping = {
            "PLAYER": "player_name",
            "Player": "player_name",
            "Name": "player_name",
            "TEAM": "team",
            "Team": "team",
            "POS": "position",
            "Pos": "position",
            "MINUTES": "minutes",
            "Minutes": "minutes",
            "FPTS": "projection",
            "Projection": "projection",
            "CEIL": "ceiling",
            "Ceiling": "ceiling",
            "FLOOR": "floor",
            "Floor": "floor",
            "OWN": "ownership",
            "ProjOwn": "ownership",
            "Ownership": "ownership",
        }
        required = {"player_name", "team", "position", "projection"}
        df = self._normalize_columns(df, mapping, required)

        normalized = pd.DataFrame(
            {
                "slate_id": slate_id,
                "player_name": self._coerce_str(df["player_name"]),
                "team": self._coerce_str(df.get("team", "")),
                "position": self._coerce_str(df["position"]),
                "projection": self._coerce_numeric(df["projection"]),
                "ceiling": self._coerce_numeric(df.get("ceiling")),
                "floor": self._coerce_numeric(df.get("floor")),
                "minutes": self._coerce_numeric(df.get("minutes")),
                "ownership": self._coerce_numeric(df.get("ownership")),
                "source": self.source,
                "pulled_at": dt.datetime.utcnow(),
            }
        )
        return normalized
