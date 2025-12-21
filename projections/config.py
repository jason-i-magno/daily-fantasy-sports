"""
Configuration for projections ingestion.

Defaults to a local SQLite database in the project root. Override via env or CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_DB_URL = os.environ.get("PROJECTIONS_DB_URL", "sqlite:///projections.db")


@dataclass
class ProviderConfig:
    """Lightweight holder for provider-specific settings."""

    csv_path: str | None = None
    url: str | None = None
    cookie: str | None = None  # User-supplied session cookie string for authenticated JSON


@dataclass
class AppConfig:
    db_url: str = DEFAULT_DB_URL
    etr: ProviderConfig = ProviderConfig()
    rg: ProviderConfig = ProviderConfig()
