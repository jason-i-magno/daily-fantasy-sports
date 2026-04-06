#!/usr/bin/env python3
"""CLI entrypoint to ingest projections into SQLite."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from projections.config import AppConfig, ProviderConfig
from projections.db.models import PlayerProjection, ProjectionSource
from projections.db.session import get_session_maker
from projections.providers.etr import ETRProvider
from projections.providers.rg import RGProvider


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-id", required=True, help="Slate identifier, e.g., 2025-01-15-NBA.")
    parser.add_argument("--provider", required=True, choices=["etr", "rg", "all"], help="Provider to ingest.")
    parser.add_argument("--db-url", default=None, help="Override database URL (default sqlite:///projections.db).")
    parser.add_argument("--etr-csv", help="Path to ETR CSV.")
    parser.add_argument("--rg-csv", help="Path to RG CSV.")
    parser.add_argument("--etr-url", help="ETR JSON endpoint (requires cookie).")
    parser.add_argument("--rg-url", help="RG JSON endpoint (requires cookie).")
    parser.add_argument("--etr-cookie", help="Cookie header for ETR endpoint (if using --etr-url).")
    parser.add_argument("--rg-cookie", help="Cookie header for RG endpoint (if using --rg-url).")
    return parser.parse_args(argv)


def provider_instances(args: argparse.Namespace) -> List:
    providers = []
    if args.provider in ("etr", "all"):
        providers.append(
            ETRProvider(csv_path=args.etr_csv, url=args.etr_url, cookie=args.etr_cookie)
        )
    if args.provider in ("rg", "all"):
        providers.append(
            RGProvider(csv_path=args.rg_csv, url=args.rg_url, cookie=args.rg_cookie)
        )
    return providers


def upsert_projections(session: Session, df: pd.DataFrame) -> tuple[int, int]:
    """Insert or update by natural key; returns (inserted, skipped)."""
    inserted = 0
    skipped = 0
    for _, row in df.iterrows():
        stmt = (
            insert(PlayerProjection)
            .values(**row.to_dict())
            .on_conflict_do_update(
                index_elements=["slate_id", "player_name", "source"],
                set_={
                    "projection": row["projection"],
                    "ceiling": row.get("ceiling"),
                    "floor": row.get("floor"),
                    "minutes": row.get("minutes"),
                    "ownership": row.get("ownership"),
                    "pulled_at": row.get("pulled_at"),
                },
            )
        )
        result = session.execute(stmt)
        if result.rowcount:
            inserted += 1
        else:
            skipped += 1
    return inserted, skipped


def record_source(session: Session, slate_id: str, source: str, pulled_at) -> None:
    stmt = (
        insert(ProjectionSource)
        .values(slate_id=slate_id, source=source, pulled_at=pulled_at)
        .on_conflict_do_update(
            index_elements=["slate_id", "source"],
            set_={"pulled_at": pulled_at},
        )
    )
    session.execute(stmt)


def ingest_provider(session: Session, provider, slate_id: str) -> tuple[int, int]:
    raw = provider.fetch_raw(slate_id)
    normalized = provider.normalize(raw, slate_id)
    # basic validation
    required_cols = {"slate_id", "player_name", "team", "position", "projection", "source", "pulled_at"}
    missing = required_cols - set(normalized.columns)
    if missing:
        raise ValueError(f"{provider.source} normalized data missing columns: {missing}")

    inserted, skipped = upsert_projections(session, normalized)
    record_source(session, slate_id, provider.source, normalized["pulled_at"].max())
    return inserted, skipped


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    config = AppConfig()
    if args.db_url:
        config.db_url = args.db_url

    SessionLocal = get_session_maker(config.db_url)
    providers = provider_instances(args)
    if not providers:
        print("No providers selected.")
        return 1

    total_inserted = 0
    total_skipped = 0
    with SessionLocal() as session:
        for provider in providers:
            ins, skip = ingest_provider(session, provider, args.slate_id)
            total_inserted += ins
            total_skipped += skip
        session.commit()

    print(f"Ingest complete: inserted={total_inserted}, skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
