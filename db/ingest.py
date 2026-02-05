#!/usr/bin/env python3
"""
Ingest players from historical NBA DK results into the player table.

For each CSV in data/raw/history with 'nba' in the filename:
- Parse metadata from filename (sport, slate, site, date).
- Load results using scripts.utils.load_results_csv (handles column parsing).
- Normalize player names with scripts.utils.normalize_name.
- Insert unique player keys into the player table (ignore existing).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# Ensure project root on path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.models import Contest, Player, Projection, Salary, Source  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from scripts.utils import (  # noqa: E402
    ETR_PROJ_DIR,
    MT,
    RESULTS_DIR,
    RG_PROJ_DIR,
    load_dk_salaries_csv,
    load_projection_csv,
    load_results_csv,
    normalize_name,
    parse_filename,
    parse_game_time,
)


def collect_player_keys() -> Set[str]:
    """Scan history CSVs and collect normalized player keys."""
    player_keys: Set[str] = set()
    for results_file in RESULTS_DIR.iterdir():
        if not results_file.is_file():
            continue
        if "nba" not in results_file.name.lower():
            continue
        meta = parse_filename(results_file.stem)
        # load_results_csv expects ResultsFileMeta and reads from RESULTS_DIR
        _, players_df = load_results_csv(meta)
        if players_df.empty:
            continue
        for name in players_df["Player"]:
            key = normalize_name(name)
            if key:
                player_keys.add(key)

    return player_keys


def get_or_create_source(session, name: str) -> Source:
    source = session.execute(
        select(Source).where(Source.name == name)
    ).scalar_one_or_none()

    if source:
        return source

    source = Source(name=name)
    session.add(source)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        # Someone else inserted it
        source = session.execute(select(Source).where(Source.name == name)).scalar_one()

    return source


def upsert_players(session, keys: Set[str]) -> int:
    """Insert any missing player keys; return count inserted."""
    inserted = 0
    existing = {row[0] for row in session.execute(select(Player.key)).all()}
    new_keys = keys - existing
    for key in new_keys:
        session.add(Player(key=key))
    try:
        session.commit()
        inserted = len(new_keys)
    except IntegrityError:
        session.rollback()
        # Fallback in case of race/duplicates: insert one by one
        for key in new_keys:
            try:
                session.add(Player(key=key))
                session.commit()
                inserted += 1
            except IntegrityError:
                session.rollback()
    return inserted


def projections_available(meta, game_times):
    """
    Check that projection files exist for each game time (handles dated and dated+time filenames).
    Returns True if all are present in both ETR and RG dirs.
    """
    if not game_times:
        return False

    times_sorted = sorted(game_times)
    earliest = times_sorted[0]

    def has_file(base_dir, prefix):
        for gt in times_sorted:
            # Salary files store times in ET; projection files are timestamped in MT.
            suffix_time_mt = gt.astimezone(MT).strftime("%Y-%m-%dT%H%M")
            suffix_date = gt.strftime("%Y-%m-%d")
            if gt == earliest:
                # Earliest game: projections often published as date-only (no time suffix)
                path = (
                    base_dir
                    / f"{meta.sport}_{meta.slate}_{meta.site}_{prefix}_{suffix_date}.csv"
                )
            else:
                # Later games: expect time-stamped projection files (stored in MT)
                path = (
                    base_dir
                    / f"{meta.sport}_{meta.slate}_{meta.site}_{prefix}_{suffix_time_mt}.csv"
                )
            if not path.exists():
                return False
        return True

    etr_ok = has_file(ETR_PROJ_DIR, "etr_projections")
    rg_ok = has_file(RG_PROJ_DIR, "rg_projections")
    return etr_ok and rg_ok


def get_or_create_player(session, name: str) -> Player | None:
    """Ensure a Player row exists for the normalized name; return Player or None."""
    key = normalize_name(name)
    if not key:
        return None
    player = session.execute(
        select(Player).where(Player.key == key)
    ).scalar_one_or_none()
    if player:
        return player
    player = Player(key=key)
    session.add(player)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        player = session.execute(select(Player).where(Player.key == key)).scalar_one()
    return player


def find_projection_file(meta, base_dir: Path, prefix: str) -> Path | None:
    """
    Locate a projection file matching the slate/date. Prefer files with time suffix; fall back to date-only.
    """
    date_only = meta.datetime.split("T")[0]
    candidates = sorted(
        base_dir.glob(
            f"{meta.sport}_{meta.slate}_{meta.site}_{prefix}_{date_only}*.csv"
        )
    )
    if not candidates:
        return None
    return candidates[-1]  # choose latest (e.g., with time suffix if present)


def projection_path_for_time(meta, base_dir: Path, prefix: str, game_time, earliest):
    """
    Build projection filename for a specific game time.
    Earliest game often has date-only files; later games use time suffix (projection files in MT).
    """
    suffix_date = game_time.strftime("%Y-%m-%d")
    if game_time == earliest:
        return (
            base_dir
            / f"{meta.sport}_{meta.slate}_{meta.site}_{prefix}_{suffix_date}.csv"
        )
    suffix_time_mt = game_time.astimezone(MT).strftime("%Y-%m-%dT%H%M")
    return (
        base_dir
        / f"{meta.sport}_{meta.slate}_{meta.site}_{prefix}_{suffix_time_mt}.csv"
    )


def ingest_projection_source(
    session, contest_id: int, proj_path: Path, source_name: str, dk_df
):
    """Load a projection CSV and insert rows; return (inserted_count, snapshot_time)."""
    if proj_path is None or not proj_path.exists():
        return

    proj_meta = parse_filename(proj_path.stem)
    snapshot_time = datetime.fromisoformat(proj_meta.datetime.replace("T", " "))

    source = get_or_create_source(session, source_name)
    proj_df, _ = load_projection_csv(proj_path, dk_df=dk_df, remove_nan=True)

    for _, row in proj_df.iterrows():
        player = get_or_create_player(session, row["player_name"])
        if not player:
            continue
        proj_row = Projection(
            contest_id=contest_id,
            player_id=player.id,
            source_id=source.id,
            snapshot_time=snapshot_time,
            proj_minutes=float(row["proj_minutes"]),
            proj_fpts=float(row["proj_fpts"]),
        )
        session.add(proj_row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            continue


def ingest_salaries(session, contest_id: int, salaries_df: pd.DataFrame) -> int:
    """
    Insert Salary rows for a contest. Returns count inserted; duplicates are skipped.
    """
    inserted = 0
    for _, row in salaries_df.iterrows():
        player = get_or_create_player(session, row["player_name"])
        if not player:
            continue
        game_id = row.get("game_id") or ""
        if not game_id:
            game_parse = parse_game_time(row.get("game_info"))
            game_id = game_parse[0] if game_parse else ""
        salary_row = Salary(
            contest_id=contest_id,
            player_id=player.id,
            salary=int(row["salary"]),
            position=str(row["position"]),
            team=str(row["team"]),
            game=game_id,
        )
        session.add(salary_row)
        try:
            session.commit()
            inserted += 1
        except IntegrityError:
            session.rollback()
            # already present
            continue
    return inserted


def ingest_contests(session) -> int:
    inserted = 0
    for results_file in RESULTS_DIR.iterdir():
        if not results_file.is_file() or "nba" not in results_file.name.lower():
            continue
        meta = parse_filename(results_file.stem)
        date_only = meta.datetime.split("T")[0]
        contest_dt = datetime.fromisoformat(date_only)

        try:
            salaries = load_dk_salaries_csv(meta)
        except FileNotFoundError as exc:
            logging.warning(str(exc))
            continue
        except ValueError as exc:
            logging.error(f"{exc}")
            continue

        game_ids = {gid for gid in salaries["game_id"] if gid}
        game_times = {gt for gt in salaries["game_time_local"] if gt}

        slate_games = len(game_ids)
        has_late_swaps = len(game_times) > 1
        missing_late_swap_projections = False
        if has_late_swaps:
            missing_late_swap_projections = not projections_available(meta, game_times)

        contest = session.execute(
            select(Contest).where(
                Contest.date == contest_dt, Contest.slate_type == meta.slate
            )
        ).scalar_one_or_none()
        if not contest:
            contest = Contest(
                slate_type=meta.slate,
                slate_games=slate_games,
                date=contest_dt,
                has_late_swaps=has_late_swaps,
                missing_late_swap_projections=missing_late_swap_projections,
            )
            session.add(contest)
            session.commit()
            inserted += 1

        salaries_inserted = ingest_salaries(session, contest.id, salaries)

        # Ingest projections (rg, etr), handling late swaps per game time when applicable
        if has_late_swaps and not missing_late_swap_projections and game_times:
            times_sorted = sorted(game_times)
            earliest = times_sorted[0]
            for gt in times_sorted:
                rg_path = projection_path_for_time(
                    meta, RG_PROJ_DIR, "rg_projections", gt, earliest
                )
                etr_path = projection_path_for_time(
                    meta, ETR_PROJ_DIR, "etr_projections", gt, earliest
                )
                ingest_projection_source(session, contest.id, rg_path, "rg", salaries)
                ingest_projection_source(session, contest.id, etr_path, "etr", salaries)
        else:
            rg_path = find_projection_file(meta, RG_PROJ_DIR, "rg_projections")
            etr_path = find_projection_file(meta, ETR_PROJ_DIR, "etr_projections")

            ingest_projection_source(session, contest.id, rg_path, "rg", salaries)
            ingest_projection_source(session, contest.id, etr_path, "etr", salaries)

        logging.info(
            "Contest %s %s: salaries=%s (late_swaps=%s)",
            meta.slate,
            contest_dt.date(),
            salaries_inserted,
            has_late_swaps,
        )

    session.commit()
    return inserted


def main() -> int:
    session = SessionLocal()

    get_or_create_source(session, "etr")
    get_or_create_source(session, "rg")

    keys = collect_player_keys()
    if not keys:
        print("No player keys found to ingest.")
        return 0
    inserted = upsert_players(session, keys)
    print(f"Ingest complete. Inserted {inserted} new players.")
    contests_inserted = ingest_contests(session)
    print(f"Contest ingest complete. Inserted {contests_inserted} contests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
