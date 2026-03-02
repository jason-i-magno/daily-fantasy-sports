from __future__ import annotations

import time
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3, scoreboardv2
from utils import (
    NBA_BOX_SCORES_DIR,
    get_slates,
    load_dk_history_csv,
    load_nba_box_scores_csv,
    normalize_name,
)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def calcDkFpts(stats: Dict[str, Any]) -> float:
    """
    DraftKings NBA fantasy points from nba.cloud boxscoretraditional 'statistics' dict.
    """
    pts = _to_int(stats.get("points"))
    reb = _to_int(stats.get("reboundsTotal"))
    ast = _to_int(stats.get("assists"))
    stl = _to_int(stats.get("steals"))
    blk = _to_int(stats.get("blocks"))
    tov = _to_int(stats.get("turnovers"))
    tpm = _to_int(stats.get("threePointersMade"))

    base = (
        1.0 * pts
        + 1.25 * reb
        + 1.5 * ast
        + 2.0 * stl
        + 2.0 * blk
        - 0.5 * tov
        + 0.5 * tpm
    )

    # DD / TD bonuses (stack: triple-double gets +4.5 total)
    cats10 = sum(x >= 10 for x in (pts, reb, ast, stl, blk))
    bonus = 0.0
    if cats10 >= 2:
        bonus += 1.5
    if cats10 >= 3:
        bonus += 3.0

    return float(base + bonus)


def parseBoxscoreTraditionalToDf(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse nba.cloud boxscoretraditional payload into per-player rows + DK FPTS.
    """
    bs = data.get("boxScoreTraditional") or {}
    game_id = bs.get("gameId")

    rows: List[Dict[str, Any]] = []
    for side in ("homeTeam", "awayTeam"):
        team = bs.get(side) or {}
        team_tricode = team.get("teamTricode")
        team_id = team.get("teamId")

        for p in team.get("players") or []:
            stats = p.get("statistics") or {}
            player_name = f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()

            # Keep DNPs too; you can filter later if desired
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "team_tricode": team_tricode,
                    "person_id": p.get("personId"),
                    "player_name": player_name,
                    "player_key": normalize_name(player_name),
                    "position": p.get("position"),
                    "comment": p.get("comment"),
                    "minutes": (stats.get("minutes") or "").strip(),
                    "pts": _to_int(stats.get("points")),
                    "reb": _to_int(stats.get("reboundsTotal")),
                    "ast": _to_int(stats.get("assists")),
                    "stl": _to_int(stats.get("steals")),
                    "blk": _to_int(stats.get("blocks")),
                    "tov": _to_int(stats.get("turnovers")),
                    "tpm": _to_int(stats.get("threePointersMade")),
                    "nba_fpts_calc": calcDkFpts(stats),
                    "dk_fpts_calc": float("nan"),
                }
            )

    return pd.DataFrame(rows)


def gameIdsForDate(game_date_iso: str) -> Set[str]:
    """
    Return GAME_ID list for a date using nba_api scoreboardv2.

    game_date_iso: 'YYYY-MM-DD'
    """
    sb = scoreboardv2.ScoreboardV2(game_date=game_date_iso, day_offset=0)
    games = sb.game_header.get_data_frame()
    # Column is usually 'GAME_ID'
    return set(games["GAME_ID"].astype(str))


def buildNbaBoxScoresForDkHistory(
    dates,
    *,
    sleep_s: float = 1,
    overwrite: bool = False,
) -> None:
    """
    Loop over every file in dk_history_dir, parse date from filename,
    fetch box scores for every NBA game on that date, and write a CSV per date to out_dir.

    Output file: {YYYY-MM-DD}_nba_box_scores.csv
    """

    for date_iso in sorted(dates):
        print(date_iso)
        out_path = NBA_BOX_SCORES_DIR / f"{date_iso}_nba_box_scores.csv"
        if out_path.exists() and not overwrite:
            # Already built for this date
            continue

        try:
            game_ids = gameIdsForDate(date_iso)
        except Exception as e:
            print(f"[WARN] Failed to get game IDs for {date_iso}: {e}")
            continue

        if not game_ids:
            # No games that date
            print(f"[INFO] No NBA games on {date_iso}")
            continue

        all_players: List[pd.DataFrame] = []

        for gid in game_ids:
            print(gid)
            try:
                data = boxscoretraditionalv3.BoxScoreTraditionalV3(
                    game_id=gid
                ).get_dict()
                df_game = parseBoxscoreTraditionalToDf(data)
                df_game.insert(0, "game_date", date_iso)
                all_players.append(df_game)
            except Exception as e:
                print(f"[WARN] Failed game {gid} on {date_iso}: {e}")
            finally:
                if sleep_s > 0:
                    time.sleep(sleep_s)

        if not all_players:
            print(f"[WARN] No boxscore data collected for {date_iso}")
            continue

        df_out = pd.concat(all_players, ignore_index=True)
        # Optional: drop DNPs
        # df_out = df_out[df_out["minutes"].astype(str).str.len() > 0]

        df_out.to_csv(out_path, index=False)
        print(f"[OK] Wrote {len(df_out)} player rows for {date_iso} -> {out_path}")


def addDkOnlyPlayersToBoxScores(
    nba: pd.DataFrame,
    dk: pd.DataFrame,
    *,
    dk_name_col: str = "Player",
    dk_key_col: str = "player_key",
    dk_fpts_col: str = "FPTS",
) -> pd.DataFrame:
    """
    Return a new nba dataframe with extra rows for DK players missing from nba.
    The appended rows will have dk_fpts_calc populated and nba_fpts_calc left NaN.
    """
    nba_out = nba.copy()

    # Ensure player_key exists on both
    if "player_key" not in nba_out.columns:
        raise ValueError("nba df must have a 'player_key' column")

    if dk_key_col not in dk.columns:
        raise ValueError(f"dk df must have a '{dk_key_col}' column")

    nba_keys = set(nba_out["player_key"].dropna().astype(str))
    dk_keys = set(dk[dk_key_col].dropna().astype(str))

    missing_keys = sorted(dk_keys - nba_keys)
    if not missing_keys:
        return nba_out

    # Build a map for DK name + fpts for those missing keys
    dk_sub = dk.loc[dk[dk_key_col].astype(str).isin(missing_keys)].copy()

    # Prefer the name column if it exists; otherwise just reuse key as name
    if dk_name_col in dk_sub.columns:
        dk_sub["player_name"] = dk_sub[dk_name_col].astype(str)
    else:
        dk_sub["player_name"] = dk_sub[dk_key_col].astype(str)

    dk_sub["dk_fpts_calc"] = pd.to_numeric(dk_sub[dk_fpts_col], errors="coerce")

    # Create placeholder rows with the same columns as nba_out (so concat works)
    new_rows = pd.DataFrame(columns=nba_out.columns)

    # Required fields
    new_rows["player_key"] = dk_sub[dk_key_col].astype(str).values

    # If nba_out has player_name/person_id/game_id/etc, fill what we can
    if "player_name" in new_rows.columns:
        new_rows["player_name"] = dk_sub["player_name"].values
    if "person_id" in new_rows.columns:
        new_rows["person_id"] = np.nan
    if "nba_fpts_calc" in new_rows.columns:
        new_rows["nba_fpts_calc"] = np.nan
    if "dk_fpts_calc" in new_rows.columns:
        new_rows["dk_fpts_calc"] = dk_sub["dk_fpts_calc"].values

    # Optional marker column (highly recommended)
    if "source" in new_rows.columns:
        new_rows["source"] = "dk_only"

    # For any remaining columns, leave NaN / blanks
    nba_out = pd.concat([nba_out, new_rows], ignore_index=True)

    return nba_out


def validatePlayerFpts(
    dk_players_df: pd.DataFrame,
    nba_actuals_df: pd.DataFrame,
    *,
    fpts_col_dk: str = "FPTS",
    fpts_col_nba: str = "nba_fpts_calc",
    tol: float = 0.01,
    box_scores_path: str | None = None,
) -> pd.DataFrame:
    """
    Returns a validation dataframe with diffs and flags.
    Does NOT mutate inputs.
    """
    dk = dk_players_df.copy()
    nba = nba_actuals_df.copy()

    # Ensure numeric
    dk[fpts_col_dk] = pd.to_numeric(dk[fpts_col_dk], errors="coerce")

    nba_name_source = "player_name" if "player_name" in nba.columns else "player_key"

    nba["player_key"] = (
        nba[nba_name_source]
        .astype(str)
        .map(
            lambda s: normalize_name(s)
        )  # or normalize_name(canonicalizePlayerName(s))
    )

    merged = dk.merge(
        nba[["player_key", "person_id", "player_name", fpts_col_nba]],
        on="player_key",
        how="left",
        suffixes=("", "_nba"),
    )

    # Populate DK fpts from history file into placeholder column.
    merged["dk_fpts_calc"] = merged[fpts_col_dk]
    merged["fpts_diff"] = merged["dk_fpts_calc"] - merged[fpts_col_nba]

    # Update DK fpts in the box scores data (by player_key).
    dk_map = dk.set_index("player_key")[fpts_col_dk].to_dict()
    nba["dk_fpts_calc"] = nba["dk_fpts_calc"].where(
        ~nba["player_key"].isin(dk_map.keys()),
        nba["player_key"].map(dk_map),
    )
    nba = addDkOnlyPlayersToBoxScores(
        nba=nba,
        dk=dk,
        dk_name_col="Player",
        dk_key_col="player_key",
        dk_fpts_col=fpts_col_dk,
    )
    if box_scores_path:
        nba.to_csv(box_scores_path, index=False)
    merged["missing_in_nba"] = merged[fpts_col_nba].isna()
    merged["mismatch"] = (~merged["missing_in_nba"]) & (merged["fpts_diff"].abs() > tol)

    missing = merged.loc[
        merged["missing_in_nba"], ["Player", "player_key", "RosterPosition", "FPTS"]
    ].sort_values(["Player"])

    print(missing.to_string(index=False))
    print(f"\nMissing count: {len(missing)}")

    return merged


if __name__ == "__main__":
    NBA_BOX_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    slates = get_slates()
    dates = {slate.datetime for slate in slates}

    buildNbaBoxScoresForDkHistory(dates)

    n_slates = len(slates)
    slates.sort(key=lambda x: x.datetime)

    for i, slate in enumerate(slates):
        print(f"[{i + 1}/{n_slates}] {slate.slate_id}")
        _, dk_history_df = load_dk_history_csv(slate)
        nba_box_scores_path = (
            NBA_BOX_SCORES_DIR / f"{slate.datetime}_nba_box_scores.csv"
        )
        box_scores_df = load_nba_box_scores_csv(slate.datetime)

        validatePlayerFpts(
            dk_history_df,
            box_scores_df,
            box_scores_path=str(nba_box_scores_path),
        )
