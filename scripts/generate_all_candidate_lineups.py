import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from generate_minutes_lineup import (
    generate_lineups,
)
from utils import (
    BLEND_CANDIDATE_DIR,
    ETR_CANDIDATE_DIR,
    ETR_PROJ_DIR,
    MT,
    RESULTS_DIR,
    RG_CANDIDATE_DIR,
    RG_PROJ_DIR,
    SLOT_ORDER,
    ResultsFileMeta,
    SlateMeta,
    lineup_df_to_player_keys,
    load_dk_salaries_csv,
    parse_filename,
)


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--k-lineups", type=int, default=1, help="Number of lineups to generate"
    )
    parser.add_argument(
        "-patch",
        "--patch-candidate-lineups",
        action="store_true",
        help="Flag to patch candidate lineups.",
    )
    return parser.parse_args(argv)


# ----------------------------
# Helpers
# ----------------------------
def build_locked_players(
    player_keys: list[str],
    player_game_time: dict,
    lock_cutoff: datetime,
) -> dict[str, str]:
    locked = {}
    for player_key, slot in zip(player_keys, SLOT_ORDER):
        game_time = player_game_time.get(player_key)
        if game_time is None:
            continue  # safety
        if game_time <= lock_cutoff:
            locked[player_key] = slot
    return locked


def write_lineups_to_file(
    out_dir: str,
    meta: ResultsFileMeta,
    slate_size: int,
    proj_source: str,
    top_fpts_player_keys: list[list[int]],
    top_minutes_player_keys: list[list[int]],
    top_adjusted_player_keys: list[list[int]],
):
    payload = {
        "slate_id": meta.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "projection_source": proj_source,
        "slate_size (games)": slate_size,
        "top_fpts": top_fpts_player_keys,
        "top_minutes": top_minutes_player_keys,
        "top_adjusted": top_adjusted_player_keys,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = (
        out_path
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
    )
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    return file_path


# ----------------------------
# Entrypoint
# ----------------------------


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    slates = []

    for results_file in RESULTS_DIR.iterdir():
        if not results_file.is_file():
            continue

        meta = parse_filename(results_file.stem)

        if meta.sport != "nba":
            continue

        slates.append(meta)

    n_slates = len(slates)

    slates.sort(key=lambda x: x.datetime)

    for slate_idx, slate in enumerate(slates):
        logging.info(f"[{slate_idx + 1}/{n_slates}] {slate.slate_id}")

        dk_salaries = load_dk_salaries_csv(
            SlateMeta(
                sport=slate.sport,
                slate=slate.slate,
                site=slate.site,
                datetime=slate.datetime,
                id=slate.slate_id,
            )
        )

        game_times = sorted(
            dk_salaries.set_index("player_key")["game_time_local"].unique()
        )
        player_game_time = dk_salaries.set_index("player_key")[
            "game_time_local"
        ].to_dict()
        prev_datetime = None

        for game_time_idx in range(len(game_times)):
            lock_time = game_times[game_time_idx].strftime("%H%M")

            meta = SlateMeta(
                sport=slate.sport,
                slate=slate.slate,
                site=slate.site,
                datetime=slate.datetime
                if game_time_idx == 0
                else f"{slate.datetime}T{lock_time}",
                id=(
                    f"{slate.sport}_{slate.slate}_{slate.site}_{slate.datetime}"
                    if game_time_idx == 0
                    else f"{slate.sport}_{slate.slate}_{slate.site}_{slate.datetime}T{lock_time}"
                ),
            )
            print(meta.id)

            etr_proj_path = (
                ETR_PROJ_DIR
                / f"{meta.sport}_{meta.slate}_{meta.site}_etr_projections_{meta.datetime}.csv"
            )

            if not etr_proj_path.is_file():
                logging.info(f"Missing ETR projections {etr_proj_path.name}")
                break

            rg_proj_path = (
                RG_PROJ_DIR
                / f"{meta.sport}_{meta.slate}_{meta.site}_rg_projections_{meta.datetime}.csv"
            )

            if not rg_proj_path.is_file():
                logging.info(f"Missing RG projections {rg_proj_path.name}")
                break

            candidate_dirs = {
                "blend": BLEND_CANDIDATE_DIR,
                "etr": ETR_CANDIDATE_DIR,
                "rg": RG_CANDIDATE_DIR,
            }

            for proj_source in ["blend", "etr", "rg"]:
                candidates_path = (
                    candidate_dirs[proj_source]
                    / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
                )

                # generate_candidates = True if game_time_idx > 0 else False
                generate_candidates = False

                if not candidates_path.exists():
                    generate_candidates = True

                    logging.info(
                        f"Missing {proj_source.upper()} candidate lineups {candidates_path.name}"
                    )

                adjusted_locked_players = {}
                fpts_locked_players = {}
                minutes_locked_players = {}
                game_time_filter = None

                if prev_datetime:
                    prev_candidates_path = (
                        candidate_dirs[proj_source]
                        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{prev_datetime}.json"
                    )
                    with open(prev_candidates_path, "r") as f:
                        data = json.load(f)

                    game_time_filter = datetime.fromisoformat(meta.datetime).astimezone(
                        MT
                    ) - timedelta(minutes=1)

                    top_fpts = data["top_fpts"][0]
                    fpts_locked_players = build_locked_players(
                        top_fpts,
                        player_game_time,
                        game_time_filter,
                    )

                    top_minutes = data["top_minutes"][0]
                    minutes_locked_players = build_locked_players(
                        top_minutes,
                        player_game_time,
                        game_time_filter,
                    )

                    top_adjusted = data["top_adjusted"][0]
                    adjusted_locked_players = build_locked_players(
                        top_adjusted,
                        player_game_time,
                        game_time_filter,
                    )

                if generate_candidates:
                    logging.info(
                        f"Generating {proj_source.upper()} candidate lineups {candidates_path.name}"
                    )

                    start_time = time.perf_counter()

                    max_fpts_lineups = generate_lineups(
                        slate_id=meta.id,
                        projection_source=proj_source,
                        k_lineups=args.k_lineups,
                        strategy="max_fpts",
                        patch_candidate_lineups=args.patch_candidate_lineups,
                        locked_players=fpts_locked_players,
                        game_time_filter=game_time_filter,
                    )

                    max_minutes_lineups = generate_lineups(
                        slate_id=meta.id,
                        projection_source=proj_source,
                        k_lineups=args.k_lineups,
                        strategy="max_minutes",
                        patch_candidate_lineups=args.patch_candidate_lineups,
                        locked_players=minutes_locked_players,
                        game_time_filter=game_time_filter,
                    )

                    max_fpts_minutes_floor_lineups = generate_lineups(
                        slate_id=meta.id,
                        projection_source=proj_source,
                        k_lineups=args.k_lineups,
                        strategy="max_fpts_minutes_floor",
                        patch_candidate_lineups=args.patch_candidate_lineups,
                        locked_players=adjusted_locked_players,
                        game_time_filter=game_time_filter,
                    )

                    end_time = time.perf_counter()

                    logging.info(
                        f"Generated lineups in {end_time - start_time:.1f} seconds"
                    )

                    max_fpts_player_keys = [
                        lineup_df_to_player_keys(lu["lineup"])
                        for lu in max_fpts_lineups
                    ]

                    if len(max_fpts_player_keys) == 0:
                        max_fpts_player_keys = [top_fpts]

                    max_minutes_player_keys = [
                        lineup_df_to_player_keys(lu["lineup"])
                        for lu in max_minutes_lineups
                    ]

                    if len(max_minutes_player_keys) == 0:
                        max_minutes_player_keys = [top_minutes]

                    max_fpts_minutes_floor_player_keys = [
                        lineup_df_to_player_keys(lu["lineup"])
                        for lu in max_fpts_minutes_floor_lineups
                    ]

                    if len(max_fpts_minutes_floor_player_keys) == 0:
                        max_fpts_minutes_floor_player_keys = [top_adjusted]

                    if proj_source == "rg":
                        candidate_subdir = "rotogrinders"
                    elif proj_source == "etr":
                        candidate_subdir = "etr"
                    elif proj_source == "blend":
                        candidate_subdir = "blend"

                    write_lineups_to_file(
                        out_dir=f"data/candidate_lineups/{candidate_subdir}",
                        meta=meta,
                        slate_size=len(set(dk_salaries["game_info"])),
                        proj_source=proj_source,
                        top_fpts_player_keys=max_fpts_player_keys,
                        top_minutes_player_keys=max_minutes_player_keys,
                        top_adjusted_player_keys=max_fpts_minutes_floor_player_keys,
                    )

            prev_datetime = meta.datetime


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
