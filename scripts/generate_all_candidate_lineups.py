import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

from scripts.generate_lineup import (
    generate_lineups,
)
from scripts.utils import (
    CANDIDATE_DIR_MAP,
    ETR_PROJ_DIR,
    MODELS,
    MT,
    RG_PROJ_DIR,
    SLOT_ORDER,
    STRATEGIES,
    ResultsFileMeta,
    SlateMeta,
    get_slates,
    lineup_df_to_player_keys,
    load_dk_salaries_csv,
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


def create_candidates_file(
    meta: ResultsFileMeta,
    slate_size: int,
    proj_source: str,
):

    payload = {
        "slate_id": meta.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "projection_source": proj_source,
        "slate_size (games)": slate_size,
    }

    out_path = CANDIDATE_DIR_MAP[proj_source]

    out_path.mkdir(parents=True, exist_ok=True)

    file_path = (
        out_path
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
    )
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    return file_path


def write_lineups_to_file(
    meta: ResultsFileMeta,
    model: str,
    strategy,
    lineups,
):
    lineups_keys = [lineup_df_to_player_keys(lu["lineup"]) for lu in lineups]

    out_path = CANDIDATE_DIR_MAP[model]

    out_path.mkdir(parents=True, exist_ok=True)

    file_path = (
        out_path
        / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
    )

    with open(file_path, "r") as f:
        data = json.load(f)

    # Add new field
    data[strategy] = lineups_keys

    # Write back (preserving formatting)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    existing = strategy in data
    if existing:
        print(f"[UPDATED] {file_path.name} (replaced {strategy})")
    else:
        print(f"[ADDED] {file_path.name} (created {strategy})")


# ----------------------------
# Entrypoint
# ----------------------------


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    slates = get_slates()
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

            for model in MODELS:
                candidates_path = (
                    CANDIDATE_DIR_MAP[model]
                    / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{meta.datetime}.json"
                )

                if not candidates_path.exists():
                    generate_candidates = True

                    logging.info(
                        f"Missing {model.upper()} candidate lineups file {candidates_path.name}"
                    )

                    create_candidates_file(
                        meta=meta,
                        slate_size=len(set(dk_salaries["game_info"])),
                        proj_source=model,
                    )

                for strategy in STRATEGIES:
                    # generate_candidates = True if game_time_idx > 0 else False
                    generate_candidates = False
                    valid_prev_candidates = True

                    with open(candidates_path, "r") as f:
                        data = json.load(f)

                    if strategy not in data:
                        generate_candidates = True

                        logging.info(
                            f"Missing {model.upper()} {strategy} lineups in {candidates_path.name}"
                        )

                    # n_top_lineups = len(data[strategy])

                    # if n_top_lineups < 100:
                    #     print(f"{n_top_lineups=}")
                    #     generate_candidates = True

                    # if "from_top_game" in strategy or "from_top_team" in strategy:
                    #     generate_candidates = True

                    locked_players = {}
                    game_time_filter = None

                    if prev_datetime:
                        prev_candidates_path = (
                            CANDIDATE_DIR_MAP[model]
                            / f"{meta.sport}_{meta.slate}_{meta.site}_candidate_lineups_{prev_datetime}.json"
                        )
                        with open(prev_candidates_path, "r") as f:
                            data = json.load(f)

                        game_time_filter = datetime.fromisoformat(
                            meta.datetime
                        ).astimezone(MT) - timedelta(minutes=1)

                        top_lineups_keys = data[strategy]

                        if len(top_lineups_keys) > 0:
                            locked_players = build_locked_players(
                                top_lineups_keys[0],
                                player_game_time,
                                game_time_filter,
                            )
                        else:
                            valid_prev_candidates = False

                    if generate_candidates:
                        if valid_prev_candidates:
                            logging.info(
                                f"Generating {model.upper()} candidate lineups {candidates_path.name}"
                            )

                            start_time = time.perf_counter()

                            n_force_top_projected = 0
                            if "force_top_proj" in strategy:
                                n_force_top_projected = int(strategy[-1])

                            force_sal_50000 = False
                            if "force_sal_50000" in strategy:
                                force_sal_50000 = True

                            min_top_team_players = 0
                            if "from_top_team" in strategy:
                                min_top_team_players = int(strategy.split("_")[2])

                            min_top_game_players = 0
                            if "from_top_game" in strategy:
                                min_top_game_players = int(strategy.split("_")[2])

                            strategy_lineups = generate_lineups(
                                slate_id=meta.id,
                                model=model,
                                k_lineups=args.k_lineups,
                                strategy=strategy,
                                locked_players=locked_players,
                                game_time_filter=game_time_filter,
                                use_db=False,
                                n_force_top_projected=n_force_top_projected,
                                force_sal_50000=force_sal_50000,
                                min_top_game_players=min_top_game_players,
                                min_top_team_players=min_top_team_players,
                            )

                            end_time = time.perf_counter()

                            logging.info(
                                f"Generated lineups in {end_time - start_time:.1f} seconds"
                            )

                            write_lineups_to_file(
                                meta=meta,
                                model=model,
                                strategy=strategy,
                                lineups=strategy_lineups,
                            )
                        else:
                            logging.info(
                                "Previous candidates null. Writing null candidates."
                            )

                            write_lineups_to_file(
                                meta=meta,
                                model=model,
                                strategy=strategy,
                                lineups=[],
                            )

            prev_datetime = meta.datetime


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
