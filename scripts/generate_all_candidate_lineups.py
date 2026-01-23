import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Iterable

from generate_minutes_lineup import (
    generate_lineups,
)
from utils import (
    BLEND_CANDIDATE_DIR,
    BLEND_OUTPUT_DIR,
    DK_SALARIES_DIR,
    ETR_CANDIDATE_DIR,
    ETR_OUTPUT_DIR,
    ETR_PROJ_DIR,
    RESULTS_DIR,
    RG_CANDIDATE_DIR,
    RG_OUTPUT_DIR,
    RG_PROJ_DIR,
    parse_filename,
)

# ----------------------------
# CLI
# ----------------------------


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--write-candidate-lineups",
        action="store_true",
        help="Flag to write candidate lineups to a file.",
    )
    parser.add_argument(
        "-patch",
        "--patch-candidate-lineups",
        action="store_true",
        help="Flag to patch candidate lineups.",
    )
    return parser.parse_args(argv)


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

    slates.sort(key=lambda x: x.date)

    for i, slate in enumerate(slates):
        logging.info(f"[{i + 1}/{n_slates}] {slate.slate_id}")

        dk_salaries_path = (
            DK_SALARIES_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_salaries_{slate.date}.csv"
        )

        if not dk_salaries_path.exists():
            logging.warning(f"Missing DK salaries file for slate {slate.slate_id}")

        etr_candidates_path = (
            ETR_CANDIDATE_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        if not etr_candidates_path.exists():
            logging.info(f"Missing ETR candidate lineups for slate {slate.slate_id}")

        etr_proj_path = (
            ETR_PROJ_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.date}.csv"
        )

        if not etr_proj_path.exists():
            logging.error(f"Missing ETR projection file {etr_proj_path}")
            sys.exit(1)

        rg_candidates_path = (
            RG_CANDIDATE_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        if not rg_candidates_path.exists():
            logging.info(f"Missing RG candidate lineups for slate {slate.slate_id}")

        rg_proj_path = (
            RG_PROJ_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.date}.csv"
        )

        if not rg_proj_path.exists():
            logging.error(f"Missing ETR projection file {rg_proj_path}")
            sys.exit(1)

        blend_candidates_path = (
            BLEND_CANDIDATE_DIR
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        k_lineups = 100
        generate_blend_candidates = False
        generate_etr_candidates = False
        generate_rg_candidates = False

        if blend_candidates_path.exists():
            last_modified = datetime.fromtimestamp(
                blend_candidates_path.stat().st_mtime
            )

            # with open(blend_candidates_path, "r") as f:
            #     data = json.load(f)

            # if "top_adjusted" not in data:
            #     logging.info(
            #         f"BLEND Top Adjusted lineups not found for slate {slate.slate_id}"
            #     )

            #     generate_blend_candidates = True

            # if datetime.now() - last_modified > timedelta(days=1):
            #     logging.info(
            #         f"BLEND candidate lineups are more than a day old for slate {slate.slate_id}"
            #     )
            #     generate_blend_candidates = True
        else:
            logging.info(f"Missing BLEND candidate lineups for slate {slate.slate_id}")

            generate_blend_candidates = True

        if generate_blend_candidates:
            blend_output_path = (
                BLEND_OUTPUT_DIR
                / "f{slate.sport}_{slate.slate}_{slate.site}_blend_output_{slate.date}.csv"
            )

            logging.info(
                f"Generating BLEND candidate lineups for slate {slate.slate_id}"
            )

            start_time = time.perf_counter()

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="blend",
                k_lineups=k_lineups,
                maximize_fpts=True,
                write_candidate_lineups=args.write_candidate_lineups,
                patch_candidate_lineups=args.patch_candidate_lineups,
                output_file=blend_output_path,
            )

            end_time = time.perf_counter()

            logging.info(f"Generated lineups in {end_time - start_time:.1f} seconds")

        if etr_candidates_path.exists():
            last_modified = datetime.fromtimestamp(etr_candidates_path.stat().st_mtime)

            # with open(etr_candidates_path, "r") as f:
            #     data = json.load(f)

            # if "top_adjusted" not in data:
            #     logging.info(
            #         f"ETR Top Adjusted lineups not found for slate {slate.slate_id}"
            #     )

            #     generate_etr_candidates = True

            # if datetime.now() - last_modified > timedelta(days=1):
            #     logging.info(
            #         f"ETR candidate lineups are more than a day old for slate {slate.slate_id}"
            #     )
            #     generate_etr_candidates = True
        else:
            logging.info(f"Missing ETR candidate lineups for slate {slate.slate_id}")

            generate_etr_candidates = True

        if generate_etr_candidates:
            etr_output_path = (
                ETR_OUTPUT_DIR
                / "f{slate.sport}_{slate.slate}_{slate.site}_etr_output_{slate.date}.csv"
            )

            logging.info(f"Generating ETR candidate lineups for slate {slate.slate_id}")

            start_time = time.perf_counter()

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="etr",
                k_lineups=k_lineups,
                maximize_fpts=True,
                write_candidate_lineups=args.write_candidate_lineups,
                patch_candidate_lineups=args.patch_candidate_lineups,
                output_file=etr_output_path,
            )

            end_time = time.perf_counter()

            logging.info(f"Generated lineups in {end_time - start_time:.1f} seconds")

        if rg_candidates_path.exists():
            last_modified = datetime.fromtimestamp(rg_candidates_path.stat().st_mtime)

            # with open(rg_candidates_path, "r") as f:
            #     data = json.load(f)

            # if "top_adjusted" not in data:
            #     logging.info(
            #         f"RG Top Adjusted lineups not found for slate {slate.slate_id}"
            #     )

            #     generate_rg_candidates = True

            # if datetime.now() - last_modified > timedelta(days=1):
            #     logging.info(
            #         f"RG candidate lineups are more than a day old for slate {slate.slate_id}"
            #     )
            #     generate_rg_candidates = True
        else:
            logging.info(f"Missing RG candidate lineups for slate {slate.slate_id}")

            generate_rg_candidates = True

        if generate_rg_candidates:
            rg_output_path = (
                RG_OUTPUT_DIR
                / "f{slate.sport}_{slate.slate}_{slate.site}_rg_output_{slate.date}.csv"
            )

            logging.info(f"Generating RG candidate lineups for slate {slate.slate_id}")

            start_time = time.perf_counter()

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="rg",
                k_lineups=k_lineups,
                maximize_fpts=True,
                write_candidate_lineups=args.write_candidate_lineups,
                patch_candidate_lineups=args.patch_candidate_lineups,
                output_file=rg_output_path,
            )

            end_time = time.perf_counter()

            logging.info(f"Generated lineups in {end_time - start_time:.1f} seconds")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
