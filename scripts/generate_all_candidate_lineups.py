import logging
import sys
from datetime import datetime
from pathlib import Path

from generate_minutes_lineup import (
    generate_lineups,
)
from utils import (
    parse_filename,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    candidate_lineups_dir = data_dir / "candidate_lineups"
    processed_lineups_dir = data_dir / "processed"
    raw_lineups_dir = data_dir / "raw"

    blend_candidate_dir = candidate_lineups_dir / "blend"
    blend_output_dir = processed_lineups_dir / "blend"
    etr_candidate_dir = candidate_lineups_dir / "etr"
    etr_output_dir = processed_lineups_dir / "etr"
    etr_proj_dir = raw_lineups_dir / "etr"
    dk_salaries_dir = raw_lineups_dir / "draftkings"
    results_dir = raw_lineups_dir / "history"
    rg_candidate_dir = candidate_lineups_dir / "rotogrinders"
    rg_output_dir = processed_lineups_dir / "rotogrinders"
    rg_proj_dir = raw_lineups_dir / "rotogrinders"

    slates = []

    for results_file in results_dir.iterdir():
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
            dk_salaries_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_salaries_{slate.date}.csv"
        )

        if not dk_salaries_path.exists():
            logging.warning(f"Missing DK salaries file for slate {slate.slate_id}")

        etr_candidates_path = (
            etr_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        if not etr_candidates_path.exists():
            logging.info(f"Missing ETR candidate lineups for slate {slate.slate_id}")

        etr_proj_path = (
            etr_proj_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.date}.csv"
        )

        if not etr_proj_path.exists():
            logging.error(f"Missing ETR projection file {etr_proj_path}")
            sys.exit(1)

        rg_candidates_path = (
            rg_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        if not rg_candidates_path.exists():
            logging.info(f"Missing RG candidate lineups for slate {slate.slate_id}")

        rg_proj_path = (
            rg_proj_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.date}.csv"
        )

        if not rg_proj_path.exists():
            logging.error(f"Missing ETR projection file {rg_proj_path}")
            sys.exit(1)

        blend_candidates_path = (
            blend_candidate_dir
            / f"{slate.sport}_{slate.slate}_{slate.site}_candidate_lineups_{slate.date}.json"
        )

        generate_blend_candidates = False

        if blend_candidates_path.exists():
            last_modified = datetime.fromtimestamp(
                blend_candidates_path.stat().st_mtime
            )

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
                blend_output_dir
                / "f{slate.sport}_{slate.slate}_{slate.site}_blend_output_{slate.date}.csv"
            )

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="blend",
                k_lineups=100,
                maximize_fpts=True,
                write_candidate_lineups=True,
                output_file=blend_output_path,
            )

        generate_etr_candidates = False

        if etr_candidates_path.exists():
            last_modified = datetime.fromtimestamp(etr_candidates_path.stat().st_mtime)

            # if datetime.now() - last_modified > timedelta(days=1):
            #     logging.info(
            #         f"ETR candidate lineups are more than a day old for slate {slate.slate_id}"
            #     )
            #     generate_etr_candidates = True
        else:
            logging.info(f"Missing BLEND candidate lineups for slate {slate.slate_id}")

            generate_etr_candidates = True

        if generate_etr_candidates:
            etr_output_path = (
                etr_output_dir
                / "f{slate.sport}_{slate.slate}_{slate.site}_etr_output_{slate.date}.csv"
            )

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="etr",
                k_lineups=100,
                maximize_fpts=True,
                write_candidate_lineups=True,
                output_file=etr_output_path,
            )

        generate_rg_candidates = False

        if rg_candidates_path.exists():
            last_modified = datetime.fromtimestamp(rg_candidates_path.stat().st_mtime)

            # if datetime.now() - last_modified > timedelta(days=1):
            #     logging.info(
            #         f"RG candidate lineups are more than a day old for slate {slate.slate_id}"
            #     )
            #     generate_rg_candidates = True
        else:
            logging.info(f"Missing BLEND candidate lineups for slate {slate.slate_id}")

            generate_rg_candidates = True

        if generate_rg_candidates:
            rg_output_path = (
                rg_output_dir
                / "f{slate.sport}_{slate.slate}_{slate.site}_rg_output_{slate.date}.csv"
            )

            generate_lineups(
                slate_id=slate.slate_id,
                projection_source="rg",
                k_lineups=100,
                maximize_fpts=True,
                write_candidate_lineups=True,
                output_file=rg_output_path,
            )
