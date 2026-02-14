from __future__ import annotations

import logging

from utils import (
    ETR_PROJ_DIR,
    RG_PROJ_DIR,
    SlateMeta,
    get_slates,
    load_dk_salaries_csv,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    slates = get_slates()
    n_slates = len(slates)

    slates.sort(key=lambda x: x.datetime)

    for i, slate in enumerate(slates):
        print(f"[{i + 1}/{n_slates}] {slate.slate_id}")

        meta = SlateMeta(
            sport=slate.sport,
            slate=slate.slate,
            site=slate.site,
            datetime=slate.datetime,
            id=slate.slate_id,
        )
        dk_salaries = load_dk_salaries_csv(meta)
        game_times = sorted(
            dk_salaries.set_index("player_key")["game_time_local"].unique()
        )

        for i in range(1, len(game_times)):
            lock_time = game_times[i].strftime("%H%M")

            if slate.slate == "turbo-1":
                slate_name = f"turbo-{i + 1}"
            else:
                slate_name = f"{slate.slate}-{i}"

            etr_proj_path = (
                ETR_PROJ_DIR
                / f"{slate.sport}_{slate_name}_{slate.site}_etr_projections_{slate.datetime}.csv"
            )

            if etr_proj_path.is_file():
                etr_proj_path_new = (
                    ETR_PROJ_DIR
                    / f"{slate.sport}_{slate.slate}_{slate.site}_etr_projections_{slate.datetime}T{lock_time}.csv"
                )

                etr_proj_path.rename(etr_proj_path_new)

                logging.info(
                    f"ETR projection file renamed from '{etr_proj_path}' to '{etr_proj_path_new}'"
                )
            else:
                logging.info(f"ETR projection file not found '{etr_proj_path}'")

            rg_proj_path = (
                RG_PROJ_DIR
                / f"{slate.sport}_{slate_name}_{slate.site}_rg_projections_{slate.datetime}.csv"
            )

            if rg_proj_path.is_file():
                rg_proj_path_new = (
                    RG_PROJ_DIR
                    / f"{slate.sport}_{slate.slate}_{slate.site}_rg_projections_{slate.datetime}T{lock_time}.csv"
                )

                rg_proj_path.rename(rg_proj_path_new)

                logging.info(
                    f"RG projection file renamed from '{rg_proj_path}' to '{rg_proj_path_new}'"
                )
            else:
                logging.info(f"RG projection file not found '{rg_proj_path}'")
