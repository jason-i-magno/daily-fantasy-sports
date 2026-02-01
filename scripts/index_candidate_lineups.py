import json
import logging
from pathlib import Path


def build_player_index(data):
    """
    Extract all player names from top_fpts + top_minutes,
    sort them for deterministic ordering, and assign unique integers.
    """
    players = set()

    for lineup in data.get("top_fpts", []):
        players.update(lineup)

    for lineup in data.get("top_minutes", []):
        players.update(lineup)

    sorted_players = sorted(players)
    # stable: alphabetical order, index starts at 1
    return {name: i for i, name in enumerate(sorted_players, start=1)}


def replace_with_indices(lineup_list, index_map):
    """
    Replace player-name lists with integer ID lists.
    """
    return [[index_map[p] for p in lineup] for lineup in lineup_list]


def convert_json(in_path, out_path):
    # Load original JSON
    with open(in_path, "r") as f:
        data = json.load(f)

    # Build index
    index_map = build_player_index(data)

    # Replace lineups with integer versions
    new_top_fpts = replace_with_indices(data.get("top_fpts", []), index_map)
    new_top_minutes = replace_with_indices(data.get("top_minutes", []), index_map)

    # Build output object
    out_data = {
        "slate_id": data.get("slate_id"),
        "generated_at": data.get("generated_at"),
        "projection_source": data.get("projection_source"),
        "slate_size (games)": data.get("slate_size (games)"),
        "player_index": index_map,
        "top_fpts": new_top_fpts,
        "top_minutes": new_top_minutes,
    }

    # Save
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Converted file written to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    candidate_lineups_dir = data_dir / "candidate_lineups"

    blend_candidate_dir = candidate_lineups_dir / "blend"
    etr_candidate_dir = candidate_lineups_dir / "etr"
    rg_candidate_dir = candidate_lineups_dir / "rotogrinders"

    candidate_paths = []

    for candidate_dir in [blend_candidate_dir, etr_candidate_dir, rg_candidate_dir]:
        for candidate_path in candidate_dir.iterdir():
            if not candidate_path.is_file():
                continue

            candidate_paths.append(candidate_path)

    n_paths = len(candidate_paths)

    for i, candidate_path in enumerate(candidate_paths):
        logging.info(f"[{i + 1}/{n_paths}] {candidate_path}")
        print(candidate_path.parent)

        convert_json(
            in_path=candidate_path,
            out_path=f"{candidate_path.parent}_indexed/{candidate_path.name}",
        )
