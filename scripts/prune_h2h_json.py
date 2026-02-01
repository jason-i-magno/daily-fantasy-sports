import json
from pathlib import Path

# ---- CONFIG ----
TOP_LEVEL_KEEP = {"fee_1", "fee_2", "fee_3"}
ENTRY_KEEP = {"username", "fantasyPoints", "lineup"}


def prune_dict(d: dict, allowed_keys: set) -> dict:
    return {k: v for k, v in d.items() if k in allowed_keys}


def clean_contest_json(input_path: Path) -> None:
    with open(input_path) as f:
        raw = json.load(f)

    cleaned = {"fee_1": {}, "fee_2": {}, "fee_3": {}}

    if len(raw["fee_1"]["lineup"]) == 0:
        print(f"File empty. Skipping {input_path.name}")

        return

    for fee in cleaned.keys():
        cleaned[fee]["entry_name"] = raw[fee]["entry_name"]
        cleaned[fee]["points"] = raw[fee]["points"]

        lineup = {}
        for player in raw[fee]["lineup"]["scorecards"]:
            lineup[player["rosterPosition"]] = player["displayName"]

        cleaned[fee]["lineup"] = lineup

    with open(f"data/processed/h2h/{input_path.name}", "w") as f:
        json.dump(cleaned, f, indent=2)


def main() -> int:
    raw_h2h_dir = Path("data/raw/h2h")

    for raw_h2h_file in raw_h2h_dir.iterdir():
        clean_contest_json(raw_h2h_file)


if __name__ == "__main__":
    raise SystemExit(main())
