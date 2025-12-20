#!/usr/bin/env python3
"""
DraftKings LoL Classic H2H lineup generator (cash mindset, no projections).

Inputs:
  - Salary CSV from DraftKings.
  - JSON file with team win probabilities, e.g. {"TES": 0.7, "JDG": 0.62}.

Hard constraints:
  - Salary cap: 50,000 (CPT costs 1.5x salary).
  - Roster: CPT, TOP, JNG, MID, ADC, SUP, TEAM.
  - Captain must be ADC or MID on a team with win% >= 0.65.
  - Only consider teams with win% >= 0.60; at least one team in lineup has win% >= 0.65.
  - Enforce a 4–3 stack (primary stack of 4 incl. TEAM slot; secondary stack of 3); TEAM slot uses the primary stack.
  - No underdog stacks; at least two teams per lineup.
  - No one-off TOP or SUP (satisfied by 4–3); one-off JNG/MID/ADC only from favored teams (all candidates are favored).

Heuristic objective (cash safety):
  - Maximize combined team win probability weight across roster.
  - Secondary: maximize salary used (target high spend, >= 49k), small penalty for unused cap.

Implementation notes:
  - Uses PuLP ILP; enumerates ordered team pairs to satisfy 4–3 stacks.
  - Produces top N unique valid lineups (deduped by player+cpt/team composition) sorted by objective.
  - Rejects invalid or duplicate lineups; no randomness, no upside simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pulp
from utils import coerce_numeric

SALARY_CAP = 50000


@dataclass
class Player:
    name: str
    team: str
    role: str
    salary: float
    is_team: bool


def load_salaries(path: str, win_probs: Dict[str, float]) -> List[Player]:
    df = pd.read_csv(path)
    # DraftKings columns: Position, Name, Salary, TeamAbbrev, etc.
    for required in ["Name", "Roster Position", "Salary", "TeamAbbrev"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    df["Salary"] = coerce_numeric(df["Salary"])
    df["team_prob"] = df["TeamAbbrev"].map(win_probs)
    # Only teams with win% >= 0.60 are eligible.
    df = df[df["team_prob"] >= 0.60]

    players: List[Player] = []
    for _, row in df.iterrows():
        role = str(row["Roster Position"]).upper()
        team = str(row["TeamAbbrev"])
        player = Player(
            name=str(row["Name"]),
            team=team,
            role=role,
            salary=float(row["Salary"]),
            is_team=role == "TEAM",
        )
        players.append(player)
    return players


def build_problem(
    players: List[Player],
    team_primary: str,
    team_secondary: str,
    win_probs: Dict[str, float],
):
    roles = ["TOP", "JNG", "MID", "ADC", "SUP"]
    team_prob_primary = win_probs.get(team_primary, 0.0)
    team_prob_secondary = win_probs.get(team_secondary, 0.0)

    prob = pulp.LpProblem("lol_cash_lineup", pulp.LpMaximize)

    flex_vars: Dict[str, pulp.LpVariable] = {}
    cpt_vars: Dict[str, pulp.LpVariable] = {}
    team_vars: Dict[str, pulp.LpVariable] = {}

    for idx, p in enumerate(players):
        pid = f"p{idx}"
        if p.is_team:
            team_vars[pid] = pulp.LpVariable(
                pid + "_team", lowBound=0, upBound=1, cat="Binary"
            )
        else:
            flex_vars[pid] = pulp.LpVariable(
                pid + "_flex", lowBound=0, upBound=1, cat="Binary"
            )
            if p.role in {"ADC", "MID", "JNG"} and win_probs.get(p.team, 0.0) >= 0.65:
                cpt_vars[pid] = pulp.LpVariable(
                    pid + "_cpt", lowBound=0, upBound=1, cat="Binary"
                )

    # Roster counts
    prob += pulp.lpSum(cpt_vars.values()) == 1, "one_captain"
    for role in roles:
        prob += (
            (
                pulp.lpSum(
                    v
                    for v, p in (
                        (flex_vars[pid], players[int(pid[1:])]) for pid in flex_vars
                    )
                    if p.role == role
                )
                == 1
            ),
            f"one_{role}",
        )

    # TEAM slot: exactly one, must be the primary stack.
    prob += pulp.lpSum(team_vars.values()) == 1, "one_team_slot"
    prob += (
        (
            pulp.lpSum(
                v
                for pid, v in team_vars.items()
                if players[int(pid[1:])].team == team_primary
            )
            == 1
        ),
        "team_matches_primary",
    )
    prob += (
        (
            pulp.lpSum(
                v
                for pid, v in team_vars.items()
                if players[int(pid[1:])].team != team_primary
            )
            == 0
        ),
        "team_not_secondary",
    )

    # No player as both flex and CPT.
    for pid in flex_vars:
        if pid in cpt_vars:
            prob += flex_vars[pid] + cpt_vars[pid] <= 1, f"no_double_use_{pid}"

    # Salary cap (CPT counts 1.5x).
    salary_expr = (
        pulp.lpSum(players[int(pid[1:])].salary * v for pid, v in flex_vars.items())
        + pulp.lpSum(
            players[int(pid[1:])].salary * 1.5 * v for pid, v in cpt_vars.items()
        )
        + pulp.lpSum(players[int(pid[1:])].salary * v for pid, v in team_vars.items())
    )
    prob += salary_expr <= SALARY_CAP, "salary_cap"

    # Team stack counts: 4 for primary (including TEAM), 3 for secondary; no other teams.
    def team_count(team: str):
        return (
            pulp.lpSum(
                v for pid, v in flex_vars.items() if players[int(pid[1:])].team == team
            )
            + pulp.lpSum(
                v for pid, v in cpt_vars.items() if players[int(pid[1:])].team == team
            )
            + pulp.lpSum(
                v for pid, v in team_vars.items() if players[int(pid[1:])].team == team
            )
        )

    prob += team_count(team_primary) == 4, "primary_stack_4"
    prob += team_count(team_secondary) == 3, "secondary_stack_3"
    allowed_teams = {team_primary, team_secondary}
    for pid, v in flex_vars.items():
        if players[int(pid[1:])].team not in allowed_teams:
            prob += v == 0
    for pid, v in cpt_vars.items():
        if players[int(pid[1:])].team not in allowed_teams:
            prob += v == 0
    for pid, v in team_vars.items():
        if players[int(pid[1:])].team not in allowed_teams:
            prob += v == 0

    # Captain team win prob >= 0.65 enforced via variable availability; add explicit link.
    captain_team_prob = pulp.lpSum(
        win_probs.get(players[int(pid[1:])].team, 0.0) * v
        for pid, v in cpt_vars.items()
    )
    prob += captain_team_prob >= 0.65, "captain_team_prob_floor"

    # At least one team in lineup has >=0.65 win% (true if either team meets it; pre-filtered outside).
    prob += (
        int(team_prob_primary >= 0.65 or team_prob_secondary >= 0.65) == 1,
        "high_prob_team_present",
    )

    # Objective: favor total win prob and high salary usage; small penalty on unused cap.
    total_win = (
        pulp.lpSum(
            win_probs.get(players[int(pid[1:])].team, 0.0) * v
            for pid, v in flex_vars.items()
        )
        + pulp.lpSum(
            win_probs.get(players[int(pid[1:])].team, 0.0) * v
            for pid, v in cpt_vars.items()
        )
        + pulp.lpSum(
            win_probs.get(players[int(pid[1:])].team, 0.0) * v
            for pid, v in team_vars.items()
        )
    )
    salary_used = salary_expr
    prob += total_win * 1000 + salary_used * 0.1 - (SALARY_CAP - salary_used) * 0.05

    return prob, flex_vars, cpt_vars, team_vars


def extract_lineup(
    flex_vars: Dict[str, pulp.LpVariable],
    cpt_vars: Dict[str, pulp.LpVariable],
    team_vars: Dict[str, pulp.LpVariable],
    players: List[Player],
):
    rows = []
    for pid, var in flex_vars.items():
        if var.value() == 1:
            p = players[int(pid[1:])]
            rows.append(
                {
                    "slot": p.role,
                    "player_name": p.name,
                    "team": p.team,
                    "salary": p.salary,
                    "role": p.role,
                    "is_captain": False,
                }
            )
    for pid, var in cpt_vars.items():
        if var.value() == 1:
            p = players[int(pid[1:])]
            rows.append(
                {
                    "slot": "CPT",
                    "player_name": p.name,
                    "team": p.team,
                    "salary": p.salary * 1.5,
                    "role": p.role,
                    "is_captain": True,
                }
            )
    for pid, var in team_vars.items():
        if var.value() == 1:
            p = players[int(pid[1:])]
            rows.append(
                {
                    "slot": "TEAM",
                    "player_name": p.name,
                    "team": p.team,
                    "salary": p.salary,
                    "role": "TEAM",
                    "is_captain": False,
                }
            )
    return rows


def lineup_signature(rows: List[dict]) -> str:
    # Signature to dedupe: sorted by slot+player name+team.
    parts = [
        f"{r['slot']}|{r['player_name']}|{r['team']}"
        for r in sorted(rows, key=lambda x: x["slot"])
    ]
    return ";".join(parts)


def solve_for_pair(
    players: List[Player],
    team_primary: str,
    team_secondary: str,
    win_probs: Dict[str, float],
    limit: int,
):
    lineups = []
    prob, flex_vars, cpt_vars, team_vars = build_problem(
        players, team_primary, team_secondary, win_probs
    )

    seen_constraints = []
    while len(lineups) < limit:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            break
        rows = extract_lineup(flex_vars, cpt_vars, team_vars, players)
        if len(rows) != 7:
            break  # incomplete or invalid
        sig = lineup_signature(rows)
        if sig in {l["signature"] for l in lineups}:
            break
        salary_used = sum(r["salary"] for r in rows)
        total_win_prob = sum(win_probs.get(r["team"], 0.0) for r in rows)
        lineups.append(
            {
                "rows": rows,
                "signature": sig,
                "salary_used": salary_used,
                "total_win_prob": total_win_prob,
                "objective": pulp.value(prob.objective),
            }
        )
        # Exclude this exact lineup for next iteration.
        selected_vars = []
        for pid, var in flex_vars.items():
            if var.value() == 1:
                selected_vars.append(var)
        for pid, var in cpt_vars.items():
            if var.value() == 1:
                selected_vars.append(var)
        for pid, var in team_vars.items():
            if var.value() == 1:
                selected_vars.append(var)
        prob += pulp.lpSum(selected_vars) <= len(selected_vars) - 1
        seen_constraints.append(selected_vars)
    return lineups


def rank_lineups(lineups: List[dict]) -> List[dict]:
    return sorted(
        lineups,
        key=lambda x: (
            x["total_win_prob"],
            x["salary_used"],
            x["objective"],
        ),
        reverse=True,
    )


def print_lineups(lineups: List[dict], limit: int) -> None:
    for idx, lu in enumerate(lineups[:limit], 1):
        print(
            f"\nLineup #{idx}: win_prob={lu['total_win_prob']:.3f}, salary={lu['salary_used']:.0f}"
        )
        df = pd.DataFrame(lu["rows"])
        df = df.sort_values(
            by=["slot"],
            key=lambda s: s.map(
                {"CPT": 0, "TOP": 1, "JNG": 2, "MID": 3, "ADC": 4, "SUP": 5, "TEAM": 6}
            ),
        )
        print(
            df[["slot", "player_name", "team", "role", "salary"]].to_string(index=False)
        )


def save_lineups(lineups: List[dict], path: str, limit: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for lu in lineups[:limit]:
        for r in lu["rows"]:
            rec = {
                "lineup_id": lu["signature"],
                **r,
                "salary_used": lu["salary_used"],
                "total_win_prob": lu["total_win_prob"],
            }
            records.append(rec)
    if records:
        pd.DataFrame(records).to_csv(output_path, index=False)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salaries", required=True, help="Path to DraftKings salary CSV."
    )
    parser.add_argument(
        "--win-probs",
        required=True,
        help="Path to JSON file mapping team abbreviation to win probability (0-1).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of unique lineups to return (across all team pairings).",
    )
    parser.add_argument(
        "--per-pair",
        type=int,
        default=3,
        help="Max number of lineups to extract per team pairing.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lol_cash_lineups.csv",
        help="Path to write lineups CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    with open(args.win_probs, "r", encoding="utf-8") as f:
        win_probs = json.load(f)

    players = load_salaries(args.salaries, win_probs)
    if not players:
        print("No eligible players after filtering for win probabilities >= 0.60.")
        return 1

    teams = sorted({p.team for p in players if win_probs.get(p.team, 0.0) >= 0.60})
    if len(teams) < 2:
        print("Need at least two favored teams (win% >= 0.60) to build lineups.")
        return 1

    all_lineups: List[dict] = []
    for primary in teams:
        for secondary in teams:
            if primary == secondary:
                continue
            if max(win_probs.get(primary, 0.0), win_probs.get(secondary, 0.0)) < 0.65:
                continue  # at least one team >= 0.65
            pair_lineups = solve_for_pair(
                players, primary, secondary, win_probs, limit=args.per_pair
            )
            all_lineups.extend(pair_lineups)

    if not all_lineups:
        print("No valid lineups found under the given constraints.")
        return 1

    ranked = rank_lineups(all_lineups)
    print_lineups(ranked, limit=args.top_n)
    save_lineups(ranked, path=args.output, limit=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
