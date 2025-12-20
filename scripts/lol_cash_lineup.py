#!/usr/bin/env python3
"""
DraftKings LoL Classic H2H lineup generator (cash mindset, no projections).

Inputs:
  - Salary CSV from DraftKings.
  - JSON file with team win probabilities, e.g. {"TES": 0.7, "JDG": 0.62}.

Team priority (hard):
  - Team A: highest win%.
  - Team B: second-highest win%.
  - TEAM slot must be Team A.
  - >=3 players from Team A; prefer 4–3 stack with A(4)+B(3) when feasible; Team B is the only secondary stack.

Hard constraints:
  - Salary cap: 50,000 (CPT costs 1.5x salary).
  - Roster: CPT, TOP, JNG, MID, ADC, SUP, TEAM.
  - CPT priority ADC > MID > JNG (JNG only if ADC/MID CPT infeasible with Team A); CPT must be from Team A or Team B; CPT cannot be TOP/SUP.
  - Enforce 4–3 stack when feasible; Team A is the primary (4, incl. TEAM); Team B is the only secondary stack (3).
  - No underdog primary stacks; no one-off SUP/TOP; JNG/MID/ADC one-offs only from Team A or Team B.

Heuristic objective (cash safety):
  - Maximize combined team win probability weight across roster.
  - Secondary: maximize salary used (target high spend, >= 49k), small penalty for unused cap.

Implementation notes:
  - Uses PuLP ILP; prioritizes A(4)+B(3) stack and captain role order.
  - Produces top N unique valid lineups (deduped by player+cpt/team composition) sorted by objective.
  - Rejects invalid or duplicate lineups; no randomness, no upside simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    for required in ["Name", "Roster Position", "Salary", "TeamAbbrev"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    df["Salary"] = coerce_numeric(df["Salary"])
    df["team_prob"] = df["TeamAbbrev"].map(win_probs)
    df = df[df["team_prob"] >= 0.60]

    players: List[Player] = []
    for _, row in df.iterrows():
        role = str(row["Roster Position"]).upper()
        if role == "CPT":
            continue  # use base salary rows; CPT handled by 1.5x multiplier
        team = str(row["TeamAbbrev"])
        players.append(
            Player(
                name=str(row["Name"]),
                team=team,
                role=role,
                salary=float(row["Salary"]),
                is_team=role == "TEAM",
            )
        )
    return players


def top_two_teams(win_probs: Dict[str, float]) -> Tuple[str, str]:
    ranked = sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        raise ValueError("Need at least two teams with win probabilities.")
    return ranked[0][0], ranked[1][0]


def build_problem(
    players: List[Player],
    team_a: str,
    team_b: str,
    win_probs: Dict[str, float],
    captain_roles_allowed: Sequence[str],
):
    roles = ["TOP", "JNG", "MID", "ADC", "SUP"]

    prob = pulp.LpProblem("lol_cash_lineup", pulp.LpMaximize)

    flex_vars: Dict[str, pulp.LpVariable] = {}
    cpt_vars: Dict[str, pulp.LpVariable] = {}
    team_vars: Dict[str, pulp.LpVariable] = {}

    allowed_teams = {team_a, team_b}

    for idx, p in enumerate(players):
        if p.team not in allowed_teams:
            continue
        pid = f"p{idx}"
        if p.is_team:
            team_vars[pid] = pulp.LpVariable(
                pid + "_team", lowBound=0, upBound=1, cat="Binary"
            )
        else:
            flex_vars[pid] = pulp.LpVariable(
                pid + "_flex", lowBound=0, upBound=1, cat="Binary"
            )
            if p.role in captain_roles_allowed:
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

    # TEAM slot: exactly one, must be Team A.
    prob += pulp.lpSum(team_vars.values()) == 1, "one_team_slot"
    prob += (
        (
            pulp.lpSum(
                v
                for pid, v in team_vars.items()
                if players[int(pid[1:])].team == team_a
            )
            == 1
        ),
        "team_matches_A",
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

    # Team stack counts: enforce A(4) + B(3) (includes TEAM).
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

    prob += team_count(team_a) == 4, "team_a_count"
    prob += team_count(team_b) == 3, "team_b_count"

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
    parts = [
        f"{r['slot']}|{r['player_name']}|{r['team']}"
        for r in sorted(rows, key=lambda x: x["slot"])
    ]
    return ";".join(parts)


def solve_for_pair(
    players: List[Player],
    team_a: str,
    team_b: str,
    win_probs: Dict[str, float],
    limit: int,
):
    lineups = []
    captain_priority = [
        ["ADC"],
        ["ADC", "MID"],
        ["ADC", "MID", "JNG"],
    ]

    seen_sigs = set()
    for roles_allowed in captain_priority:
        prob, flex_vars, cpt_vars, team_vars = build_problem(
            players, team_a, team_b, win_probs, captain_roles_allowed=roles_allowed
        )

        found_this_level = False
        while len(lineups) < limit:
            status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[status] != "Optimal":
                break
            rows = extract_lineup(flex_vars, cpt_vars, team_vars, players)
            if len(rows) != 7:
                break
            sig = lineup_signature(rows)
            if sig in seen_sigs:
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
            seen_sigs.add(sig)
            found_this_level = True

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

        if found_this_level:
            break  # Respect captain priority; only fall through if infeasible.
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
        help="Number of unique lineups to return.",
    )
    parser.add_argument(
        "--per-pair",
        type=int,
        default=5,
        help="Max number of lineups to extract (respecting captain priority).",
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

    try:
        team_a, team_b = top_two_teams(win_probs)
    except ValueError as exc:
        print(exc)
        return 1

    players = load_salaries(args.salaries, win_probs)
    if not players:
        print("No eligible players after filtering for win probabilities >= 0.60.")
        return 1

    lineups = solve_for_pair(players, team_a, team_b, win_probs, limit=args.per_pair)

    if not lineups:
        print("No valid lineups found under the given constraints.")
        return 1

    ranked = rank_lineups(lineups)
    print_lineups(ranked, limit=args.top_n)
    save_lineups(ranked, path=args.output, limit=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
