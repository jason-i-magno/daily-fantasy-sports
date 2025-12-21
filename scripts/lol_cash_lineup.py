#!/usr/bin/env python3
"""
DraftKings LoL Classic H2H lineup generator (cash mindset, no projections).

Inputs:
  - Salary CSV from DraftKings.
  - JSON file with team win probabilities, e.g. {"TES": 0.7, "JDG": 0.62}.

Team priority (hard):
  - Team A: highest win%.
  - Team B: second-highest win%.
  - Team C: third-highest win% (only used for 4–2–1 fallback).
  - TEAM slot must be Team A.
  - At least 3 players from Team A; prefer 4 when feasible. Stack shapes: 4–3 (preferred), fallback 4–2–1 (A+B+C singleton), fallback 3–3–1 (A+B+C singleton).

Hard constraints:
  - Salary cap: 50,000 (CPT costs 1.5x salary).
  - Roster: CPT, TOP, JNG, MID, ADC, SUP, TEAM.
  - CPT priority ADC > MID > JNG (JNG only if ADC/MID CPT infeasible with Team A); CPT must be from Team A or Team B; CPT cannot be TOP/SUP.
  - Enforce 4–3 when feasible; fallback 4–2–1 (A primary 4 incl. TEAM, B 2, C singleton) or 3–3–1 (A primary 3 incl. TEAM, B 3, C singleton).
  - No underdog primary stacks; no one-off SUP/TOP; JNG/MID/ADC one-offs only from Team A or Team B; only one Team C player max (in fallback shapes).

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

CPT_ROLE_PENALTY = {
    "ADC": 0,
    "MID": 5,
    "JNG": 10,
    "SUP": 40,  # emergency
    "TOP": 60,  # absolute last resort
}

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
    if "Roster Position" in df.columns:
        pos_col = "Roster Position"
    elif "Position" in df.columns:
        pos_col = "Position"
    else:
        raise ValueError("Missing position column (Roster Position or Position).")
    for required in ["Name", pos_col, "Salary", "TeamAbbrev"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    # Validate team identifiers match between salary CSV and win_probs.
    csv_teams = set(df["TeamAbbrev"].unique())
    prob_teams = set(win_probs.keys())
    missing_in_csv = prob_teams - csv_teams
    missing_in_probs = csv_teams - prob_teams
    if missing_in_csv:
        raise ValueError(
            f"Teams in win_probs missing from salaries CSV: {sorted(missing_in_csv)}"
        )
    if missing_in_probs:
        raise ValueError(
            f"Teams in salaries CSV missing from win_probs: {sorted(missing_in_probs)}"
        )

    df["Salary"] = coerce_numeric(df["Salary"])
    df["team_prob"] = df["TeamAbbrev"].map(win_probs)
    df = df[df["team_prob"] >= 0.60]

    players: List[Player] = []
    for _, row in df.iterrows():
        role = str(row[pos_col]).upper()
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


def top_three_teams(win_probs: Dict[str, float]) -> Tuple[str, str, str | None]:
    ranked = sorted(win_probs.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        raise ValueError("Need at least two teams with win probabilities.")
    team_c = ranked[2][0] if len(ranked) >= 3 else None
    return ranked[0][0], ranked[1][0], team_c


def build_problem(
    players: List[Player],
    team_a: str,
    team_b: str,
    win_probs: Dict[str, float],
    captain_roles_allowed: Sequence[str],
    shape: str,
    team_c: str | None = None,
    max_c: int = 0,
    allow_c_roles: set[str] | None = None,
    require_c_pair: str | None = None,
    allow_c_top_one_off: bool = False,
):
    roles = ["TOP", "JNG", "MID", "ADC", "SUP"]

    prob = pulp.LpProblem("lol_cash_lineup", pulp.LpMaximize)

    flex_vars: Dict[str, pulp.LpVariable] = {}
    cpt_vars: Dict[str, pulp.LpVariable] = {}
    team_vars: Dict[str, pulp.LpVariable] = {}

    allowed_teams = {team_a, team_b}
    if team_c and max_c > 0:
        allowed_teams.add(team_c)

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
            if p.role in captain_roles_allowed and p.team in {team_a, team_b}:
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

    # Team stack counts: enforce shape (includes TEAM).
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

    # Team counts per shape; ensure at least 3 from A unless special relaxation.
    if shape == "4B-2A-1C":
        prob += team_count(team_a) == 2, "team_a_count_special"
        prob += team_count(team_b) == 4, "team_b_count_special"
        if team_c:
            prob += team_count(team_c) == 1, "team_c_single_special"
    else:
        prob += team_count(team_a) >= 3, "team_a_min3"
        if shape == "4-3":
            prob += team_count(team_a) == 4, "team_a_count"
            prob += team_count(team_b) == 3, "team_b_count"
        elif shape == "4-2-1":
            prob += team_count(team_a) == 4, "team_a_count"
            prob += team_count(team_b) == 2, "team_b_count"
            if team_c:
                prob += team_count(team_c) == 1, "team_c_single"
        elif shape == "3-3-1":
            prob += team_count(team_a) == 3, "team_a_count"
            prob += team_count(team_b) == 3, "team_b_count"
            if team_c:
                prob += team_count(team_c) == 1, "team_c_single"

    if shape in {"4-2-1", "3-3-1"} and team_c:
        # Singleton cannot be SUP; cannot be TOP unless explicitly allowed for a one-off top policy; cannot be CPT (captain vars exclude team C).
        for pid, var in flex_vars.items():
            player = players[int(pid[1:])]
            if player.team != team_c:
                continue
            if player.role == "SUP":
                prob += var == 0, f"no_singleton_support_{pid}"
            if player.role == "TOP" and not allow_c_top_one_off:
                prob += var == 0, f"no_singleton_top_{pid}"

    # Team C limits and pairing rules.
    if team_c and max_c >= 0:
        prob += team_count(team_c) <= max_c, "team_c_max"
        if max_c == 0:
            for pid, var in flex_vars.items():
                if players[int(pid[1:])].team == team_c:
                    prob += var == 0
            for pid, var in team_vars.items():
                if players[int(pid[1:])].team == team_c:
                    prob += var == 0
        if allow_c_roles is not None:
            for pid, var in flex_vars.items():
                player = players[int(pid[1:])]
                if player.team == team_c and player.role not in allow_c_roles:
                    prob += var == 0, f"c_role_restrict_{pid}"
        if require_c_pair and max_c == 2:
            # enforce specific pair: e.g., "ADC+SUP" or "MID+JNG"
            if require_c_pair == "ADC+SUP":
                prob += (
                    pulp.lpSum(
                        v
                        for pid, v in flex_vars.items()
                        if players[int(pid[1:])].team == team_c
                        and players[int(pid[1:])].role == "ADC"
                    )
                    == 1
                )
                prob += (
                    pulp.lpSum(
                        v
                        for pid, v in flex_vars.items()
                        if players[int(pid[1:])].team == team_c
                        and players[int(pid[1:])].role == "SUP"
                    )
                    == 1
                )
            elif require_c_pair == "MID+JNG":
                prob += (
                    pulp.lpSum(
                        v
                        for pid, v in flex_vars.items()
                        if players[int(pid[1:])].team == team_c
                        and players[int(pid[1:])].role == "MID"
                    )
                    == 1
                )
                prob += (
                    pulp.lpSum(
                        v
                        for pid, v in flex_vars.items()
                        if players[int(pid[1:])].team == team_c
                        and players[int(pid[1:])].role == "JNG"
                    )
                    == 1
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
    cpt_penalty = pulp.lpSum(
        CPT_ROLE_PENALTY.get(players[int(pid[1:])].role, 0) * v
        for pid, v in cpt_vars.items()
    )

    cpt_team_penalty = pulp.lpSum(
        (5 if players[int(pid[1:])].team == team_b else 0) * v
        for pid, v in cpt_vars.items()
    )

    prob += (
        total_win * 1000
        + salary_used * 0.1
        - (SALARY_CAP - salary_used) * 0.05
        - cpt_penalty
        - cpt_team_penalty
    )

    # Forbid one-off TOP/SUP unless explicitly allowed
    for pid, var in flex_vars.items():
        p = players[int(pid[1:])]
        if p.role in {"TOP", "SUP"}:
            same_team_count = pulp.lpSum(
                v
                for qid, v in flex_vars.items()
                if players[int(qid[1:])].team == p.team
            )
            if not (team_c and p.team == team_c and allow_c_top_one_off):
                prob += same_team_count >= 2, f"no_one_off_{p.team}_{p.role}"

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
    team_c: str | None = None,
):
    lineups = []
    first_reject: str | None = None

    # Captain priority stages (stop at first feasible): ADC -> MID -> JNG -> SUP -> TOP
    captain_priority = [
        ["ADC"],
        ["ADC", "MID"],
        ["ADC", "MID", "JNG"],
        ["ADC", "MID", "JNG", "SUP"],
        ["ADC", "MID", "JNG", "SUP", "TOP"],
    ]

    seen_sigs = set()
    stack_shapes = ["4-3", "4-2-1", "3-3-1"]

    # Team C policy stages: 0C, 1C (MID/JNG/ADC only), 2C paired (ADC+SUP then MID+JNG).
    c_policy = [
        {"max_c": 0, "allow_roles": None, "pairs": [], "allow_top_one_off": False},
        {
            "max_c": 1,
            "allow_roles": {"MID", "JNG", "ADC"},
            "pairs": [],
            "allow_top_one_off": False,
        },
        {
            "max_c": 2,
            "allow_roles": None,
            "pairs": ["ADC+SUP", "MID+JNG"],
            "allow_top_one_off": False,
        },
        {"max_c": 1, "allow_roles": {"TOP"}, "pairs": [], "allow_top_one_off": True},
        {"max_c": 1, "allow_roles": {"SUP"}, "pairs": [], "allow_top_one_off": False},
    ]

    for policy in c_policy:
        max_c = policy["max_c"]
        allow_roles = policy["allow_roles"]
        pair_options = policy["pairs"] or [None]
        for pair_req in pair_options:
            for shape in stack_shapes:
                if shape in {"4-2-1", "3-3-1"} and not team_c:
                    continue
                for roles_allowed in captain_priority:
                    prob, flex_vars, cpt_vars, team_vars = build_problem(
                        players,
                        team_a,
                        team_b,
                        win_probs,
                        captain_roles_allowed=roles_allowed,
                        shape=shape,
                        team_c=team_c,
                        max_c=max_c,
                        allow_c_roles=allow_roles,
                        require_c_pair=pair_req,
                        allow_c_top_one_off=policy.get("allow_top_one_off", False),
                    )

                    found_this_level = False
                    while len(lineups) < limit:
                        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
                        if pulp.LpStatus[status] != "Optimal":
                            break
                        rows = extract_lineup(flex_vars, cpt_vars, team_vars, players)
                        is_valid, reason = validate_lineup(
                            rows,
                            team_a,
                            team_b,
                            team_c,
                            shape,
                            max_c,
                            allow_roles,
                            pair_req,
                            policy.get("allow_top_one_off", False),
                        )
                        if not is_valid:
                            if first_reject is None:
                                first_reject = reason
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
                            print(
                                f"Rejected lineup (shape={shape}, policy_c={max_c}, pair={pair_req}, roles={roles_allowed}): {reason}",
                                file=sys.stderr,
                            )
                            continue
                        sig = lineup_signature(rows)
                        if sig in seen_sigs:
                            break
                        salary_used = sum(r["salary"] for r in rows)
                        total_win_prob = sum(
                            win_probs.get(r["team"], 0.0) for r in rows
                        )
                        team_a_count = sum(1 for r in rows if r["team"] == team_a)
                        team_c_count = sum(1 for r in rows if r["team"] == team_c)
                        # Enforce TEAM as part of largest stack is inherent with count constraints; ensure captain not team C.
                        if any(r["is_captain"] and r["team"] == team_c for r in rows):
                            break
                        lineups.append(
                            {
                                "rows": rows,
                                "signature": sig,
                                "salary_used": salary_used,
                                "total_win_prob": total_win_prob,
                                "objective": pulp.value(prob.objective),
                                "team_a_count": team_a_count,
                                "team_c_count": team_c_count,
                                "shape": shape,
                                "c_policy": max_c,
                            }
                        )
                        seen_sigs.add(sig)
                        found_this_level = True

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
                        break  # respect captain priority within shape
                if lineups:
                    break  # found lineups for this shape under current policy
            if lineups:
                break  # found lineups for this policy+pair
        if lineups:
            break  # stop at earliest feasible Team C policy

    # Special relaxation: 4B-2A-1C (Team B primary) only if nothing else found.
    if not lineups and team_c:
        shape_special = "4B-2A-1C"
        policy = {
            "max_c": 1,
            "allow_roles": None,
            "pairs": [None],
            "allow_top_one_off": False,
        }
        max_c = policy["max_c"]
        allow_roles = policy["allow_roles"]
        pair_req = None
        roles_allowed_list = captain_priority
        for roles_allowed in roles_allowed_list:
            prob, flex_vars, cpt_vars, team_vars = build_problem(
                players,
                team_a,
                team_b,
                win_probs,
                captain_roles_allowed=roles_allowed,
                shape=shape_special,
                team_c=team_c,
                max_c=max_c,
                allow_c_roles=allow_roles,
                require_c_pair=pair_req,
                allow_c_top_one_off=False,
            )
            while len(lineups) < limit:
                status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
                if pulp.LpStatus[status] != "Optimal":
                    break
                rows = extract_lineup(flex_vars, cpt_vars, team_vars, players)
                is_valid, reason = validate_lineup(
                    rows,
                    team_a,
                    team_b,
                    team_c,
                    shape_special,
                    max_c,
                    allow_roles,
                    pair_req,
                    False,
                )
                if not is_valid:
                    if first_reject is None:
                        first_reject = reason
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
                    print(
                        f"Rejected lineup (shape={shape_special}, policy_c={max_c}, pair={pair_req}, roles={roles_allowed}): {reason}",
                        file=sys.stderr,
                    )
                    continue
                sig = lineup_signature(rows)
                if sig in seen_sigs:
                    break
                salary_used = sum(r["salary"] for r in rows)
                total_win_prob = sum(win_probs.get(r["team"], 0.0) for r in rows)
                team_a_count = sum(1 for r in rows if r["team"] == team_a)
                team_c_count = sum(1 for r in rows if r["team"] == team_c)
                lineups.append(
                    {
                        "rows": rows,
                        "signature": sig,
                        "salary_used": salary_used,
                        "total_win_prob": total_win_prob,
                        "objective": pulp.value(prob.objective),
                        "team_a_count": team_a_count,
                        "team_c_count": team_c_count,
                        "shape": shape_special,
                        "c_policy": max_c,
                    }
                )
                seen_sigs.add(sig)

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
            if lineups:
                break
    return lineups, first_reject


def rank_lineups(lineups: List[dict]) -> List[dict]:
    return sorted(
        lineups,
        key=lambda x: (
            x.get("team_a_count", 0),
            x["total_win_prob"],
            x["salary_used"],
            x["objective"],
        ),
        reverse=True,
    )


def print_diagnostics(
    players: List[Player], team_a: str, team_b: str, team_c: str | None
) -> None:
    roles = ["TOP", "JNG", "MID", "ADC", "SUP", "TEAM"]
    # Slot pools (excluding CPT multiplier)
    print("=== Candidate pool by slot ===")
    for role in roles:
        pool = [p.salary for p in players if p.role == role]
        if pool:
            print(
                f"{role}: count={len(pool)}, min_salary={min(pool)}, max_salary={max(pool)}"
            )
        else:
            print(f"{role}: count=0")
    # CPT pool (ADC/MID/JNG from Team A/B)
    cpt_pool = [
        p.salary
        for p in players
        if p.role in {"ADC", "MID", "JNG"} and p.team in {team_a, team_b}
    ]
    if cpt_pool:
        print(
            f"CPT candidates (ADC/MID/JNG from A/B): count={len(cpt_pool)}, min_salary={min(cpt_pool)}, max_salary={max(cpt_pool)}"
        )
    else:
        print("CPT candidates (ADC/MID/JNG from A/B): count=0")

    print("\n=== Candidates by team and role ===")
    for t in [team_a, team_b, team_c]:
        if not t:
            continue
        print(f"Team {t}:")
        for role in roles:
            cnt = sum(1 for p in players if p.team == t and p.role == role)
            print(f"  {role}: {cnt}")


def validate_lineup(
    rows: List[dict],
    team_a: str,
    team_b: str,
    team_c: str | None,
    shape: str,
    max_c: int,
    allow_roles: set[str] | None,
    pair_req: str | None,
    allow_top_one_off: bool,
) -> tuple[bool, str | None]:
    if len(rows) != 7:
        return False, f"expected 7 slots, got {len(rows)}"

    salary_used = sum(r["salary"] for r in rows)
    if salary_used > SALARY_CAP + 1e-6:
        return False, f"salary cap exceeded: {salary_used}"

    slots = {r["slot"] for r in rows}
    required_slots = {"CPT", "TOP", "JNG", "MID", "ADC", "SUP", "TEAM"}
    if slots != required_slots:
        return (
            False,
            f"slots mismatch: have {sorted(slots)}, need {sorted(required_slots)}",
        )

    team_slot = next(r for r in rows if r["slot"] == "TEAM")
    if team_slot["team"] != team_a:
        return False, f"TEAM slot not Team A (got {team_slot['team']})"

    cpt = next(r for r in rows if r["slot"] == "CPT")
    if cpt["role"] not in {"ADC", "MID", "JNG", "SUP", "TOP"}:
        return False, f"captain role invalid: {cpt['role']}"
    if cpt["team"] not in {team_a, team_b}:
        return False, f"captain team invalid: {cpt['team']}"
    if team_c and cpt["team"] == team_c:
        return False, "captain from Team C"

    team_counts: Dict[str, int] = {}
    for r in rows:
        team_counts[r["team"]] = team_counts.get(r["team"], 0) + 1

    c_count = team_counts.get(team_c, 0) if team_c else 0
    if c_count >= 3:
        return False, f"Team C count too high: {c_count}"
    if team_counts.get(team_a, 0) < 3 and shape != "4B-2A-1C":
        return False, f"Team A count below 3: {team_counts.get(team_a, 0)}"
    if shape == "4-3":
        if team_counts.get(team_a, 0) < 4:
            return (
                False,
                f"Team A count below 4 for shape 4-3: {team_counts.get(team_a, 0)}",
            )
        if team_counts.get(team_b, 0) != 3:
            return (
                False,
                f"Team B count not 3 for shape 4-3: {team_counts.get(team_b, 0)}",
            )
    elif shape == "4-2-1":
        if team_counts.get(team_a, 0) < 4:
            return (
                False,
                f"Team A count below 4 for shape 4-2-1: {team_counts.get(team_a, 0)}",
            )
        if team_c and team_counts.get(team_c, 0) != 1:
            return (
                False,
                f"Team C count invalid for 4-2-1: {team_counts.get(team_c, 0)}",
            )
    elif shape == "3-3-1":
        if team_counts.get(team_a, 0) != 3 or team_counts.get(team_b, 0) != 3:
            return (
                False,
                f"Team A/B counts invalid for shape 3-3-1: A={team_counts.get(team_a, 0)}, B={team_counts.get(team_b, 0)}",
            )
        if team_c and team_counts.get(team_c, 0) != 1:
            return (
                False,
                f"Team C count invalid for 3-3-1: {team_counts.get(team_c, 0)}",
            )
    elif shape == "4B-2A-1C":
        if team_counts.get(team_b, 0) != 4 or team_counts.get(team_a, 0) != 2:
            return (
                False,
                f"Counts invalid for 4B-2A-1C: A={team_counts.get(team_a, 0)}, B={team_counts.get(team_b, 0)}",
            )
        if team_c and team_counts.get(team_c, 0) != 1:
            return (
                False,
                f"Team C count invalid for 4B-2A-1C: {team_counts.get(team_c, 0)}",
            )

    # forbid 4A-3C
    if team_counts.get(team_a, 0) == 4 and c_count == 3:
        return False, "4A-3C forbidden"

    if team_c is not None:
        if c_count > max_c:
            return False, f"Team C exceeds max {max_c}: {c_count}"
        if max_c == 1 and allow_roles:
            c_roles = {r["role"] for r in rows if r["team"] == team_c}
            if not c_roles.issubset(allow_roles):
                return False, f"Team C role not allowed: {c_roles}"
        if max_c == 2 and pair_req:
            c_roles = sorted(r["role"] for r in rows if r["team"] == team_c)
            required = sorted(pair_req.split("+"))
            if c_roles != required:
                return False, f"Team C pair mismatch: got {c_roles}, need {required}"
        if max_c == 1 and allow_top_one_off and c_count == 1:
            c_roles = {r["role"] for r in rows if r["team"] == team_c}
            if "TOP" not in c_roles:
                return False, f"Team C TOP one-off expected, got {c_roles}"
        if max_c == 1 and allow_roles == {"SUP"} and c_count == 1:
            c_roles = {r["role"] for r in rows if r["team"] == team_c}
            if "SUP" not in c_roles:
                return False, f"Team C SUP one-off expected, got {c_roles}"

    team_slots: Dict[str, List[str]] = {}
    for r in rows:
        team_slots.setdefault(r["team"], []).append(r["role"])

    return True, None


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
        team_a, team_b, team_c = top_three_teams(win_probs)
    except ValueError as exc:
        print(exc)
        return 1

    players = load_salaries(args.salaries, win_probs)
    if not players:
        print("No eligible players after filtering for win probabilities >= 0.60.")
        return 1

    print_diagnostics(players, team_a, team_b, team_c)

    lineups, first_reject = solve_for_pair(
        players, team_a, team_b, win_probs, limit=args.per_pair, team_c=team_c
    )

    if not lineups:
        print("No valid lineups found under the given constraints.")
        if first_reject:
            print(f"First rejection reason observed: {first_reject}")
        return 1

    ranked = rank_lineups(lineups)
    print_lineups(ranked, limit=args.top_n)
    save_lineups(ranked, path=args.output, limit=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
