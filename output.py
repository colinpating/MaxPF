"""Display results and export to CSV."""

import csv
from typing import NamedTuple

from tabulate import tabulate

from simulator import TeamResult
from sleeper import Roster, SleeperUser, get_team_name


def _build_rows(
    results: list[TeamResult],
    users: dict[str, SleeperUser],
    rosters: list[Roster],
) -> list[dict]:
    roster_map = {r.roster_id: r for r in rosters}
    rows = []
    for rank, result in enumerate(results, 1):
        roster = roster_map.get(result.roster_id)
        user = users.get(result.owner_id) if roster else None
        team_name = get_team_name(roster, user) if roster else f"Roster {result.roster_id}"
        owner = user.display_name if user else "Unknown"
        rows.append({
            "rank": rank,
            "team_name": team_name,
            "owner": owner,
            "proj_maxpf": round(result.mean_maxpf, 1),
            "p10": round(result.p10_maxpf, 1),
            "p90": round(result.p90_maxpf, 1),
            "ci_width": round(result.p90_maxpf - result.p10_maxpf, 1),
            "roster_size": result.n_players,
            "unmatched": result.n_unmatched,
            "weekly_means": result.weekly_means,
        })
    return rows


def display_results(
    results: list[TeamResult],
    users: dict[str, SleeperUser],
    rosters: list[Roster],
    league_name: str = "",
    n_sims: int = 5000,
    n_weeks: int = 14,
    include_ir: bool = False,
) -> None:
    rows = _build_rows(results, users, rosters)

    header_parts = [f"Dynasty MaxPF Projections — {n_weeks}-Week Regular Season"]
    if league_name:
        header_parts.append(f"League: {league_name}")
    header_parts.append(f"Sims: {n_sims:,}")
    header_parts.append("IR: " + ("Included" if include_ir else "Excluded"))
    print("\n" + " | ".join(header_parts))
    print("Sorted ascending — lowest MaxPF receives earliest draft pick\n")

    table_rows = [
        [
            r["rank"],
            r["team_name"],
            r["owner"],
            f"{r['proj_maxpf']:,.1f}",
            f"{r['p10']:,.1f}",
            f"{r['p90']:,.1f}",
        ]
        for r in rows
    ]
    headers = ["Rank", "Team Name", "Owner", "Proj MaxPF", "P10", "P90"]
    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))

    if any(r["unmatched"] > 0 for r in rows):
        print(f"\n* Players without a projection match contribute 0 pts.")
        for r in rows:
            if r["unmatched"] > 0:
                print(f"  {r['team_name']}: {r['unmatched']} unmatched player(s)")


def display_weekly_breakdown(
    results: list[TeamResult],
    users: dict[str, SleeperUser],
    rosters: list[Roster],
) -> None:
    rows = _build_rows(results, users, rosters)
    roster_map = {r.roster_id: r for r in rosters}
    n_weeks = len(results[0].weekly_means) if results else 0

    print("\nWeekly MaxPF Breakdown (avg optimal score per week):\n")
    week_headers = ["Team"] + [f"Wk {w}" for w in range(1, n_weeks + 1)] + ["Total"]
    table_rows = []
    for r in rows:
        weekly = [f"{v:.1f}" for v in r["weekly_means"]]
        total = f"{sum(r['weekly_means']):.1f}"
        table_rows.append([r["team_name"]] + weekly + [total])

    print(tabulate(table_rows, headers=week_headers, tablefmt="rounded_outline"))


def export_csv(
    results: list[TeamResult],
    users: dict[str, SleeperUser],
    rosters: list[Roster],
    filepath: str,
    n_weeks: int = 14,
) -> None:
    rows = _build_rows(results, users, rosters)
    fieldnames = ["rank", "team_name", "owner", "proj_maxpf", "p10", "p90", "ci_width", "roster_size", "unmatched"]
    for w in range(1, n_weeks + 1):
        fieldnames.append(f"week_{w}_mean")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row = {k: v for k, v in r.items() if k != "weekly_means"}
            for w_idx, wval in enumerate(r["weekly_means"], 1):
                row[f"week_{w_idx}_mean"] = round(wval, 2)
            writer.writerow(row)

    print(f"\nResults exported to {filepath}")
