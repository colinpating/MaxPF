"""Monte Carlo MaxPF simulation engine."""

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from lineup import (
    LineupSlot,
    build_lineup_slots,
    build_slot_labels,
    fast_lineup_slot_scores,
    greedy_lineup_score,
    precompute_eligibility_mask,
)
from projections import PlayerProjection, match_sleeper_player
from sleeper import Roster
from cv_calculator import get_cv, FALLBACK_CV

DEFAULT_CV = 0.55


@dataclass
class TeamSimData:
    roster_id: int
    owner_id: str
    player_ids: list[str]
    player_positions: list[str]
    player_means: np.ndarray    # per-game projected pts
    player_stds: np.ndarray     # per-game std dev
    eligibility_mask: np.ndarray
    lineup_slots: list[LineupSlot]


@dataclass
class TeamResult:
    roster_id: int
    owner_id: str
    mean_maxpf: float
    p10_maxpf: float
    p90_maxpf: float
    weekly_means: list[float]
    n_players: int
    n_unmatched: int
    slot_labels: list[str] = field(default_factory=list)
    slot_avg_pts_per_week: list[float] = field(default_factory=list)


def build_team_sim_data(
    roster: Roster,
    players_db: dict[str, dict],
    projections: dict[str, PlayerProjection],
    lineup_slots: list[LineupSlot],
    include_ir: bool = False,
    sleeper_proj: dict[str, float] | None = None,
    empirical_cvs: dict | None = None,
    sleeper_ppr: dict[str, float] | None = None,
) -> tuple[TeamSimData, list[str], list[dict]]:
    """
    Returns (TeamSimData, list_of_unmatched_player_names).
    Active pool = players + taxi; reserve (IR) excluded unless include_ir=True.
    """
    active_ids = list(roster.players) + list(roster.taxi)
    if include_ir:
        active_ids += list(roster.reserve)
    active_ids = list(dict.fromkeys(active_ids))  # deduplicate, preserve order

    player_ids = []
    player_positions = []
    player_means = []
    player_stds = []
    unmatched = []
    discrepancies = []

    for pid in active_ids:
        sleeper_player = players_db.get(pid)
        if sleeper_player is None:
            # DEF units are keyed by team abbrev in Sleeper, not in players_db
            if len(pid) <= 4 and pid.isalpha():
                sleeper_player = {"position": "DEF", "team": pid, "full_name": f"DEF {pid}"}
            else:
                continue

        position = sleeper_player.get("position") or sleeper_player.get("fantasy_positions", [""])[0]
        if not position:
            continue

        proj = match_sleeper_player(pid, sleeper_player, projections, sleeper_proj, sleeper_ppr)
        if proj is None or proj.pts_per_game <= 0:
            pts_pg = 0.0
            std_pg = 0.0
            if proj is None:
                name = sleeper_player.get("full_name") or pid
                unmatched.append(f"{name} ({position})")
        else:
            pts_pg = proj.pts_per_game
            std_pg = pts_pg * get_cv(position, pts_pg, empirical_cvs)
            if proj.source == "average" and proj.clay_pts_per_game > 0:
                abs_diff = abs(proj.clay_pts_per_game - proj.sleeper_pts_per_game)
                avg_ppg = proj.pts_per_game
                # Only flag players who matter: meaningful projection AND sizeable abs gap
                if avg_ppg >= 4.0 and abs_diff >= 2.0:
                    diff_pct = (proj.sleeper_pts_per_game - proj.clay_pts_per_game) / proj.clay_pts_per_game
                    discrepancies.append({
                        "Player": proj.player_name,
                        "Pos": proj.position,
                        "Team": proj.team,
                        "Clay": round(proj.clay_pts_per_game, 1),
                        "Sleeper": round(proj.sleeper_pts_per_game, 1),
                        "Avg Used": round(avg_ppg, 1),
                        "Diff (pts)": round(proj.sleeper_pts_per_game - proj.clay_pts_per_game, 1),
                        "Diff (%)": f"{diff_pct:+.0%}",
                        "_abs_diff": abs_diff,
                    })

        player_ids.append(pid)
        player_positions.append(position)
        player_means.append(pts_pg)
        player_stds.append(std_pg)

    means = np.array(player_means, dtype=float)
    stds = np.array(player_stds, dtype=float)
    mask = precompute_eligibility_mask(player_positions, lineup_slots)

    return TeamSimData(
        roster_id=roster.roster_id,
        owner_id=roster.owner_id,
        player_ids=player_ids,
        player_positions=player_positions,
        player_means=means,
        player_stds=stds,
        eligibility_mask=mask,
        lineup_slots=lineup_slots,
    ), unmatched, discrepancies


def simulate_team(
    team: TeamSimData,
    n_sims: int = 5000,
    n_weeks: int = 14,
    use_greedy: bool = False,
    rng: np.random.Generator | None = None,
) -> TeamResult:
    if rng is None:
        rng = np.random.default_rng()

    n_players = len(team.player_ids)
    n_slots = len(team.lineup_slots)
    slot_labels = build_slot_labels(team.lineup_slots)

    if n_players == 0:
        return TeamResult(
            roster_id=team.roster_id,
            owner_id=team.owner_id,
            mean_maxpf=0.0, p10_maxpf=0.0, p90_maxpf=0.0,
            weekly_means=[0.0] * n_weeks,
            n_players=0, n_unmatched=0,
            slot_labels=slot_labels,
            slot_avg_pts_per_week=[0.0] * n_slots,
        )

    # Sample all at once: shape (n_sims, n_weeks, n_players)
    all_scores = rng.normal(
        team.player_means,
        team.player_stds,
        size=(n_sims, n_weeks, n_players),
    )
    all_scores = np.maximum(all_scores, 0.0)

    # Flatten to (n_sims * n_weeks, n_players) for batch processing
    flat = all_scores.reshape(n_sims * n_weeks, n_players)

    results = np.zeros(n_sims * n_weeks, dtype=float)
    slot_results = np.zeros((n_sims * n_weeks, n_slots), dtype=float)

    if use_greedy:
        for i in range(len(flat)):
            results[i] = greedy_lineup_score(
                flat[i], team.player_positions, team.lineup_slots
            )
        # slot_results stays zero in fast mode (per-slot breakdown not tracked)
    else:
        for i in range(len(flat)):
            ss = fast_lineup_slot_scores(flat[i], team.eligibility_mask)
            results[i] = ss.sum()
            slot_results[i] = ss

    # Reshape back to (n_sims, n_weeks)
    weekly_matrix = results.reshape(n_sims, n_weeks)
    season_totals = weekly_matrix.sum(axis=1)
    weekly_means = weekly_matrix.mean(axis=0).tolist()
    slot_avg = slot_results.reshape(n_sims, n_weeks, n_slots).mean(axis=(0, 1)).tolist()

    return TeamResult(
        roster_id=team.roster_id,
        owner_id=team.owner_id,
        mean_maxpf=float(season_totals.mean()),
        p10_maxpf=float(np.percentile(season_totals, 10)),
        p90_maxpf=float(np.percentile(season_totals, 90)),
        weekly_means=weekly_means,
        n_players=n_players,
        n_unmatched=0,
        slot_labels=slot_labels,
        slot_avg_pts_per_week=slot_avg,
    )


def run_simulation(
    rosters: list[Roster],
    players_db: dict[str, dict],
    projections: dict[str, PlayerProjection],
    roster_positions: list[str],
    n_sims: int = 5000,
    n_weeks: int = 14,
    include_ir: bool = False,
    use_greedy: bool = False,
    seed: int | None = None,
    progress_callback=None,
    sleeper_proj: dict[str, float] | None = None,
    empirical_cvs: dict | None = None,
    sleeper_ppr: dict[str, float] | None = None,
) -> tuple[list[TeamResult], list[dict]]:
    lineup_slots = build_lineup_slots(roster_positions)
    print(f"\nLineup slots: {[s.slot_name for s in lineup_slots]}")

    rng = np.random.default_rng(seed)
    results = []
    all_unmatched = {}
    seen_discrepancies: dict[str, dict] = {}  # player_name → dict, deduplicated

    total = len(rosters)
    print(f"\nSimulating {total} teams x {n_sims} sims x {n_weeks} weeks...")
    iterable = tqdm(rosters, desc="Teams", unit="team") if progress_callback is None else rosters
    for i, roster in enumerate(iterable):
        team_data, unmatched, discrepancies = build_team_sim_data(
            roster, players_db, projections, lineup_slots, include_ir, sleeper_proj, empirical_cvs, sleeper_ppr
        )
        if unmatched:
            all_unmatched[roster.roster_id] = unmatched
        for d in discrepancies:
            seen_discrepancies[d["Player"]] = d

        result = simulate_team(team_data, n_sims, n_weeks, use_greedy, rng)
        result.n_unmatched = len(unmatched)
        results.append(result)
        if progress_callback is not None:
            progress_callback(i + 1, total)

    if all_unmatched:
        print(f"\nWarning: {sum(len(v) for v in all_unmatched.values())} players had no projection match.")
        for roster_id, names in all_unmatched.items():
            if names:
                sample = names[:5]
                print(f"  Roster {roster_id}: {', '.join(sample)}" + (f" (+{len(names)-5} more)" if len(names) > 5 else ""))

    results.sort(key=lambda r: r.mean_maxpf)
    disc_list = sorted(seen_discrepancies.values(), key=lambda d: d["_abs_diff"], reverse=True)
    for d in disc_list:
        d.pop("_abs_diff", None)
    return results, disc_list
