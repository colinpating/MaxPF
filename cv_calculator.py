"""
Empirical per-position CV tiers derived from Sleeper historical weekly scores.

Instead of flat position CVs, groups players into projection tiers and uses
observed historical variance: top-projected players are more consistent (lower CV),
fringe players more volatile (higher CV).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

SLEEPER_BASE = "https://api.sleeper.app/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CV_CACHE = os.path.join(CACHE_DIR, "empirical_cvs.json")
CV_CACHE_TTL_DAYS = 30  # season stats don't change

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "K"}
MIN_ACTIVE_WEEKS = 4      # need at least 4 active weeks for a meaningful CV
MIN_MEAN_PPG = 1.0        # exclude scrubs / special teamers
ACTIVE_THRESHOLD = 0.5    # pts below this treated as "didn't play" (bye/inactive)
N_TIERS = 4               # tiers per position, sorted low→high by avg ppg

# Fallback if empirical data unavailable (original hardcoded values)
FALLBACK_CV = {
    "QB":  0.35,
    "RB":  0.55,
    "WR":  0.55,
    "TE":  0.60,
    "K":   0.50,
    "DEF": 0.65,
}
DEFAULT_CV = 0.55


# ── Fetching ───────────────────────────────────────────────────────────────────

def _fetch_week(season: int, week: int, scoring_format: str) -> tuple[int, dict[str, float]]:
    try:
        resp = requests.get(
            f"{SLEEPER_BASE}/stats/nfl/regular/{season}/{week}", timeout=30
        )
        if resp.status_code != 200:
            return week, {}
        data = resp.json() or {}
        result = {}
        for pid, stats in data.items():
            if not isinstance(stats, dict):
                continue
            pts = float(stats.get(scoring_format) or stats.get("pts_ppr") or 0)
            if pts >= ACTIVE_THRESHOLD:
                result[pid] = pts
        return week, result
    except Exception:
        return week, {}


def fetch_season_weekly_scores(
    season: int = 2024,
    scoring_format: str = "pts_ppr",
    n_weeks: int = 17,
) -> dict[str, list[float]]:
    """
    Parallel-fetch all weekly scores for a season.
    Returns {player_id: [score_week_A, score_week_B, ...]} — only active weeks included.
    """
    player_scores: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_week, season, w, scoring_format): w
            for w in range(1, n_weeks + 1)
        }
        for future in as_completed(futures):
            _, week_data = future.result()
            for pid, pts in week_data.items():
                player_scores.setdefault(pid, []).append(pts)
    return player_scores


# ── CV computation ─────────────────────────────────────────────────────────────

def compute_empirical_cvs(
    season: int = 2024,
    scoring_format: str = "pts_ppr",
    force: bool = False,
) -> dict[str, list[list]]:
    """
    Build empirical CV tiers from historical weekly fantasy scores.

    Returns {position: [[max_ppg_threshold, cv], ...]} sorted ascending by ppg.
    The last tier's threshold is 999 (catches all high-end projections).

    Example:
        {"RB": [[5.2, 0.74], [10.1, 0.61], [15.3, 0.52], [999.0, 0.44]]}
    Lookup: for an RB projected at 12 ppg → tier 3 (≤15.3) → CV = 0.52
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force and os.path.exists(CV_CACHE):
        if (time.time() - os.path.getmtime(CV_CACHE)) / 86400 < CV_CACHE_TTL_DAYS:
            with open(CV_CACHE, encoding="utf-8") as f:
                return json.load(f)

    players_cache = os.path.join(CACHE_DIR, "players_cache.json")
    if not os.path.exists(players_cache):
        print("Warning: players_cache.json not found — using fallback CVs")
        return {}

    with open(players_cache, encoding="utf-8") as f:
        players_db = json.load(f)

    print(f"Computing empirical CVs from {season} Sleeper weekly stats ({scoring_format})…")
    weekly_scores = fetch_season_weekly_scores(season, scoring_format)
    print(f"  Fetched data for {len(weekly_scores)} players across 17 weeks")

    from collections import defaultdict
    pos_data: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for pid, scores in weekly_scores.items():
        if len(scores) < MIN_ACTIVE_WEEKS:
            continue
        pos = players_db.get(pid, {}).get("position", "")
        if pos not in SKILL_POSITIONS:
            continue
        mean_ppg = float(np.mean(scores))
        if mean_ppg < MIN_MEAN_PPG:
            continue
        cv = float(np.std(scores) / mean_ppg)
        pos_data[pos].append((mean_ppg, cv))

    result: dict[str, list[list]] = {}
    for pos, data in pos_data.items():
        if len(data) < N_TIERS:
            continue
        data.sort(key=lambda x: x[0])  # ascending by mean_ppg
        tier_size = max(1, len(data) // N_TIERS)
        tiers = []
        for i in range(N_TIERS):
            start = i * tier_size
            end = start + tier_size if i < N_TIERS - 1 else len(data)
            tier = data[start:end]
            max_ppg = tier[-1][0]
            mean_cv = float(np.mean([cv for _, cv in tier]))
            tiers.append([round(max_ppg, 2), round(mean_cv, 4)])
        tiers[-1][0] = 999.0
        result[pos] = tiers

    with open(CV_CACHE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("  Empirical CVs:")
    for pos, tiers in sorted(result.items()):
        row = " | ".join(f"≤{t[0]:.0f}ppg → {t[1]:.2f}" for t in tiers)
        print(f"    {pos}: {row}")

    return result


# ── Lookup ─────────────────────────────────────────────────────────────────────

def get_cv(position: str, pts_per_game: float, empirical_cvs: dict | None) -> float:
    """Return CV for a player given position and projected pts/game."""
    if not empirical_cvs or position not in empirical_cvs:
        return FALLBACK_CV.get(position, DEFAULT_CV)
    for max_ppg, cv in empirical_cvs[position]:
        if pts_per_game <= max_ppg:
            return cv
    return empirical_cvs[position][-1][1]  # above all thresholds → top tier


def cv_summary_table(empirical_cvs: dict) -> list[dict]:
    """Flatten CV tiers into a list of dicts for display."""
    rows = []
    prev = {pos: 0.0 for pos in empirical_cvs}
    for pos in sorted(empirical_cvs):
        tiers = empirical_cvs[pos]
        for i, (max_ppg, cv) in enumerate(tiers):
            lo = prev[pos]
            hi = f"{max_ppg:.0f}" if max_ppg < 999 else "∞"
            rows.append({
                "Position": pos,
                "PPG Range": f"{lo:.0f}–{hi}",
                "CV (std/mean)": round(cv, 3),
                "Std at 10ppg": round(10 * cv, 1),
                "Std at 20ppg": round(20 * cv, 1),
            })
            prev[pos] = max_ppg
    return rows
