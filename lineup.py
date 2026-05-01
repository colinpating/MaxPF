"""Optimal fantasy lineup selection using the Hungarian algorithm."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Maps NFL position -> set of lineup slot names it can fill
POSITION_ELIGIBILITY: dict[str, frozenset] = {
    "QB":  frozenset({"QB", "SUPER_FLEX", "FLEX"}),
    "RB":  frozenset({"RB", "FLEX", "SUPER_FLEX"}),
    "WR":  frozenset({"WR", "FLEX", "SUPER_FLEX", "REC_FLEX"}),
    "TE":  frozenset({"TE", "FLEX", "SUPER_FLEX", "REC_FLEX"}),
    "K":   frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    # IDP
    "LB":  frozenset({"LB", "IDP_FLEX"}),
    "DB":  frozenset({"DB", "IDP_FLEX"}),
    "DL":  frozenset({"DL", "IDP_FLEX"}),
}

# Slot name canonicalization (Sleeper uses SUPER_FLEX, not SUPERFLEX)
SLOT_ALIASES = {
    "SUPERFLEX": "SUPER_FLEX",
    "SF": "SUPER_FLEX",
    "WR/RB": "FLEX",
    "WR/RB/TE": "FLEX",
    "RB/WR/TE": "FLEX",
    "WR/TE": "REC_FLEX",
    "OP": "SUPER_FLEX",
}

# Which NFL positions are eligible per slot
SLOT_ELIGIBILITY: dict[str, frozenset] = {
    "QB":        frozenset({"QB"}),
    "RB":        frozenset({"RB"}),
    "WR":        frozenset({"WR"}),
    "TE":        frozenset({"TE"}),
    "K":         frozenset({"K"}),
    "DEF":       frozenset({"DEF"}),
    "FLEX":      frozenset({"RB", "WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "REC_FLEX":  frozenset({"WR", "TE"}),
    "IDP_FLEX":  frozenset({"LB", "DB", "DL"}),
    "LB":        frozenset({"LB"}),
    "DB":        frozenset({"DB"}),
    "DL":        frozenset({"DL"}),
}

# Slots that count toward the active lineup (not bench/IR/taxi)
BENCH_SLOTS = frozenset({"BN", "BE", "IR", "TAXI"})

SENTINEL = 1e9  # large positive cost = impossible/ineligible assignment


@dataclass
class LineupSlot:
    slot_name: str
    eligible_positions: frozenset


def build_lineup_slots(roster_positions: list[str]) -> list[LineupSlot]:
    slots = []
    for pos in roster_positions:
        canonical = SLOT_ALIASES.get(pos.upper(), pos.upper())
        if canonical in BENCH_SLOTS:
            continue
        eligible = SLOT_ELIGIBILITY.get(canonical, frozenset({canonical}))
        slots.append(LineupSlot(slot_name=canonical, eligible_positions=eligible))
    return slots


def precompute_eligibility_mask(
    player_positions: list[str],
    slots: list[LineupSlot],
) -> np.ndarray:
    """Return bool array of shape (n_slots, n_players): True if player can fill slot."""
    n_slots = len(slots)
    n_players = len(player_positions)
    mask = np.zeros((n_slots, n_players), dtype=bool)
    for i, slot in enumerate(slots):
        for j, pos in enumerate(player_positions):
            if pos in slot.eligible_positions:
                mask[i, j] = True
    return mask


def fast_lineup_score(
    scores: np.ndarray,
    eligibility_mask: np.ndarray,
) -> float:
    """
    Given per-player scores (shape n_players) and precomputed eligibility mask
    (shape n_slots x n_players), return the maximum achievable lineup score.
    Uses Hungarian algorithm via scipy.optimize.linear_sum_assignment.
    """
    n_slots, n_players = eligibility_mask.shape
    # Cost matrix: negative score where eligible, sentinel elsewhere
    cost = np.where(eligibility_mask, -scores, SENTINEL)

    row_ind, col_ind = linear_sum_assignment(cost)

    total = 0.0
    for r, c in zip(row_ind, col_ind):
        val = cost[r, c]
        if val < SENTINEL / 2:  # real assignment (not an ineligible forced fill)
            total += -val  # negate back to get positive score
    return total


def greedy_lineup_score(
    scores: np.ndarray,
    player_positions: list[str],
    slots: list[LineupSlot],
) -> float:
    """
    Fast greedy approximation: assign best-scoring eligible player to each slot
    in descending score order. ~10x faster than Hungarian, ~5% lower accuracy.
    """
    order = np.argsort(scores)[::-1]
    used = set()
    slot_filled = [False] * len(slots)
    total = 0.0

    for idx in order:
        pos = player_positions[idx]
        score = scores[idx]
        if score <= 0:
            break
        for s_idx, slot in enumerate(slots):
            if not slot_filled[s_idx] and pos in slot.eligible_positions:
                slot_filled[s_idx] = True
                used.add(idx)
                total += score
                break
    return total


def fast_lineup_slot_scores(
    scores: np.ndarray,
    eligibility_mask: np.ndarray,
) -> np.ndarray:
    """Like fast_lineup_score but returns per-slot scores instead of the sum."""
    n_slots = eligibility_mask.shape[0]
    cost = np.where(eligibility_mask, -scores, SENTINEL)
    row_ind, col_ind = linear_sum_assignment(cost)
    slot_scores = np.zeros(n_slots, dtype=float)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < SENTINEL / 2:
            slot_scores[r] = scores[c]
    return slot_scores


def get_optimal_assignment(
    scores: np.ndarray,
    eligibility_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Returns (slot_idx, player_idx) pairs for each filled slot, sorted by slot."""
    cost = np.where(eligibility_mask, -scores, SENTINEL)
    row_ind, col_ind = linear_sum_assignment(cost)
    return sorted(
        (r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] < SENTINEL / 2
    )


def build_slot_labels(lineup_slots: list[LineupSlot]) -> list[str]:
    """Generate display labels: QB, RB1, RB2, WR1, WR2, FLEX, SFLEX, etc."""
    from collections import Counter
    counts = Counter(s.slot_name for s in lineup_slots)
    seen: Counter = Counter()
    _short = {"SUPER_FLEX": "SFLEX", "REC_FLEX": "RFLEX", "IDP_FLEX": "IDP"}
    labels = []
    for slot in lineup_slots:
        name = slot.slot_name
        seen[name] += 1
        short = _short.get(name, name)
        labels.append(short if counts[name] == 1 else f"{short}{seen[name]}")
    return labels


def optimal_lineup_score(
    player_scores: dict[str, float],
    player_positions: dict[str, str],
    lineup_slots: list[LineupSlot],
    use_greedy: bool = False,
) -> float:
    """
    Convenience wrapper used in unit tests and one-off calls.
    player_scores: {player_id: score}
    player_positions: {player_id: NFL_position}
    """
    ids = list(player_scores.keys())
    scores = np.array([player_scores[pid] for pid in ids], dtype=float)
    positions = [player_positions[pid] for pid in ids]

    if use_greedy:
        return greedy_lineup_score(scores, positions, lineup_slots)

    mask = precompute_eligibility_mask(positions, lineup_slots)
    return fast_lineup_score(scores, mask)
