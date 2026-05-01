"""Sleeper fantasy football API wrapper."""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.sleeper.app/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PLAYERS_CACHE = os.path.join(CACHE_DIR, "players_cache.json")
PLAYERS_CACHE_TTL_DAYS = 7


@dataclass
class League:
    league_id: str
    name: str
    season: str
    roster_positions: list
    scoring_settings: dict
    total_rosters: int
    status: str


@dataclass
class Roster:
    roster_id: int
    owner_id: str
    players: list = field(default_factory=list)
    starters: list = field(default_factory=list)
    reserve: list = field(default_factory=list)
    taxi: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SleeperUser:
    user_id: str
    display_name: str
    username: str
    metadata: dict = field(default_factory=dict)


def _get(url: str, timeout: int = 30) -> dict | list:
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch {url}: {e}") from e
            time.sleep(2 ** attempt)


def get_user(username: str) -> dict:
    return _get(f"{BASE_URL}/user/{username}")


def get_leagues(user_id: str, season: int) -> list:
    return _get(f"{BASE_URL}/user/{user_id}/leagues/nfl/{season}") or []


def get_league(league_id: str) -> League:
    data = _get(f"{BASE_URL}/league/{league_id}")
    settings = data.get("scoring_settings") or {}
    positions = data.get("roster_positions") or []
    return League(
        league_id=data["league_id"],
        name=data.get("name", "Unknown League"),
        season=data.get("season", ""),
        roster_positions=positions,
        scoring_settings=settings,
        total_rosters=data.get("total_rosters", 0),
        status=data.get("status", ""),
    )


def get_rosters(league_id: str) -> list[Roster]:
    data = _get(f"{BASE_URL}/league/{league_id}/rosters") or []
    rosters = []
    for r in data:
        rosters.append(Roster(
            roster_id=r.get("roster_id", 0),
            owner_id=r.get("owner_id") or "",
            players=r.get("players") or [],
            starters=r.get("starters") or [],
            reserve=r.get("reserve") or [],
            taxi=r.get("taxi") or [],
            metadata=r.get("metadata") or {},
        ))
    return rosters


def get_users(league_id: str) -> dict[str, SleeperUser]:
    data = _get(f"{BASE_URL}/league/{league_id}/users") or []
    result = {}
    for u in data:
        user_id = u.get("user_id", "")
        meta = u.get("metadata") or {}
        result[user_id] = SleeperUser(
            user_id=user_id,
            display_name=u.get("display_name", "Unknown"),
            username=u.get("username", ""),
            metadata=meta,
        )
    return result


def get_players() -> dict[str, dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(PLAYERS_CACHE):
        age_days = (time.time() - os.path.getmtime(PLAYERS_CACHE)) / 86400
        if age_days < PLAYERS_CACHE_TTL_DAYS:
            with open(PLAYERS_CACHE, encoding="utf-8") as f:
                return json.load(f)

    print("Fetching NFL players database from Sleeper (one-time ~5MB download)...")
    data = _get(f"{BASE_URL}/players/nfl")
    with open(PLAYERS_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def get_nfl_state() -> dict:
    return _get(f"{BASE_URL}/state/nfl")


def get_nfl_leagues(username: str, season: int) -> list[dict]:
    """Return all NFL leagues for a username in a given season."""
    user = get_user(username)
    user_id = user["user_id"]
    leagues = get_leagues(user_id, season)
    return [lg for lg in leagues if lg.get("sport") == "nfl"]


def resolve_league_id(league_id: str | None, username: str | None, season: int) -> str:
    """CLI-only: resolves league interactively via terminal input."""
    if league_id:
        return league_id

    nfl_leagues = get_nfl_leagues(username, season)
    if not nfl_leagues:
        raise RuntimeError(f"No leagues found for {username} in {season}")
    if len(nfl_leagues) == 1:
        return nfl_leagues[0]["league_id"]

    _safe_print(f"\nFound {len(nfl_leagues)} leagues for {username}:")
    for i, lg in enumerate(nfl_leagues):
        _safe_print(f"  [{i + 1}] {lg.get('name', 'Unnamed')} (ID: {lg['league_id']})")
    while True:
        try:
            choice = int(input("\nSelect league number: ")) - 1
            if 0 <= choice < len(nfl_leagues):
                return nfl_leagues[choice]["league_id"]
        except (ValueError, KeyboardInterrupt):
            pass
        _safe_print("Invalid selection, try again.")


def _safe_print(text: str) -> None:
    """Print with emoji/unicode chars replaced when terminal doesn't support them."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def get_team_name(roster: Roster, user: SleeperUser | None) -> str:
    name = roster.metadata.get("team_name", "").strip()
    if name:
        return name
    if user:
        name = user.metadata.get("team_name", "").strip()
        if name:
            return name
        return user.display_name
    return f"Team {roster.roster_id}"
