"""
Parse Mike Clay ESPN projections PDF.

Reads the pre-calculated FF Pt (fantasy points) column directly from the
positional projection pages (pages 35–57). No stat re-calculation needed.

Page layout:
  35      → Quarterback Projections
  36–38   → Running Back Projections
  39–43   → Wide Receiver Projections
  44–45   → Tight End Projections
  57      → Kicker Projections
  (no team-DEF fantasy page; DEF uses a flat default)
"""

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, replace as dc_replace

import requests

SLEEPER_BASE = "https://api.sleeper.app/v1"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PDF_CACHE = os.path.join(CACHE_DIR, "clay_projections.pdf")
ESPN_PDF_URL = "https://g.espncdn.com/s/ffldraftkit/26/NFLDK2026_CS_ClayProjections2026.pdf"

# Page ranges for each position (0-based indices)
POSITION_PAGES = {
    "QB":  list(range(34, 35)),   # page 35
    "RB":  list(range(35, 38)),   # pages 36–38
    "WR":  list(range(38, 43)),   # pages 39–43
    "TE":  list(range(43, 45)),   # pages 44–45
    "K":   list(range(56, 57)),   # page 57
}

# Average DEF pts per game (no Clay DEF fantasy page exists)
DEF_DEFAULT_PTS_PER_GAME = 7.5

# Team abbreviations used by Clay → Sleeper equivalents
TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "LA":  "LAR",
    "WSH": "WAS",
    "GBP": "GB",
    "KCC": "KC",
    "NOR": "NO",
    "NEP": "NE",
    "SFO": "SF",
    "TBB": "TB",
    "SDC": "LAC",
}

# All valid Clay team abbreviations (used to locate team token in text lines)
KNOWN_TEAMS = {
    "ARZ", "ATL", "BLT", "BUF", "CHI", "CIN", "CLV", "DAL", "DEN", "DET",
    "GB",  "HST", "IND", "JAX", "KC",  "LAC", "LAR", "LV",  "MIA", "MIN",
    "NE",  "NO",  "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN",
    "WAS", "JAC",
}

# Tokens that look like team abbreviations but aren't
NOT_TEAMS = {
    "FF", "PT", "TD", "YD", "ATT", "RK", "TM", "G", "INT", "SK",
    "FGM", "FGA", "XPM", "XPA", "IDP", "QB", "RB", "WR", "TE",
    "LB", "CB", "DB", "DL", "SS", "FS", "DE", "DT", "NT",
    "TFL", "QBH", "PBU", "CAR", "REC",
}


@dataclass
class PlayerProjection:
    player_name: str
    team: str
    position: str
    games: float = 17.0
    pts_season: float = 0.0
    pts_per_game: float = 0.0
    clay_pts_per_game: float = 0.0    # 0 when only one source available
    sleeper_pts_per_game: float = 0.0
    source: str = "clay"              # "clay" | "sleeper" | "average"


# ── Utilities ──────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_team(team: str) -> str:
    if not team:
        return ""
    t = team.upper().strip()
    return TEAM_ALIASES.get(t, t)


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(str(val).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _find_team_idx(tokens: list[str]) -> int | None:
    """Return index of the first token that is a known team abbreviation."""
    for i, tok in enumerate(tokens):
        upper = tok.upper().rstrip(".")
        if upper in KNOWN_TEAMS and upper not in NOT_TEAMS and i > 0:
            return i
    return None


# ── Line parsers ───────────────────────────────────────────────────────────────

def _parse_skill_line(line: str, position: str) -> PlayerProjection | None:
    """
    Parse a skill-position data line.
    Format: [Name...] [TEAM] [PosRank] [FF_Pt] [G] ...
    Example: "Josh Allen BUF 1 360 17 498 331 ..."
    """
    tokens = line.split()
    if len(tokens) < 5:
        return None

    team_idx = _find_team_idx(tokens)
    if team_idx is None or team_idx == 0:
        return None

    name = " ".join(tokens[:team_idx]).strip()
    team = normalize_team(tokens[team_idx])

    # After team: PosRank, FF_Pt, G
    remaining = tokens[team_idx + 1:]
    if len(remaining) < 3:
        return None

    # PosRank should be a small integer (1–200)
    rank_str = remaining[0]
    if not re.match(r"^\d+$", rank_str):
        return None

    pts = _safe_float(remaining[1])
    games = _safe_float(remaining[2]) or 17.0

    if pts <= 0 or not name or len(name) < 2:
        return None

    proj = PlayerProjection(
        player_name=name, team=team, position=position,
        games=games, pts_season=pts,
    )
    proj.pts_per_game = pts / games
    return proj


def _parse_kicker_line(line: str) -> PlayerProjection | None:
    """
    Parse a kicker data line.
    Format: [Name...] [TEAM] [FF_Pt] [FGM] [FGA] ...
    Example: "Brandon Aubrey DAL 172 36 40 90% ..."
    """
    tokens = line.split()
    if len(tokens) < 4:
        return None

    team_idx = _find_team_idx(tokens)
    if team_idx is None or team_idx == 0:
        return None

    name = " ".join(tokens[:team_idx]).strip()
    team = normalize_team(tokens[team_idx])

    remaining = tokens[team_idx + 1:]
    if not remaining:
        return None

    pts = _safe_float(remaining[0])
    if pts <= 0 or not name or len(name) < 2:
        return None

    proj = PlayerProjection(
        player_name=name, team=team, position="K",
        games=17.0, pts_season=pts,
    )
    proj.pts_per_game = pts / 17.0
    return proj


# ── PDF parsing ────────────────────────────────────────────────────────────────

def _is_data_line(line: str, position: str) -> bool:
    """Return True if this line looks like a player data row (not a header)."""
    line = line.strip()
    if not line:
        return False
    # Skip obvious headers
    header_starts = (
        "quarterback", "running back", "wide receiver", "tight end",
        "kicker", "interior", "edge", "off-ball", "cornerback", "safety",
        "returner", "player", "name", "team", "rk ", "ff pt", "carry",
        "passing", "rushing", "receiving", "tackle", "sack",
    )
    low = line.lower()
    if any(low.startswith(h) for h in header_starts):
        return False
    # Must have at least one number in it (stats)
    if not re.search(r"\d", line):
        return False
    return True


def parse_pdf(pdf_path: str) -> list[PlayerProjection]:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    projections = []

    with pdfplumber.open(pdf_path) as pdf:
        for position, page_indices in POSITION_PAGES.items():
            for page_idx in page_indices:
                if page_idx >= len(pdf.pages):
                    continue
                page = pdf.pages[page_idx]
                text = page.extract_text() or ""

                for line in text.split("\n"):
                    if not _is_data_line(line, position):
                        continue
                    if position == "K":
                        proj = _parse_kicker_line(line)
                    else:
                        proj = _parse_skill_line(line, position)
                    if proj is not None:
                        projections.append(proj)

    return projections


def pdf_debug_info(pdf_path: str) -> dict:
    """Return diagnostic info for troubleshooting."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            info = {"n_pages": len(pdf.pages), "position_samples": {}}
            for position, page_indices in POSITION_PAGES.items():
                samples = []
                for page_idx in page_indices[:1]:
                    if page_idx < len(pdf.pages):
                        text = pdf.pages[page_idx].extract_text() or ""
                        lines = [l for l in text.split("\n") if _is_data_line(l, position)]
                        samples = lines[:5]
                info["position_samples"][position] = samples
            return info
    except Exception as e:
        return {"error": str(e)}


# ── Download ───────────────────────────────────────────────────────────────────

def download_pdf(url: str = ESPN_PDF_URL) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(PDF_CACHE):
        print(f"Using cached PDF: {PDF_CACHE}")
        return PDF_CACHE
    print("Downloading Clay projections PDF from ESPN...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, timeout=60, stream=True, headers=headers)
    resp.raise_for_status()
    with open(PDF_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {PDF_CACHE}")
    return PDF_CACHE


# ── Sleeper projections ────────────────────────────────────────────────────────

def get_scoring_format(scoring_settings: dict) -> str:
    """Map league scoring settings to Sleeper's pts field name."""
    rec = float(scoring_settings.get("rec", 0) or 0)
    if rec >= 1.0:
        return "pts_ppr"
    elif rec >= 0.5:
        return "pts_half_ppr"
    return "pts_std"


def fetch_sleeper_projections(season: int, scoring_format: str = "pts_ppr") -> dict[str, float]:
    """
    Fetch Sleeper week-1 projections as a per-game proxy.
    Returns {player_id: pts_per_game}. Empty dict on failure.
    Results cached for 7 days.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"sleeper_proj_{season}_{scoring_format}.json")
    if os.path.exists(cache_path):
        if (time.time() - os.path.getmtime(cache_path)) / 86400 < 7:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

    # Try season-total endpoint first, then weekly as fallback
    urls = [
        (f"{SLEEPER_BASE}/projections/nfl/regular/{season}", True),   # season: divide by gp
        (f"{SLEEPER_BASE}/projections/nfl/regular/{season}/1", False), # week 1: already per-game
        (f"{SLEEPER_BASE}/projections/nfl/{season}/1", False),
    ]
    for url, is_season in urls:
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue
            result = {}
            for pid, stats in data.items():
                if not isinstance(stats, dict):
                    continue
                pts = float(stats.get(scoring_format) or stats.get("pts_ppr") or 0)
                if pts <= 0:
                    continue
                if is_season:
                    gp = float(stats.get("gp") or 17)
                    result[pid] = pts / gp
                else:
                    result[pid] = pts
            if result:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                print(f"Fetched {len(result)} Sleeper projections ({season}, {scoring_format})")
                return result
        except Exception:
            continue

    print(f"Warning: Sleeper projections unavailable for {season} — using Clay only")
    return {}


# ── Load + match ───────────────────────────────────────────────────────────────

def _make_key(name: str, team: str) -> str:
    return f"{normalize_name(name)}_{normalize_team(team).lower()}"


def load_projections(
    pdf_path: str | None,
    scoring_settings: dict = None,   # unused; pts come from PDF directly
    games_per_season: int = 17,
) -> dict[str, PlayerProjection]:
    if pdf_path is None:
        pdf_path = download_pdf()

    raw = parse_pdf(pdf_path)
    print(f"Parsed {len(raw)} player projections from PDF")

    result = {}
    for proj in raw:
        key = _make_key(proj.player_name, proj.team)
        result[key] = proj

    return result


def match_sleeper_player(
    player_id: str,
    sleeper_player: dict,
    projections: dict[str, PlayerProjection],
    sleeper_proj: dict[str, float] | None = None,
) -> PlayerProjection | None:
    position = sleeper_player.get("position", "")
    team = normalize_team(sleeper_player.get("team") or "")

    # DEF: synthetic flat projection (no Clay DEF page)
    if position == "DEF":
        return PlayerProjection(
            player_name=f"DEF {player_id}",
            team=player_id,
            position="DEF",
            games=17.0,
            pts_season=DEF_DEFAULT_PTS_PER_GAME * 17,
            pts_per_game=DEF_DEFAULT_PTS_PER_GAME,
        )

    full_name = sleeper_player.get("full_name") or (
        f"{sleeper_player.get('first_name', '')} {sleeper_player.get('last_name', '')}"
    ).strip()

    # Find Clay projection via name+team, name-only, or last+team
    clay_proj = None
    key = _make_key(full_name, team)
    if key in projections:
        clay_proj = projections[key]
    else:
        norm_name = normalize_name(full_name)
        for k, v in projections.items():
            if k.startswith(norm_name + "_"):
                clay_proj = v
                break
        if clay_proj is None:
            last_name = sleeper_player.get("last_name", "")
            if last_name and team:
                norm_last = normalize_name(last_name)
                norm_team_lower = normalize_team(team).lower()
                for k, v in projections.items():
                    if k.endswith(f"_{norm_team_lower}") and norm_last in k:
                        clay_proj = v
                        break

    sleeper_ppg = float((sleeper_proj or {}).get(player_id) or 0)

    if clay_proj is not None and sleeper_ppg > 0:
        avg_ppg = (clay_proj.pts_per_game + sleeper_ppg) / 2
        return dc_replace(
            clay_proj,
            pts_per_game=avg_ppg,
            pts_season=avg_ppg * clay_proj.games,
            clay_pts_per_game=clay_proj.pts_per_game,
            sleeper_pts_per_game=sleeper_ppg,
            source="average",
        )
    elif clay_proj is not None:
        return clay_proj
    elif sleeper_ppg > 0:
        return PlayerProjection(
            player_name=full_name,
            team=team,
            position=position,
            games=17.0,
            pts_season=sleeper_ppg * 17,
            pts_per_game=sleeper_ppg,
            sleeper_pts_per_game=sleeper_ppg,
            source="sleeper",
        )

    return None
