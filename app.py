"""Streamlit browser UI for the MaxPF dynasty fantasy football projection tool."""

import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynasty MaxPF Projector",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏈 Dynasty MaxPF Projector")
st.caption("Projects each team's Maximum Possible Fantasy Points over the regular season. Lower MaxPF = earlier draft pick.")

# ── Sidebar: inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("League Settings")

    input_method = st.radio("Find league by", ["Username", "League ID"], horizontal=True)

    if input_method == "Username":
        username = st.text_input("Sleeper username", placeholder="e.g. john_doe")
        league_id_input = None
    else:
        username = None
        league_id_input = st.text_input("Sleeper league ID", placeholder="e.g. 1234567890")

    season = st.number_input("Season", min_value=2020, max_value=2030, value=2026, step=1)

    st.divider()
    st.header("Simulation Settings")

    n_sims = st.select_slider(
        "Simulations",
        options=[500, 1000, 2000, 5000, 10000],
        value=2000,
        help="More sims = more accurate but slower. 2,000 is a good balance.",
    )
    n_weeks = st.slider("Regular season weeks", min_value=10, max_value=17, value=14)
    include_ir = st.toggle("Include IR players", value=False,
                           help="Count IR/reserve players in MaxPF calculation")
    use_fast = st.toggle("Fast mode", value=False,
                         help="~10x faster, ~5% less accurate lineup math")

    st.divider()
    st.header("Projections")
    pdf_source = st.radio("Projection source", ["Auto-download (ESPN Clay)", "Upload PDF"], horizontal=False)
    uploaded_pdf = None
    if pdf_source == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload Clay projections PDF", type="pdf")

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    if st.button("Clear cache", use_container_width=True, help="Force re-download of PDF and player data"):
        st.cache_data.clear()
        st.success("Cache cleared — re-run the simulation.")


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching NFL player database...")
def cached_get_players():
    import sleeper as sl
    return sl.get_players()

@st.cache_data(show_spinner="Loading projections from PDF...")
def cached_load_projections(pdf_bytes: bytes, scoring_settings_json: str):
    import json, projections as proj
    scoring_settings = dict(json.loads(scoring_settings_json))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return proj.load_projections(pdf_path=tmp_path, scoring_settings=scoring_settings)
    finally:
        os.unlink(tmp_path)

@st.cache_data(show_spinner="Fetching Sleeper projections...")
def cached_fetch_sleeper_proj(season: int, scoring_settings_json: str) -> tuple[dict, dict]:
    import json
    from projections import fetch_sleeper_projections
    scoring_settings = json.loads(scoring_settings_json)
    return fetch_sleeper_projections(season, scoring_settings)

@st.cache_data(show_spinner="Computing empirical CVs from historical data...")
def cached_compute_cvs(hist_season: int, scoring_format: str) -> dict:
    from cv_calculator import compute_empirical_cvs
    return compute_empirical_cvs(season=hist_season, scoring_format=scoring_format)

def get_pdf_bytes() -> bytes | None:
    if uploaded_pdf is not None:
        return uploaded_pdf.read()
    # Auto-download
    import requests
    from projections import ESPN_PDF_URL, PDF_CACHE, CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(PDF_CACHE):
        with open(PDF_CACHE, "rb") as f:
            return f.read()
    try:
        with st.spinner("Downloading Clay projections PDF from ESPN..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(ESPN_PDF_URL, timeout=60, stream=True, headers=headers)
            resp.raise_for_status()
            data = resp.content
            with open(PDF_CACHE, "wb") as f:
                f.write(data)
            return data
    except Exception as e:
        st.error(f"Could not download PDF: {e}\n\nPlease download it manually and use the 'Upload PDF' option.")
        return None


def results_to_dataframe(results, users, rosters):
    import sleeper as sl
    roster_map = {r.roster_id: r for r in rosters}
    rows = []
    for rank, result in enumerate(results, 1):
        roster = roster_map.get(result.roster_id)
        user = users.get(result.owner_id) if roster else None
        team_name = sl.get_team_name(roster, user) if roster else f"Roster {result.roster_id}"
        owner = user.display_name if user else "Unknown"
        rows.append({
            "Pick": rank,
            "Team": team_name,
            "Owner": owner,
            "Proj MaxPF": round(result.mean_maxpf, 1),
            "P10 (floor)": round(result.p10_maxpf, 1),
            "P90 (ceiling)": round(result.p90_maxpf, 1),
            "Roster Size": result.n_players,
            "Unmatched": result.n_unmatched,
            "_weekly_means": result.weekly_means,
            "_roster_id": result.roster_id,
        })
    return pd.DataFrame(rows)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    cols = [c for c in df.columns if not c.startswith("_")]
    return df[cols].to_csv(index=False).encode("utf-8")


# ── League selection (username flow) ──────────────────────────────────────────
# When using username, we may need the user to pick from multiple leagues.
# We do this as a separate step before the simulation button, stored in session state.

import sleeper as sl
import simulator as sim

if input_method == "Username" and (username or "").strip():
    if st.session_state.get("_username_loaded") != username:
        # Clear stale league selection when username changes
        st.session_state.pop("selected_league_id", None)
        st.session_state.pop("_username_loaded", None)

    if "selected_league_id" not in st.session_state:
        with st.spinner(f"Looking up leagues for {username}..."):
            try:
                resolved_season = season
                try:
                    state = sl.get_nfl_state()
                    resolved_season = int(state.get("season", 2026))
                except Exception:
                    pass
                nfl_leagues = sl.get_nfl_leagues(username, resolved_season)
                st.session_state["_nfl_leagues"] = nfl_leagues
                st.session_state["_username_loaded"] = username
                if len(nfl_leagues) == 1:
                    st.session_state["selected_league_id"] = nfl_leagues[0]["league_id"]
                elif len(nfl_leagues) == 0:
                    st.error(f"No NFL leagues found for **{username}** in {resolved_season}.")
            except Exception as e:
                st.error(f"Could not look up user: {e}")

    nfl_leagues = st.session_state.get("_nfl_leagues", [])
    if len(nfl_leagues) > 1 and "selected_league_id" not in st.session_state:
        options = {lg.get("name", f"League {lg['league_id']}"): lg["league_id"] for lg in nfl_leagues}
        chosen_name = st.selectbox("Select league", list(options.keys()), key="league_picker")
        if st.button("Confirm league", type="secondary"):
            st.session_state["selected_league_id"] = options[chosen_name]
            st.rerun()


# ── Main run logic ─────────────────────────────────────────────────────────────
if run_btn:
    # Validate inputs
    if input_method == "Username" and not (username or "").strip():
        st.error("Please enter your Sleeper username.")
        st.stop()
    if input_method == "League ID" and not (league_id_input or "").strip():
        st.error("Please enter your Sleeper league ID.")
        st.stop()

    try:
        # Resolve league ID
        if input_method == "League ID":
            resolved_league_id = league_id_input.strip()
        else:
            resolved_league_id = st.session_state.get("selected_league_id")
            if not resolved_league_id:
                st.error("Please wait for leagues to load, then select one above.")
                st.stop()

        with st.spinner("Fetching league info, rosters, and users..."):
            league = sl.get_league(resolved_league_id)
            rosters = sl.get_rosters(resolved_league_id)
            users = sl.get_users(resolved_league_id)

        st.success(f"Loaded **{league.name}** — {len(rosters)} teams")

        with st.spinner("Loading player database..."):
            players_db = cached_get_players()

        # PDF
        pdf_bytes = get_pdf_bytes()
        if pdf_bytes is None:
            st.stop()

        # Use a stable cache key based on scoring settings
        import json
        scoring_json = json.dumps(league.scoring_settings, sort_keys=True)
        with st.spinner("Parsing projections..."):
            projections = cached_load_projections(pdf_bytes, scoring_json)

        if len(projections) == 0:
            st.error(
                "No projections were parsed from the PDF. "
                "This usually means the PDF layout is different than expected. "
                "See the diagnostic info below."
            )
            import projections as proj_mod, tempfile, os as _os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            debug = proj_mod.pdf_debug_info(tmp_path)
            _os.unlink(tmp_path)
            with st.expander("PDF diagnostic info"):
                st.json(debug)
            st.stop()

        st.info(f"Loaded {len(projections):,} Clay player projections")
        top_proj = sorted(projections.values(), key=lambda p: p.pts_per_game, reverse=True)[:10]
        with st.expander("Top projected players (verify these look right)"):
            top_rows = [{"Player": p.player_name, "Team": p.team, "Pos": p.position,
                         "Pts/Game": round(p.pts_per_game, 1)} for p in top_proj]
            st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

        # Fetch Sleeper projections for consensus averaging
        import projections as proj_mod
        scoring_format = proj_mod.get_scoring_format(league.scoring_settings)
        with st.spinner("Fetching Sleeper projections..."):
            sleeper_proj, sleeper_ppr = cached_fetch_sleeper_proj(int(league.season or season), scoring_json)
        if sleeper_proj:
            st.info(f"Averaging Clay + Sleeper projections ({len(sleeper_proj):,} Sleeper players, adjusted for league scoring)")
        else:
            st.warning("Sleeper projections unavailable — using Clay only")

        # Empirical CVs from 2024 historical weekly scores
        empirical_cvs = cached_compute_cvs(2024, scoring_format)
        if empirical_cvs:
            st.info(f"Using empirical CVs from 2024 historical data ({', '.join(sorted(empirical_cvs))})")
        else:
            st.warning("Empirical CVs unavailable — using fallback flat CVs")

        # Simulation with progress bar
        st.write(f"**Running {n_sims:,} simulations × {n_weeks} weeks × {len(rosters)} teams...**")
        progress_bar = st.progress(0, text="Starting simulation...")

        def on_progress(completed: int, total: int):
            pct = completed / total
            progress_bar.progress(pct, text=f"Simulating team {completed} of {total}...")

        results, discrepancies = sim.run_simulation(
            rosters=rosters,
            players_db=players_db,
            projections=projections,
            roster_positions=league.roster_positions,
            n_sims=n_sims,
            n_weeks=n_weeks,
            include_ir=include_ir,
            use_greedy=use_fast,
            progress_callback=on_progress,
            sleeper_proj=sleeper_proj,
            empirical_cvs=empirical_cvs,
            sleeper_ppr=sleeper_ppr,
        )

        progress_bar.progress(1.0, text="Done!")
        st.session_state["results"] = results
        st.session_state["discrepancies"] = discrepancies
        st.session_state["empirical_cvs"] = empirical_cvs
        st.session_state["users"] = users
        st.session_state["rosters"] = rosters
        st.session_state["league"] = league
        st.session_state["n_sims"] = n_sims
        st.session_state["n_weeks"] = n_weeks

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        with st.expander("Details"):
            st.code(traceback.format_exc())


# ── Display results ────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results      = st.session_state["results"]
    users        = st.session_state["users"]
    rosters      = st.session_state["rosters"]
    league       = st.session_state["league"]
    n_sims       = st.session_state["n_sims"]
    n_weeks      = st.session_state["n_weeks"]
    discrepancies = st.session_state.get("discrepancies", [])

    df = results_to_dataframe(results, users, rosters)

    st.divider()
    st.subheader(f"Draft Pick Order — {league.name}")
    st.caption(f"{n_sims:,} simulations · {n_weeks}-week season · Pick 1 = lowest projected MaxPF")

    # Summary metric cards
    cols = st.columns(min(len(df), 4))
    for i, (_, row) in enumerate(df.head(4).iterrows()):
        with cols[i]:
            st.metric(
                label=f"Pick #{int(row['Pick'])} — {row['Team']}",
                value=f"{row['Proj MaxPF']:,.0f} pts",
                delta=f"Range: {row['P10 (floor)']:,.0f}–{row['P90 (ceiling)']:,.0f}",
                delta_color="off",
            )

    st.write("")

    # Main results table
    display_cols = ["Pick", "Team", "Owner", "Proj MaxPF", "P10 (floor)", "P90 (ceiling)"]
    styled = df[display_cols].style.background_gradient(
        subset=["Proj MaxPF"], cmap="RdYlGn_r"
    ).format({
        "Proj MaxPF": "{:,.1f}",
        "P10 (floor)": "{:,.1f}",
        "P90 (ceiling)": "{:,.1f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = df_to_csv_bytes(df[display_cols])
    st.download_button(
        label="⬇  Download CSV",
        data=csv_bytes,
        file_name=f"maxpf_{league.name.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Weekly breakdown
    with st.expander("Weekly breakdown (avg optimal score per week — bye weeks not modeled)"):
        weekly_rows = []
        for _, row in df.iterrows():
            wrow = {"Team": row["Team"]}
            for w, val in enumerate(row["_weekly_means"], 1):
                wrow[f"Wk {w}"] = round(val, 1)
            wrow["Season Total"] = round(sum(row["_weekly_means"]), 1)
            weekly_rows.append(wrow)
        weekly_df = pd.DataFrame(weekly_rows)
        st.dataframe(weekly_df, use_container_width=True, hide_index=True)

    # Per-slot MaxPF breakdown
    slot_labels = next((r.slot_labels for r in results if hasattr(r, "slot_labels") and r.slot_labels), None)
    if slot_labels:
        with st.expander("MaxPF breakdown by position slot"):
            rid_to_result = {r.roster_id: r for r in results}
            slot_rows = []
            for _, row in df.iterrows():
                r = rid_to_result.get(row["_roster_id"])
                if r is None:
                    continue
                srow = {"Team": row["Team"]}
                for label, val in zip(r.slot_labels, r.slot_avg_pts_per_week):
                    srow[label] = round(val, 1)
                srow["Total/Wk"] = round(sum(r.slot_avg_pts_per_week), 1)
                srow["Season Total"] = round(sum(r.slot_avg_pts_per_week) * n_weeks, 1)
                slot_rows.append(srow)
            slot_df = pd.DataFrame(slot_rows, columns=["Team"] + slot_labels + ["Total/Wk", "Season Total"])
            st.caption("Avg pts contributed per week by each lineup slot, averaged across all simulations. Season Total = Total/Wk × regular season weeks.")
            st.dataframe(slot_df, use_container_width=True, hide_index=True)

    # Empirical CV tiers
    empirical_cvs_display = st.session_state.get("empirical_cvs")
    if empirical_cvs_display:
        with st.expander("Empirical CV tiers (weekly variance by position & projection level)"):
            from cv_calculator import cv_summary_table
            cv_rows = cv_summary_table(empirical_cvs_display)
            st.caption("CV = std/mean of actual 2024 weekly scores. Lower-projected players are more volatile (higher CV).")
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    # Clay vs Sleeper discrepancies
    if discrepancies:
        with st.expander(f"⚠ {len(discrepancies)} meaningful Clay vs Sleeper disagreements (≥4 pts/game avg, ≥2 pts apart)"):
            st.caption("Sorted by absolute gap. Projections used are the average of both sources. Large gaps may reflect injuries, role changes, or differing methodologies.")
            disc_df = pd.DataFrame(discrepancies)
            st.dataframe(disc_df, use_container_width=True, hide_index=True)

    # Unmatched players warning
    unmatched_total = df["Unmatched"].sum()
    if unmatched_total > 0:
        with st.expander(f"⚠ {int(unmatched_total)} players had no projection match (contribute 0 pts)"):
            st.write("These are typically backup players, rookies, or players whose names didn't match the PDF.")
            st.write("They contribute 0 pts per game in the simulation — this is conservative.")

else:
    st.info("Configure your league settings in the sidebar and click **Run Simulation** to get started.")
    with st.expander("How it works"):
        st.markdown("""
        1. **Enter your Sleeper username or league ID** in the sidebar
        2. **Click Run Simulation** — the tool will:
           - Pull your league's rosters and scoring settings from Sleeper
           - Download Mike Clay's ESPN season projections for every player
           - Run thousands of simulated seasons, picking the best possible lineup each week
        3. **Results show each team's projected MaxPF** — the score they'd get if they always set the perfect lineup
        4. Teams are ranked from **lowest MaxPF (picks first) to highest**

        *MaxPF is used in many dynasty leagues to assign draft picks — it measures roster strength, not luck.*
        """)
