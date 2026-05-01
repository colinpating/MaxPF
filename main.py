"""
MaxPF Dynasty Fantasy Football Projection Tool

Projects each team's Maximum Possible Fantasy Points (MaxPF) over a 14-week
regular season using Sleeper roster data and Mike Clay's ESPN projections.

Usage:
  python main.py --username myuser --season 2026
  python main.py --league-id 123456789 --simulations 10000 --output-csv results.csv
  python main.py --league-id 123456789 --pdf-path ./clay2026.pdf --include-ir
  python main.py --league-id 123456789 --fast --weekly-breakdown
"""

import argparse
import sys

import sleeper as sl
import projections as proj
import simulator as sim
import output as out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project dynasty league MaxPF using Sleeper rosters + Clay projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--league-id", metavar="ID", help="Sleeper league ID")
    source.add_argument("--username", metavar="USER", help="Sleeper username (prompts to select league)")

    parser.add_argument("--season", type=int, default=None,
                        help="NFL season year (default: current season from Sleeper)")
    parser.add_argument("--simulations", type=int, default=5000, metavar="N",
                        help="Number of Monte Carlo simulations (default: 5000)")
    parser.add_argument("--weeks", type=int, default=14, metavar="N",
                        help="Regular season weeks to simulate (default: 14)")
    parser.add_argument("--pdf-path", metavar="PATH",
                        help="Local path to Clay projections PDF (downloads if omitted)")
    parser.add_argument("--include-ir", action="store_true",
                        help="Include IR/reserve players in MaxPF calculation")
    parser.add_argument("--output-csv", metavar="PATH",
                        help="Export results to CSV file")
    parser.add_argument("--fast", action="store_true",
                        help="Use greedy lineup selection (~10x faster, ~5%% less accurate)")
    parser.add_argument("--weekly-breakdown", action="store_true",
                        help="Print per-week average optimal scores in addition to season totals")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve season
    season = args.season
    if season is None:
        try:
            state = sl.get_nfl_state()
            season = int(state.get("season", 2026))
        except Exception:
            season = 2026
    print(f"Season: {season}")

    # Resolve league
    league_id = sl.resolve_league_id(args.league_id, args.username, season)
    print(f"League ID: {league_id}")

    # Fetch league metadata
    print("Fetching league info...")
    league = sl.get_league(league_id)
    print(f"League: {league.name} ({league.total_rosters} teams)")

    if not league.roster_positions:
        print("ERROR: No roster positions found in league settings.", file=sys.stderr)
        sys.exit(1)

    # Fetch rosters and users
    print("Fetching rosters and users...")
    rosters = sl.get_rosters(league_id)
    users = sl.get_users(league_id)
    players_db = sl.get_players()
    print(f"Loaded {len(rosters)} rosters, {len(users)} users, {len(players_db):,} NFL players")

    # Load projections
    print("\nLoading player projections...")
    projections = proj.load_projections(
        pdf_path=args.pdf_path,
        scoring_settings=league.scoring_settings,
    )
    print(f"Loaded {len(projections)} player projections")

    # Run simulation
    results = sim.run_simulation(
        rosters=rosters,
        players_db=players_db,
        projections=projections,
        roster_positions=league.roster_positions,
        n_sims=args.simulations,
        n_weeks=args.weeks,
        include_ir=args.include_ir,
        use_greedy=args.fast,
        seed=args.seed,
    )

    # Display results
    out.display_results(
        results=results,
        users=users,
        rosters=rosters,
        league_name=league.name,
        n_sims=args.simulations,
        n_weeks=args.weeks,
        include_ir=args.include_ir,
    )

    if args.weekly_breakdown:
        out.display_weekly_breakdown(results, users, rosters)

    if args.output_csv:
        out.export_csv(results, users, rosters, args.output_csv, n_weeks=args.weeks)


if __name__ == "__main__":
    main()
