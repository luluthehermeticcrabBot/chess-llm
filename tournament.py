#!/usr/bin/env python3
"""
Tournament runner for LLM Chess.

Runs round-robin or gauntlet tournaments between models, tracking ELO.
Imports chess_llm's player classes and match engine directly.

Usage:
    # Round-robin: every model plays every other (both colors)
    python tournament.py --round-robin model-a model-b model-c

    # Gauntlet: one model plays against all others
    python tournament.py --gauntlet champion --opponents challenger1 challenger2

    # With ELO tracking and custom settings
    python tournament.py --round-robin docker/ai/qwen3.5:9B \
        docker/ai/ministral3:3B stockfish --elo --no-tools --delay 0
"""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent to path so we can import chess_llm modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elo import EloTracker


def parse_player_spec(spec: str):
    """Parse a player spec string. Returns (player_type, label, kwargs)."""
    from chess_llm import (
        HumanPlayer, RandomPlayer, StockfishPlayer, LLMPlayer,
    )

    spec = spec.strip()

    if spec.lower() == "human":
        return HumanPlayer(name="Human"), "human"
    if spec.lower() == "random":
        return RandomPlayer(name="Random"), "random"
    if spec.lower() == "stockfish":
        return StockfishPlayer(name="Stockfish 20", skill_level=20), "stockfish-20"
    if spec.startswith("stockfish:"):
        skill = int(spec.split(":")[1])
        return StockfishPlayer(name=f"Stockfish {skill}", skill_level=skill), f"stockfish-{skill}"

    # LLM player
    player = LLMPlayer(
        model=spec,
        name=spec,
        use_tools=False,  # default for tournaments: text mode is more reliable
    )
    return player, spec


def run_match(white_spec: str, black_spec: str,
              delay: float = 0.5,
              elo_tracker: Optional[EloTracker] = None,
              log_dir: Optional[str] = None,
              **player_kwargs) -> str:
    """Run a single match. Returns result string ('1-0', '0-1', '1/2-1/2')."""
    from chess_llm import ChessMatch

    white, w_id = parse_player_spec(white_spec)
    black, b_id = parse_player_spec(black_spec)

    # Apply kwargs to LLMPlayers
    for p in (white, black):
        if hasattr(p, "temperature"):
            for k, v in player_kwargs.items():
                if hasattr(p, k):
                    setattr(p, k, v)

    # Pre-flight
    for p in (white, black):
        if hasattr(p, "check_connectivity"):
            try:
                p.check_connectivity()
            except Exception as e:
                print(f"   ⚠ {p.name}: connectivity check failed: {e}")

    print(f"\n{'=' * 50}")
    print(f"  {white.name} (White)  vs  {black.name} (Black)")
    print(f"{'=' * 50}")

    match = ChessMatch(white, black, delay=delay)
    result = match.play()

    # Save PGN
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    w_name = white.name.replace("/", "_").replace(" ", "_")[:40]
    b_name = black.name.replace("/", "_").replace(" ", "_")[:40]
    games_dir = "games"
    os.makedirs(games_dir, exist_ok=True)
    pgn_path = f"{games_dir}/{ts}_{w_name}_vs_{b_name}.pgn"
    match.save_pgn(pgn_path)

    # ELO
    if elo_tracker:
        updated = elo_tracker.update(w_id, b_id, result)
        print(f"  ELO: {w_id} → {updated['white']['rating']}  |  "
              f"{b_id} → {updated['black']['rating']}")

    return result


def round_robin(models: list[str], games_per_pair: int = 1,
                delay: float = 0.5, elo_tracker: Optional[EloTracker] = None,
                **player_kwargs):
    """Run a round-robin tournament (each model plays every other as both colors)."""
    print(f"\n🏟  Round-Robin Tournament: {len(models)} players")
    print(f"   Models: {', '.join(models)}")
    print(f"   Games per pair: {games_per_pair}")
    print(f"   Total games: {len(models) * (len(models) - 1) * games_per_pair}")

    results = []
    start_time = time.time()

    for i, white_model in enumerate(models):
        for j, black_model in enumerate(models):
            if i == j:
                continue
            for g in range(games_per_pair):
                print(f"\n── Round: {white_model} vs {black_model} "
                      f"(game {g + 1}/{games_per_pair}) ──")
                try:
                    result = run_match(
                        white_model, black_model,
                        delay=delay,
                        elo_tracker=elo_tracker,
                        **player_kwargs,
                    )
                    results.append((white_model, black_model, result))
                except Exception as e:
                    print(f"  ❌ Match failed: {e}")
                    results.append((white_model, black_model, "error"))
                time.sleep(0.5)  # brief pause between games

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Tournament complete in {elapsed:.0f}s")
    print(f"  Games: {len(results)}")
    if elo_tracker:
        elo_tracker.print_leaderboard()
    return results


def gauntlet(champion: str, opponents: list[str],
             games_per_opponent: int = 1, delay: float = 0.5,
             elo_tracker: Optional[EloTracker] = None, **player_kwargs):
    """Gauntlet: champion plays each opponent (both colors)."""
    print(f"\n🏟  Gauntlet: {champion} vs the field")
    print(f"   Opponents: {', '.join(opponents)}")
    print(f"   Games per opponent: {games_per_opponent * 2} (both colors)")

    results = []
    start_time = time.time()

    for opponent in opponents:
        for g in range(games_per_opponent):
            # Champion as white
            print(f"\n── Gauntlet: {champion} (W) vs {opponent} (B) (game {g + 1}) ──")
            try:
                result = run_match(champion, opponent, delay=delay,
                                   elo_tracker=elo_tracker, **player_kwargs)
                results.append((champion, opponent, result))
            except Exception as e:
                print(f"  ❌ Match failed: {e}")

            # Champion as black
            print(f"\n── Gauntlet: {opponent} (W) vs {champion} (B) (game {g + 1}) ──")
            try:
                result = run_match(opponent, champion, delay=delay,
                                   elo_tracker=elo_tracker, **player_kwargs)
                results.append((opponent, champion, result))
            except Exception as e:
                print(f"  ❌ Match failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Gauntlet complete in {elapsed:.0f}s")
    if elo_tracker:
        elo_tracker.print_leaderboard()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM Chess Tournament Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python tournament.py --round-robin stockfish docker/ai/qwen3.5:9B docker/ai/ministral3:3B\n"
               "  python tournament.py --gauntlet docker/ai/qwen3.5:9B --opponents stockfish random\n"
               "  python tournament.py --round-robin model-a model-b --elo --delay 0",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--round-robin", nargs="+", metavar="MODEL",
                      help="Round-robin: every model plays every other (both colors)")
    mode.add_argument("--gauntlet", metavar="CHAMPION",
                      help="Gauntlet: champion plays all opponents")

    parser.add_argument("--opponents", nargs="+", metavar="MODEL",
                        help="Opponents for gauntlet mode")
    parser.add_argument("--games", type=int, default=1,
                        help="Games per pair (default: 1)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between moves in seconds (default: 0.5)")
    parser.add_argument("--elo", action="store_true",
                        help="Track ELO ratings")
    parser.add_argument("--elo-db", default="ratings.json")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool calling for LLM players")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--stockfish-skill", type=int, default=20)
    parser.add_argument("--stockfish-time", type=float, default=0.1)

    args = parser.parse_args()

    # ELO tracker
    elo_tracker = EloTracker(args.elo_db) if args.elo else None

    player_kwargs = dict(
        use_tools=not args.no_tools,
        max_retries=args.retries,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    if args.round_robin:
        round_robin(
            args.round_robin,
            games_per_pair=args.games,
            delay=args.delay,
            elo_tracker=elo_tracker,
            **player_kwargs,
        )
    elif args.gauntlet:
        if not args.opponents:
            print("❌ --gauntlet requires --opponents")
            sys.exit(1)
        gauntlet(
            args.gauntlet,
            args.opponents,
            games_per_opponent=args.games,
            delay=args.delay,
            elo_tracker=elo_tracker,
            **player_kwargs,
        )


if __name__ == "__main__":
    main()
