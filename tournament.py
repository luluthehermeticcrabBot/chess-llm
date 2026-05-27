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
import concurrent.futures
import datetime
import os
import sys
import threading
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
        return HumanPlayer(name="Human"), "human-Human"
    if spec.startswith("human:"):
        name = spec.split(":", 1)[1].strip()
        return HumanPlayer(name=name), f"human-{name}"
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


def _elo_id_from_spec(spec: str) -> str:
    """Extract the ELO key from a player spec WITHOUT creating a player instance."""
    spec = spec.strip()
    if spec.lower() == "random":
        return "random"
    if spec.lower() == "human":
        return "human-Human"
    if spec.startswith("human:"):
        name = spec.split(":", 1)[1].strip()
        return f"human-{name}"
    if spec.lower() == "stockfish":
        return "stockfish-20"
    if spec.startswith("stockfish:"):
        skill = int(spec.split(":")[1])
        return f"stockfish-{skill}"
    return spec


def run_match(white_spec: str, black_spec: str,
              delay: float = 0.5,
              elo_tracker: Optional[EloTracker] = None,
              log_dir: Optional[str] = None,
              **player_kwargs) -> str:
    """Run a single match. Returns result string ('1-0', '0-1', '1/2-1/2')."""
    from chess_llm import ChessMatch

    white, w_id = parse_player_spec(white_spec)
    black, b_id = parse_player_spec(black_spec)

    # Apply kwargs to players (LLM and Stockfish)
    for p in (white, black):
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
                max_workers: int = 1, **player_kwargs):
    """Run a round-robin tournament (each model plays every other as both colors).

    When max_workers > 1, games run in parallel via ThreadPoolExecutor.
    ELO updates are batched after all games complete to avoid file corruption.
    """
    total_games = len(models) * (len(models) - 1) * games_per_pair
    print(f"\n🏟  Round-Robin Tournament: {len(models)} players")
    print(f"   Models: {', '.join(models)}")
    print(f"   Games per pair: {games_per_pair}")
    print(f"   Total games: {total_games}")
    if max_workers > 1:
        print(f"   Parallel workers: {max_workers}  |  ELO: batch-at-end")

    results = []
    start_time = time.time()

    # ── Build task list ───────────────────────────────────────────────
    tasks = []
    for i, white_model in enumerate(models):
        for j, black_model in enumerate(models):
            if i == j:
                continue
            for g in range(games_per_pair):
                tasks.append((white_model, black_model, g))

    # ── Sequential mode ───────────────────────────────────────────────
    if max_workers <= 1:
        for white_model, black_model, g in tasks:
            print(f"\n── Round: {white_model} vs {black_model} "
                  f"(game {g + 1}/{games_per_pair}) ──")
            try:
                result = run_match(
                    white_model, black_model,
                    delay=delay, elo_tracker=elo_tracker,
                    **player_kwargs,
                )
                results.append((white_model, black_model, result))
            except Exception as e:
                print(f"  ❌ Match failed: {e}")
                results.append((white_model, black_model, "error"))
            time.sleep(0.5)

    # ── Parallel mode ─────────────────────────────────────────────────
    else:
        completed = 0
        failed = 0
        _lock = threading.Lock()

        def _run_one(task):
            white_model, black_model, g = task
            try:
                result = run_match(
                    white_model, black_model,
                    delay=delay, elo_tracker=None,  # no ELO during parallel
                    **player_kwargs,
                )
                return (white_model, black_model, result, None)
            except Exception as e:
                return (white_model, black_model, "error", str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                white_model, black_model, result, error = future.result()
                with _lock:
                    completed += 1
                    if error:
                        failed += 1
                    results.append((white_model, black_model, result))
                    pct = completed * 100 // len(tasks)
                    status = f"  ⚡ [{completed}/{len(tasks)}] {pct}%  "
                    if failed:
                        status += f"({failed} failed)  "
                    status += f"{white_model} vs {black_model} → {result}"
                    print(status)

        # ── Batch ELO update ──────────────────────────────────────────
        if elo_tracker:
            print(f"\n📊 Computing ELO ratings...")
            for white_model, black_model, result in results:
                if result == "error":
                    continue
                w_id = _elo_id_from_spec(white_model)
                b_id = _elo_id_from_spec(black_model)
                elo_tracker.update(w_id, b_id, result)
            elo_tracker.print_leaderboard()
            return results

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Tournament complete in {elapsed:.0f}s")
    print(f"  Games: {len(results)}")
    if elo_tracker and max_workers <= 1:
        elo_tracker.print_leaderboard()
    return results


def gauntlet(champion: str, opponents: list[str],
             games_per_opponent: int = 1, delay: float = 0.5,
             elo_tracker: Optional[EloTracker] = None,
             max_workers: int = 1, **player_kwargs):
    """Gauntlet: champion plays each opponent (both colors).

    When max_workers > 1, games run in parallel via ThreadPoolExecutor.
    ELO updates are batched after all games complete.
    """
    total_games = len(opponents) * games_per_opponent * 2
    print(f"\n🏟  Gauntlet: {champion} vs the field")
    print(f"   Opponents: {', '.join(opponents)}")
    print(f"   Games per opponent: {games_per_opponent * 2} (both colors)")
    if max_workers > 1:
        print(f"   Parallel workers: {max_workers}  |  ELO: batch-at-end")

    results = []
    start_time = time.time()

    # ── Build task list ───────────────────────────────────────────────
    tasks = []
    for opponent in opponents:
        for g in range(games_per_opponent):
            tasks.append((champion, opponent, g, "W"))   # champion as white
            tasks.append((opponent, champion, g, "B"))   # champion as black

    # ── Sequential mode ───────────────────────────────────────────────
    if max_workers <= 1:
        for white_model, black_model, g, _ in tasks:
            print(f"\n── Gauntlet: {white_model} (W) vs {black_model} (B) "
                  f"(game {g + 1}) ──")
            try:
                result = run_match(white_model, black_model, delay=delay,
                                   elo_tracker=elo_tracker, **player_kwargs)
                results.append((white_model, black_model, result))
            except Exception as e:
                print(f"  ❌ Match failed: {e}")
                results.append((white_model, black_model, "error"))

    # ── Parallel mode ─────────────────────────────────────────────────
    else:
        completed = 0
        failed = 0
        _lock = threading.Lock()

        def _run_one(task):
            white_model, black_model, g, _ = task
            try:
                result = run_match(
                    white_model, black_model,
                    delay=delay, elo_tracker=None,
                    **player_kwargs,
                )
                return (white_model, black_model, result, None)
            except Exception as e:
                return (white_model, black_model, "error", str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                white_model, black_model, result, error = future.result()
                with _lock:
                    completed += 1
                    if error:
                        failed += 1
                    results.append((white_model, black_model, result))
                    pct = completed * 100 // len(tasks)
                    status = f"  ⚡ [{completed}/{len(tasks)}] {pct}%  "
                    if failed:
                        status += f"({failed} failed)  "
                    status += f"{white_model} vs {black_model} → {result}"
                    print(status)

        # ── Batch ELO update ──────────────────────────────────────────
        if elo_tracker:
            print(f"\n📊 Computing ELO ratings...")
            for white_model, black_model, result in results:
                if result == "error":
                    continue
                w_id = _elo_id_from_spec(white_model)
                b_id = _elo_id_from_spec(black_model)
                elo_tracker.update(w_id, b_id, result)
            elo_tracker.print_leaderboard()
            return results

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Gauntlet complete in {elapsed:.0f}s")
    if elo_tracker and max_workers <= 1:
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
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Run N games in parallel (default: 1, sequential). "
                             "When >1 and --elo is set, ELO is batch-computed at the end.")
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
    parser.add_argument(
        "--stockfish-threads", type=int, default=None,
        help="Number of CPU threads for Stockfish (default: use all cores). "
             "Set to 1 or 2 to keep the fan quiet during tournaments.",
    )

    args = parser.parse_args()

    # ELO tracker
    elo_tracker = EloTracker(args.elo_db) if args.elo else None

    player_kwargs = dict(
        use_tools=not args.no_tools,
        max_retries=args.retries,
        temperature=args.temperature,
        timeout=args.timeout,
        threads=args.stockfish_threads,
        skill_level=args.stockfish_skill,
        think_time=args.stockfish_time,
    )

    if args.round_robin:
        round_robin(
            args.round_robin,
            games_per_pair=args.games,
            delay=args.delay,
            elo_tracker=elo_tracker,
            max_workers=args.parallel,
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
            max_workers=args.parallel,
            **player_kwargs,
        )


if __name__ == "__main__":
    main()
