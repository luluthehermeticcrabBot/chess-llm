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
              starting_board = None,
              opening_name: str = "",
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

    if starting_board is not None and opening_name:
        print(f"\n{'=' * 50}")
        print(f"  {white.name} (White)  vs  {black.name} (Black)")
        print(f"  Opening: {opening_name}")
        print(f"{'=' * 50}")
    else:
        print(f"\n{'=' * 50}")
        print(f"  {white.name} (White)  vs  {black.name} (Black)")
        print(f"{'=' * 50}")

    match = ChessMatch(white, black, delay=delay, starting_board=starting_board)
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
                max_workers: int = 1, openings_mode: str = "standard",
                **player_kwargs):
    """Run a round-robin tournament (each model plays every other as both colors).

    When max_workers > 1, games run in parallel via ThreadPoolExecutor.
    ELO updates are batched after all games complete to avoid file corruption.
    openings_mode: 'standard' (default) or 'unbalanced' (forces decisive games).
    """
    total_games = len(models) * (len(models) - 1) * games_per_pair
    print(f"\n🏟  Round-Robin Tournament: {len(models)} players")
    print(f"   Models: {', '.join(models)}")
    print(f"   Games per pair: {games_per_pair}")
    print(f"   Total games: {total_games}")
    if openings_mode == "unbalanced":
        from openings import OPENINGS
        print(f"   Openings: unbalanced ({len(OPENINGS)} lines, alternating advantage)")
    if max_workers > 1:
        print(f"   Parallel workers: {max_workers}  |  ELO: batch-at-end")

    # Lazy-load openings module
    _openings_uci = None

    def _get_starting_board(opening_idx: int):
        """Return (board, opening_name) for the given opening index, or (None, '')."""
        nonlocal _openings_uci
        if openings_mode != "unbalanced":
            return None, ""
        if _openings_uci is None:
            from openings import OPENINGS
            _openings_uci = OPENINGS
        import chess
        name, moves, advantage = _openings_uci[opening_idx % len(_openings_uci)]
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        return board, name

    results = []
    start_time = time.time()

    # ── Build task list ───────────────────────────────────────────────
    tasks = []
    opening_counter = 0
    for i, white_model in enumerate(models):
        for j, black_model in enumerate(models):
            if i == j:
                continue
            for g in range(games_per_pair):
                tasks.append((white_model, black_model, g, opening_counter))
                opening_counter += 1

    # ── Sequential mode ───────────────────────────────────────────────
    if max_workers <= 1:
        for white_model, black_model, g, oi in tasks:
            board, oname = _get_starting_board(oi)
            print(f"\n── Round: {white_model} vs {black_model} "
                  f"(game {g + 1}/{games_per_pair}) ──")
            try:
                result = run_match(
                    white_model, black_model,
                    delay=delay, elo_tracker=elo_tracker,
                    starting_board=board, opening_name=oname,
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
            white_model, black_model, g, oi = task
            board, oname = _get_starting_board(oi)
            try:
                result = run_match(
                    white_model, black_model,
                    delay=delay, elo_tracker=None,
                    starting_board=board, opening_name=oname,
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
    parser.add_argument("--openings", choices=["standard", "unbalanced"],
                        default="standard",
                        help="Opening mode: 'standard' (from start position) or "
                             "'unbalanced' (pre-played lines that favor one side, "
                             "reducing draw rates in engine tournaments)")
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

    # ── TUI ───────────────────────────────────────────────────────────
    tui_group = parser.add_mutually_exclusive_group()
    tui_group.add_argument("--tui", action="store_true", default=None,
                           help="Force TUI dashboard mode")
    tui_group.add_argument("--no-tui", action="store_true", default=None,
                           help="Force plain text mode (no dashboard)")

    # ── Resume / log ──────────────────────────────────────────────────
    parser.add_argument("--resume", action="store_true",
                        help="Resume a saved tournament from tournament_state.json")
    parser.add_argument("--log", "-l", nargs="?", const="auto", default=None,
                        help="Tee output to a log file (auto-names by date; or give a path)")

    args = parser.parse_args()

    # ── Log file tee (for non-TUI mode) ───────────────────────────────
    if args.log and not sys.stdout.isatty():
        import datetime as _dt
        if args.log == "auto":
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".logs")
            os.makedirs(log_dir, exist_ok=True)
            args.log = os.path.join(log_dir, f"tournament_{ts}.txt")
        # Tee stdout to log file
        _log_file = open(args.log, "w", buffering=1)
        _orig_stdout = sys.stdout
        class _Tee:
            def write(self, data):
                _orig_stdout.write(data)
                _log_file.write(data)
            def flush(self):
                _orig_stdout.flush()
                _log_file.flush()
        sys.stdout = _Tee()
        print(f"📝 Logging to {args.log}")

    # Auto-detect TUI: use when stdout is a terminal unless explicitly disabled
    use_tui = args.tui if args.tui is not None else (
        sys.stdout.isatty() and args.no_tui is not True
    )

    # ── TUI path ──────────────────────────────────────────────────────
    if use_tui:
        from tui.tournament import TournamentApp
        from tui.state import load_state

        # Resume from saved state
        resume_completed = None
        resume_elo = None

        if args.resume:
            state = load_state()
            if state is None:
                print("❌ No saved tournament state found (tournament_state.json missing)")
                sys.exit(1)
            cfg = state["config"]
            models = cfg["models"]
            games_per_pair = cfg["games_per_pair"]
            delay = cfg["delay"]
            workers = cfg["max_workers"]
            player_kwargs = cfg["player_kwargs"]
            elo_db = cfg.get("elo_db_path")
            resume_completed = [tuple(r) for r in state["completed"]]
            resume_elo = state.get("elo_ratings", {})
            print(f"📂 Resuming tournament: {len(resume_completed)} games already completed, "
                  f"{len(models)} players")
        else:
            models = args.round_robin if args.round_robin else (
                [args.gauntlet] + (args.opponents or [])
            )
            games_per_pair = args.games
            delay = args.delay
            workers = max(args.parallel, 1)
            player_kwargs = dict(
                use_tools=not args.no_tools,
                max_retries=args.retries,
                temperature=args.temperature,
                timeout=args.timeout,
                threads=args.stockfish_threads,
                think_time=args.stockfish_time,
            )
            elo_db = args.elo_db if args.elo else None

        app = TournamentApp(
            models=models,
            games_per_pair=games_per_pair,
            delay=delay,
            elo_db_path=elo_db,
            player_kwargs=player_kwargs,
            max_workers=workers,
            resume_completed=resume_completed,
            resume_elo=resume_elo,
            openings_mode=args.openings if not args.resume else resume_elo.get("openings_mode", "standard"),
        )
        app.run()
        return

    # ELO tracker
    elo_tracker = EloTracker(args.elo_db) if args.elo else None

    player_kwargs = dict(
        use_tools=not args.no_tools,
        max_retries=args.retries,
        temperature=args.temperature,
        timeout=args.timeout,
        threads=args.stockfish_threads,
        think_time=args.stockfish_time,
    )

    if args.round_robin:
        round_robin(
            args.round_robin,
            games_per_pair=args.games,
            delay=args.delay,
            elo_tracker=elo_tracker,
            max_workers=args.parallel,
            openings_mode=args.openings,
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
