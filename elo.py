"""
ELO rating system for LLM Chess.

Standard chess ELO with K=32 for new/provisional players.
Ratings persisted as JSON. Player IDs are model names (e.g.
"docker/ai/qwen3.5:9B-UD-Q4_K_XL") or custom labels.

Usage:
    from elo import EloTracker
    tracker = EloTracker("ratings.json")
    tracker.update("model-a", "model-b", "1-0")
    tracker.leaderboard()
"""

import json
import math
import os
from pathlib import Path
from typing import Optional


DEFAULT_K = 32
PROVISIONAL_K = 64        # higher K for players with < 10 games
DEFAULT_RATING = 1200
PROVISIONAL_GAMES = 10    # games before switching from provisional K


class EloTracker:
    """Persistent ELO ratings for chess players (models, humans, engines)."""

    def __init__(self, db_path: str = "ratings.json"):
        self.db_path = db_path
        self._ratings: dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                self._ratings = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self._ratings, f, indent=2)

    # ── public API ───────────────────────────────────────────────────────

    def get(self, player_id: str) -> dict:
        """Return {rating, games, wins, losses, draws} or defaults."""
        return self._ratings.get(
            player_id,
            {"rating": DEFAULT_RATING, "games": 0, "wins": 0, "losses": 0, "draws": 0},
        )

    def get_rating(self, player_id: str) -> int:
        return self.get(player_id)["rating"]

    def update(self, white_id: str, black_id: str, result: str) -> dict:
        """
        Update ELO for both players after a game.

        result: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
        Returns the two new rating entries as a dict.
        """
        w = self.get(white_id)
        b = self.get(black_id)

        w_rating, b_rating = w["rating"], b["rating"]

        # Expected scores
        w_expected = 1.0 / (1.0 + 10.0 ** ((b_rating - w_rating) / 400.0))
        b_expected = 1.0 - w_expected

        # Actual scores
        if result == "1-0":
            w_score, b_score = 1.0, 0.0
        elif result == "0-1":
            w_score, b_score = 0.0, 1.0
        else:
            w_score, b_score = 0.5, 0.5

        # K-factor: higher for provisional players
        w_k = PROVISIONAL_K if w["games"] < PROVISIONAL_GAMES else DEFAULT_K
        b_k = PROVISIONAL_K if b["games"] < PROVISIONAL_GAMES else DEFAULT_K

        # New ratings
        w_new = round(w_rating + w_k * (w_score - w_expected))
        b_new = round(b_rating + b_k * (b_score - w_expected))

        # Update stats
        w["rating"] = w_new
        w["games"] += 1
        b["rating"] = b_new
        b["games"] += 1

        if result == "1-0":
            w["wins"] = w.get("wins", 0) + 1
            b["losses"] = b.get("losses", 0) + 1
        elif result == "0-1":
            w["losses"] = w.get("losses", 0) + 1
            b["wins"] = b.get("wins", 0) + 1
        else:
            w["draws"] = w.get("draws", 0) + 1
            b["draws"] = b.get("draws", 0) + 1

        self._ratings[white_id] = w
        self._ratings[black_id] = b
        self._save()

        return {"white": w, "black": b}

    def leaderboard(self, n: int = 20) -> list[tuple[str, dict]]:
        """Return top N players sorted by rating (descending)."""
        ranked = sorted(
            self._ratings.items(),
            key=lambda x: x[1]["rating"],
            reverse=True,
        )
        return ranked[:n]

    def print_leaderboard(self, n: int = 20):
        """Pretty-print the leaderboard."""
        board = self.leaderboard(n)
        if not board:
            print("No ratings yet. Play some games!")
            return

        # Column widths
        rank_w = 4
        name_w = max(len(name) for name, _ in board) + 2
        name_w = min(name_w, 55)  # clamp long names

        header = f"{'#':>{rank_w}}  {'Player':<{name_w}}  {'ELO':>5}  {'G':>4}  {'W':>4}  {'L':>4}  {'D':>4}"
        line = "─" * len(header)

        print(f"\n🏆  ELO Leaderboard\n")
        print(line)
        print(header)
        print(line)

        for i, (name, stats) in enumerate(board, 1):
            display = name if len(name) <= name_w else name[:name_w - 2] + "…"
            print(
                f"{i:>{rank_w}}  {display:<{name_w}}  "
                f"{stats['rating']:>5}  "
                f"{stats['games']:>4}  "
                f"{stats.get('wins', 0):>4}  "
                f"{stats.get('losses', 0):>4}  "
                f"{stats.get('draws', 0):>4}"
            )
        print(line)

    def predict(self, white_id: str, black_id: str) -> tuple[float, float]:
        """Return expected scores for both players (white, black)."""
        w = self.get_rating(white_id)
        b = self.get_rating(black_id)
        w_exp = 1.0 / (1.0 + 10.0 ** ((b - w) / 400.0))
        return w_exp, 1.0 - w_exp
