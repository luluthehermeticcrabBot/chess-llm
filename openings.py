"""Unbalanced chess openings — positions that force decisive games.

Each opening is a list of UCI moves from the starting position.
The `advantage` field indicates which side has the edge ("white" or "black").
Used with --openings unbalanced to reduce draw rates in engine tournaments.
"""

import chess
from typing import Optional


Opening = tuple[str, list[str], str]  # (name, uci_moves, advantage)


OPENINGS: list[Opening] = [
    # ── White-advantage openings (+0.5 to +1.5) ──────────────────────
    (
        "King's Gambit Accepted",
        ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "f1c4", "g5g4", "e1g1", "g4f3", "d1f3"],
        "white",
    ),
    (
        "Sicilian Smith-Morra Gambit",
        ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3", "d4c3", "b1c3", "b8c6", "g1f3", "d7d6", "f1c4"],
        "white",
    ),
    (
        "Italian Evans Gambit",
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "c2c3", "b4a5", "d2d4"],
        "white",
    ),
    (
        "Danish Gambit",
        ["e2e4", "e7e5", "d2d4", "e5d4", "c2c3", "d4c3", "f1c4", "c3b2", "c1b2"],
        "white",
    ),
    (
        "Scotch Gambit",
        ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f1c4", "f8c5", "c2c3", "d4c3", "b1c3"],
        "white",
    ),
    (
        "Vienna Gambit",
        ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "e5f4", "e4e5", "d8e7", "d1e2"],
        "white",
    ),
    (
        "French Advance Milner-Barry",
        ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3", "d8b6", "f1d3", "c5d4", "c3d4", "c6d4", "f3d4", "b6d4"],
        "white",
    ),
    (
        "Caro-Kann Panov Attack",
        ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4", "g8f6", "b1c3", "e7e6", "g1f3", "f8b4"],
        "white",
    ),
    (
        "Spanish Exchange",
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "d2d4", "e5d4", "d1d4", "d8d4", "f3d4"],
        "white",
    ),
    (
        "Grünfeld Exchange",
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3", "b2c3", "f8g7", "f1c4", "e8g8", "g1e2"],
        "white",
    ),

    # ── Black-advantage openings (White gambits declined/suboptimal) ──
    (
        "Sicilian Dragon Yugoslav",
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7", "f2f3", "e8g8", "d1d2", "b8c6"],
        "black",
    ),
    (
        "Nimzo-Indian Classical",
        ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "d7d5", "a2a3", "b4c3", "c2c3", "f6e4"],
        "black",
    ),
    (
        "King's Indian Classical",
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5", "e1g1", "b8c6", "d4d5", "c6e7"],
        "black",
    ),
    (
        "Queen's Gambit Declined Lasker",
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "h7h6", "g5h4", "f6e4", "h4e7", "d8e7"],
        "black",
    ),
    (
        "Benko Gambit",
        ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6", "b5a6", "c8a6", "b1c3", "d7d6", "g1f3", "g7g6"],
        "black",
    ),
    (
        "Pirc Defense",
        ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7", "f1e2", "e8g8", "e1g1", "c7c6", "a2a4", "b8d7"],
        "black",
    ),
    (
        "Alekhine Defense Modern",
        ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "c8g4", "f1e2", "e7e6", "e1g1", "f8e7", "c2c4", "d5b6"],
        "black",
    ),
    (
        "Scandinavian Defense",
        ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3", "c8g4", "h2h3", "g4h5", "g2g4", "h5g6", "f3e5"],
        "black",
    ),
    (
        "Dutch Leningrad",
        ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "g7g6", "g1f3", "f8g7", "e1g1", "e8g8", "c2c4", "d7d6", "b1c3", "d8e8"],
        "black",
    ),
    (
        "Modern Benoni",
        ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6", "e2e4", "g7g6", "g1f3", "f8g7", "f1e2", "e8g8"],
        "black",
    ),
]


def apply_opening(board: chess.Board, opening_index: int) -> chess.Board:
    """Apply an opening to a fresh board and return it.

    The opening is chosen modulo the pool size, so the tournament
    cycles through openings naturally as games progress.
    """
    name, moves, advantage = OPENINGS[opening_index % len(OPENINGS)]
    board.reset()
    for uci in moves:
        board.push_uci(uci)
    return board


def opening_info(opening_index: int) -> tuple[str, str]:
    """Return (name, advantage) for display/logging."""
    name, _, advantage = OPENINGS[opening_index % len(OPENINGS)]
    return name, advantage
