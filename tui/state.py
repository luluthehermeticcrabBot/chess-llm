"""Tournament state persistence — save/resume support."""

import json
import os
import time
from typing import Optional


STATE_FILE = "tournament_state.json"


def save_state(
    models: list[str],
    games_per_pair: int,
    delay: float,
    max_workers: int,
    player_kwargs: dict,
    completed: list[tuple[str, str, str]],
    elo_ratings: dict,
    elo_db_path: Optional[str],
    started_at: float,
):
    """Save tournament progress so it can be resumed later."""
    state = {
        "config": {
            "models": models,
            "games_per_pair": games_per_pair,
            "delay": delay,
            "max_workers": max_workers,
            "player_kwargs": player_kwargs,
            "elo_db_path": elo_db_path,
        },
        "completed": [[w, b, r] for w, b, r in completed],
        "elo_ratings": elo_ratings,
        "started_at": started_at,
        "saved_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return STATE_FILE


def load_state() -> dict | None:
    """Load a saved tournament state, or None if no save exists."""
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE) as f:
        return json.load(f)


def clear_state():
    """Remove the saved state file (called on clean completion)."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def compute_remaining_tasks(
    models: list[str],
    games_per_pair: int,
    completed: list[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Build task list, excluding already-completed matchups."""
    # Count completed games per (white, black) pair
    done_counts: dict[tuple[str, str], int] = {}
    for w, b, r in completed:
        key = (w, b)
        done_counts[key] = done_counts.get(key, 0) + 1

    remaining = []
    for i, wm in enumerate(models):
        for j, bm in enumerate(models):
            if i == j:
                continue
            needed = games_per_pair - done_counts.get((wm, bm), 0)
            for _ in range(needed):
                remaining.append((wm, bm))
    return remaining
