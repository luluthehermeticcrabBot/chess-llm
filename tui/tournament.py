"""Tournament TUI dashboard for LLM Chess."""

import queue
import threading
import time
from typing import Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import (
    Header, Footer, Static, ProgressBar, DataTable, RichLog,
    Label, Button,
)
from textual.reactive import reactive
from textual import work

from elo import EloTracker
from .common import (
    TUI_CSS, BG, SURFACE, ACCENT, HIGHLIGHT,
    TEXT_PRIMARY, TEXT_SECONDARY, SUCCESS, WARNING, ERROR, DRAW_COLOR,
)
from .state import save_state, clear_state, compute_remaining_tasks


# ── Lightweight ELO key extractor (no player creation) ──────────────────
def _elo_id(spec: str) -> str:
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


class ELOComputer:
    """In-memory ELO computer — no disk I/O, fast for live updates."""

    def __init__(self, initial_rating: int = 1200):
        self.initial = initial_rating
        self.ratings: dict[str, dict] = {}

    def _get(self, pid: str) -> dict:
        return self.ratings.get(pid, {
            "rating": self.initial, "games": 0, "wins": 0, "losses": 0, "draws": 0,
        })

    def add_game(self, white_id: str, black_id: str, result: str):
        w = self._get(white_id)
        b = self._get(black_id)

        w_r, b_r = w["rating"], b["rating"]
        w_exp = 1.0 / (1.0 + 10.0 ** ((b_r - w_r) / 400.0))

        K = 64 if w["games"] < 10 else 32
        K_b = 64 if b["games"] < 10 else 32

        if result == "1-0":
            w_score, b_score = 1.0, 0.0
            w["wins"] += 1
            b["losses"] += 1
        elif result == "0-1":
            w_score, b_score = 0.0, 1.0
            w["losses"] += 1
            b["wins"] += 1
        else:
            w_score, b_score = 0.5, 0.5
            w["draws"] = w.get("draws", 0) + 1
            b["draws"] = b.get("draws", 0) + 1

        w["rating"] = round(w_r + K * (w_score - w_exp))
        b["rating"] = round(b_r + K_b * (b_score - (1.0 - w_exp)))
        w["games"] += 1
        b["games"] += 1

        self.ratings[white_id] = w
        self.ratings[black_id] = b

    def leaderboard(self) -> list[tuple[str, dict]]:
        return sorted(
            self.ratings.items(),
            key=lambda x: x[1]["rating"],
            reverse=True,
        )


class TournamentApp(App):
    """Textual TUI for tournament dashboard."""

    CSS = TUI_CSS
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("s", "save", "Save progress"),
    ]

    def __init__(
        self,
        models: list[str],
        games_per_pair: int,
        delay: float,
        elo_db_path: Optional[str],
        player_kwargs: dict,
        max_workers: int,
        resume_completed: Optional[list[tuple[str, str, str]]] = None,
        resume_elo: Optional[dict] = None,
        openings_mode: str = "standard",
    ):
        super().__init__()
        self.models = models
        self.games_per_pair = games_per_pair
        self.delay = delay
        self.elo_db_path = elo_db_path
        self.player_kwargs = player_kwargs
        self.max_workers = max_workers
        self.openings_mode = openings_mode

        # Resume state (pre-seeded results from a previous run)
        self._resume_completed = resume_completed or []
        self._resume_elo = resume_elo or {}

        # State
        self._completed: list[tuple[str, str, str]] = []
        self._active: dict[int, tuple[str, str, float]] = {}
        self._results_queue: queue.Queue = queue.Queue()
        self._task_list: list[tuple[str, str]] = []
        self._total_tasks = 0
        self._start_time = 0.0
        self._paused = False
        self._finished = False
        self._stop_flag = threading.Event()
        self.elo = ELOComputer()
        self._task_id = 0
        self._lock = threading.Lock()

    # ── Compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="progress-section"):
            yield Label("", id="progress-label")
            yield ProgressBar(id="progress-bar", total=100, show_eta=False)
        yield Static("", id="active-games")
        yield DataTable(id="elo-table", cursor_type="row")
        yield Static("", id="matchup-summary")
        yield RichLog(id="worker-log", max_lines=20, highlight=True, markup=True)
        yield Footer()

    # ── Mount ────────────────────────────────────────────────────────────

    def on_mount(self):
        # Pre-load resume data into ELO computer
        if self._resume_elo:
            self.elo.ratings = self._resume_elo

        # Build remaining task list (skipping already-completed)
        self._task_list = compute_remaining_tasks(
            self.models, self.games_per_pair, self._resume_completed
        )
        self._completed = list(self._resume_completed)
        self._total_tasks = len(self._task_list) + len(self._resume_completed)
        self._start_time = time.time()

        # Setup ELO table columns
        elo_table = self.query_one("#elo-table", DataTable)
        elo_table.add_columns("#", "Player", "ELO", "G", "W", "L", "D")

        # Show resume info
        if self._resume_completed:
            self.notify(
                f"Resumed with {len(self._resume_completed)} completed games, "
                f"{len(self._task_list)} remaining",
                title="📂 Resume"
            )

        self._tui_log(f"⚙ Tournament: {len(self.models)} players, "
                      f"{self._total_tasks} tasks, {self.max_workers} workers")
        self._tui_log(f"📋 Task list: {len(self._task_list)} to run, "
                      f"{len(self._resume_completed)} already done")
        self._tui_log("🔧 Starting worker thread...")

        # Start workers
        self._start_workers()
        # Poll for results every 200ms
        self.set_interval(0.2, self._poll)

    # ── Thread-safe TUI logging ─────────────────────────────────────────

    def _tui_log(self, msg: str):
        """Log a message to the visible RichLog widget (thread-safe)."""
        log_widget = self.query_one("#worker-log", RichLog)
        log_widget.write(msg)

    def _tui_log_safe(self, msg: str):
        """Post a log message from any thread."""
        self.call_from_thread(self._tui_log, msg)

    # ── Worker management ────────────────────────────────────────────────

    @work(thread=True)
    def _start_workers(self):
        """Run tournament games sequentially in this thread (no nested executor).

        Each game runs, result goes on the queue, then the next game starts.
        This avoids ThreadPoolExecutor-within-Textual-thread complexity that
        causes silent hangs and zero-progress TUI.
        """
        self._tui_log_safe("[worker] Starting tournament worker thread")

        # Pre-load openings if needed
        openings_list = None
        if self.openings_mode == "unbalanced":
            from openings import OPENINGS
            openings_list = OPENINGS
            import chess as _chess

        _opening_idx = 0

        def _get_opening():
            nonlocal _opening_idx
            if openings_list is None:
                return None, ""
            idx = _opening_idx
            _opening_idx += 1
            name, moves, advantage = openings_list[idx % len(openings_list)]
            board = _chess.Board()
            for uci in moves:
                board.push_uci(uci)
            return board, name

        if not self._task_list:
            self._finished = True
            self._tui_log_safe("[worker] No tasks — exiting")
            return

        self._tui_log_safe(f"[worker] {len(self._task_list)} tasks to run ({self.max_workers} max-parallel hint)")

        # Resolve run_match once
        import sys
        _mod = sys.modules.get("__main__")
        if _mod is not None and hasattr(_mod, "run_match"):
            _run_match_fn = _mod.run_match
        else:
            from tournament import run_match as _run_match_fn
        self._tui_log_safe(f"[worker] run_match resolved: {_run_match_fn}")

        for idx, (wm, bm) in enumerate(self._task_list):
            if self._stop_flag.is_set():
                self._tui_log_safe(f"[worker] Stop flag set — aborting at task {idx}/{len(self._task_list)}")
                break

            with self._lock:
                tid = self._task_id
                self._task_id += 1
                self._active[tid] = (wm, bm, time.time())
            self._tui_log_safe(f"[worker] Starting task {idx}: {wm} vs {bm}")

            board, oname = _get_opening()
            try:
                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    result = _run_match_fn(
                        wm, bm,
                        delay=self.delay, elo_tracker=None,
                        starting_board=board, opening_name=oname,
                        **self.player_kwargs,
                    )
                finally:
                    sys.stdout = old_stdout
                self._results_queue.put((wm, bm, result))
                self._tui_log_safe(f"[worker] Task {idx} done: {result}")
            except Exception as e:
                self._tui_log_safe(f"[worker] Task {idx} FAILED: {e}")
                self._results_queue.put((wm, bm, "error"))

            with self._lock:
                self._active.pop(tid, None)

        self._finished = True
        self._tui_log_safe("[worker] All tasks complete")

    # ── Polling ──────────────────────────────────────────────────────────

    def _poll(self):
        """Drain result queue and update dashboard."""
        # Drain queue
        while not self._results_queue.empty():
            try:
                white, black, result = self._results_queue.get_nowait()
                self._completed.append((white, black, result))
                if result != "error":
                    w_id = _elo_id(white)
                    b_id = _elo_id(black)
                    self.elo.add_game(w_id, b_id, result)
            except queue.Empty:
                break

        self._update_progress()
        self._update_active_games()
        self._update_elo_table()
        self._update_matchup_summary()

        if self._finished and len(self._completed) >= self._total_tasks:
            self._on_complete()

    # ── Widget updates ───────────────────────────────────────────────────

    def _update_progress(self):
        done = len(self._completed)
        total = self._total_tasks
        pct = done * 100 // total if total > 0 else 0
        elapsed = time.time() - self._start_time

        if done > 0 and not self._finished:
            eta = elapsed / (done - len(self._resume_completed)) * (total - done) if done > len(self._resume_completed) else 0
            eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}m"
        else:
            eta_str = "..."

        elapsed_str = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed/60:.1f}m"

        label = self.query_one("#progress-label", Label)
        status = "🏁 Complete!" if self._finished else "⚡ Running"
        if self._paused:
            status = "⏸ Paused"
        label.update(
            f"  {status}  [{done}/{total}] {pct}%  "
            f"⏱ {elapsed_str} elapsed  ·  ~{eta_str} remaining  "
            f"·  {self.max_workers} workers"
        )

        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=total, progress=done)

    def _update_active_games(self):
        active = self.query_one("#active-games", Static)
        if not self._active:
            active.update("  No active games.")
            return

        lines = ["  ▶ Live games:"]
        with self._lock:
            items = list(self._active.items())
        for tid, (wm, bm, started) in items[:8]:
            elapsed = time.time() - started
            lines.append(f"    {wm}  vs  {bm}   ···  {elapsed:.0f}s")
        if len(items) > 8:
            lines.append(f"    ... and {len(items) - 8} more")
        active.update("\n".join(lines))

    def _update_elo_table(self):
        table = self.query_one("#elo-table", DataTable)
        table.clear()
        table.add_columns("#", "Player", "ELO", "G", "W", "L", "D")

        board = self.elo.leaderboard()
        for i, (pid, stats) in enumerate(board[:20], 1):
            table.add_row(
                str(i), pid,
                str(stats["rating"]),
                str(stats["games"]),
                str(stats.get("wins", 0)),
                str(stats.get("losses", 0)),
                str(stats.get("draws", 0)),
            )

    def _update_matchup_summary(self):
        widget = self.query_one("#matchup-summary", Static)
        if not self._completed:
            widget.update("")
            return

        summary: dict[str, dict[str, int]] = {}
        for wm, bm, result in self._completed:
            wid = _elo_id(wm)
            bid = _elo_id(bm)
            for pid in (wid, bid):
                if pid not in summary:
                    summary[pid] = {"W": 0, "D": 0, "L": 0}
            if result == "1-0":
                summary[wid]["W"] += 1
                summary[bid]["L"] += 1
            elif result == "0-1":
                summary[wid]["L"] += 1
                summary[bid]["W"] += 1
            else:
                summary[wid]["D"] += 1
                summary[bid]["D"] += 1

        lines = ["  📊  Results:  W–D–L"]
        board = self.elo.leaderboard()
        for pid, _ in board[:10]:
            if pid in summary:
                s = summary[pid]
                short = pid if len(pid) <= 16 else pid[:13] + "..."
                lines.append(f"    {short:<17} {s['W']}–{s['D']}–{s['L']}")

        widget.update("\n".join(lines))

    def _save_elo(self):
        """Save current ELO ratings to disk."""
        if self.elo_db_path and self.elo.ratings:
            from elo import EloTracker
            tracker = EloTracker(self.elo_db_path)
            for pid, stats in self.elo.ratings.items():
                tracker._ratings[pid] = stats
            tracker._save()
            return True
        return False

    def _on_complete(self):
        """Tournament finished — show final state and save ELO."""
        self.query_one("#progress-label", Label).update(
            "  🏁 Tournament complete!  Press q to quit."
        )
        if self._save_elo():
            self.notify(f"ELO saved to {self.elo_db_path}", title="✅ Done")
        clear_state()

    # ── Actions ──────────────────────────────────────────────────────────

    def action_quit(self):
        """Quit: save ELO, stop workers, then exit."""
        self._stop_flag.set()
        # Drain any remaining results
        self._poll()
        if not self._finished:
            self._save_elo()
            clear_state()
        self.exit()

    def action_toggle_pause(self):
        self._paused = not self._paused
        self.notify("Paused" if self._paused else "Resumed")

    def action_save(self):
        """Save tournament progress to disk for later resume."""
        path = save_state(
            models=self.models,
            games_per_pair=self.games_per_pair,
            delay=self.delay,
            max_workers=self.max_workers,
            player_kwargs=self.player_kwargs,
            completed=self._completed,
            elo_ratings=self.elo.ratings,
            elo_db_path=self.elo_db_path,
            started_at=self._start_time,
        )
        self.notify(f"Saved {len(self._completed)} games to {path}", title="💾 Saved")
