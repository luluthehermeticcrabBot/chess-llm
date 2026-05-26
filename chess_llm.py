#!/usr/bin/env python3
"""
LLM Chess — pit language models against each other in chess.

Players:
  llm     — any model via litellm + provider shortcuts (see below)
  human   — you, via terminal input
  random  — random legal move (baseline benchmark)
  stockfish — Stockfish engine

Provider shortcuts (auto-detect API key from env):
  opencode-go/deepseek-v4-pro   →  https://opencode.ai/zen/go/v1  ($OPENCODE_GO_API_KEY)
  opencode-zen/deepseek-v4-pro  →  https://opencode.ai/zen/v1     ($OPENCODE_ZEN_API_KEY)
  openrouter/anthropic/claude-sonnet-4 → https://openrouter.ai   ($OPENROUTER_API_KEY)
  openai/gpt-4o                 →  https://api.openai.com/v1      ($OPENAI_API_KEY)
  anthropic/claude-sonnet-4     →  native Anthropic API            ($ANTHROPIC_API_KEY)
  groq/llama-3.3-70b            →  native Groq API                ($GROQ_API_KEY)
  ollama/llama3                 →  local Ollama                    (no key needed)

Usage:
  python chess_llm.py                          # LLM vs LLM (default models)
  python chess_llm.py --white human            # You play white
  python chess_llm.py --white opencode-go/deepseek-v4-pro --black openrouter/anthropic/claude-sonnet-4
  python chess_llm.py --model opencode-go/deepseek-v4-pro  # both sides same model
  python chess_llm.py --white human --black stockfish      # You vs Stockfish
"""

import argparse
import datetime
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

# Save real stderr before we possibly replace it (litellm spam suppression)
_REAL_STDERR = sys.stderr

# Debug mode: set LLM_CHESS_DEBUG=1 (or DEBUG_CHESS_LLM=1) to see model responses
DEBUG = (
    os.environ.get("LLM_CHESS_DEBUG", "") == "1"
    or os.environ.get("DEBUG_CHESS_LLM", "") == "1"
)

# ── Provider presets ──────────────────────────────────────────────────────────
#   prefix            → (api_base,                          api_key_env_var)
#   opencode-go/model → https://opencode.ai/zen/go/v1       OPENCODE_GO_API_KEY
#   opencode-zen/model→ https://opencode.ai/zen/v1          OPENCODE_ZEN_API_KEY
#   openrouter/model  → https://openrouter.ai/api/v1        OPENROUTER_API_KEY
#   openai/model      → https://api.openai.com/v1           OPENAI_API_KEY
#   anthropic/model   → (native litellm, no override needed) ANTHROPIC_API_KEY

PROVIDER_PRESETS: dict[str, tuple[str, str]] = {
    "opencode-go":  ("https://opencode.ai/zen/go/v1", "OPENCODE_GO_API_KEY"),
    "opencode-zen": ("https://opencode.ai/zen/v1",    "OPENCODE_ZEN_API_KEY"),
    "openrouter":   ("https://openrouter.ai/api/v1",  "OPENROUTER_API_KEY"),
    "openai":       ("https://api.openai.com/v1",     "OPENAI_API_KEY"),
}


def _resolve_provider(model: str) -> tuple[str, str | None, str | None]:
    """
    Parse a model string like 'opencode-go/deepseek-v4-pro' into
    (litellm_model, api_base_override, api_key).

    For known prefixes (opencode-go, opencode-zen, openrouter, openai),
    we set the base URL and read the API key from the matching env var.
    The returned litellm_model strips the prefix for OpenAI-compatible
    endpoints, or keeps it for providers litellm knows natively (openrouter).

    For unrecognised prefixes (anthropic/, groq/, ollama/ etc.) we leave
    api_base + api_key as None and let litellm use its defaults.
    """
    for prefix, (base_url, key_env) in PROVIDER_PRESETS.items():
        if model.startswith(prefix + "/"):
            model_name = model[len(prefix) + 1:]  # e.g. "deepseek-v4-pro"
            api_key = os.environ.get(key_env)
            # openrouter is known to litellm natively — keep the prefix
            if prefix == "openrouter":
                return model, base_url, api_key
            # For OpenAI-compatible providers, use openai/ prefix so litellm
            # routes through its OpenAI-compatible handler
            return f"openai/{model_name}", base_url, api_key

    return model, None, None


# ── Silence litellm's stderr spam once at module load ────────────────────────
# litellm writes "Provider List: https://...", "Give Feedback / Get Help", etc.
# to stderr on every error. We redirect stderr to /dev/null globally unless
# LLM_CHESS_DEBUG=1 (or DEBUG_CHESS_LLM=1).
if not DEBUG:
    import logging
    logging.getLogger("litellm").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    # Redirect stderr to /dev/null for the lifetime of the process.
    # The --log tee in main() replaces sys.stderr again after this,
    # so stderr still ends up in the log file (just without litellm noise).
    _litellm_devnull = open(os.devnull, "w")
    sys.stderr = _litellm_devnull

# ── LLM interface ────────────────────────────────────────────────────────────

def _call_llm(
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.3,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    max_tokens: int | None = None,
) -> dict:
    """Thin wrapper around litellm. Returns {'content': str, 'tool_calls': [...]}."""
    import litellm

    # Quiet litellm (redundant with module-level, but belt-and-suspenders)
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    # Resolve provider preset → overrides base_url / key if applicable
    resolved_model, preset_base, preset_key = _resolve_provider(model)
    effective_base = api_base or preset_base
    effective_key = api_key or preset_key

    if effective_base:
        litellm.api_base = effective_base
    if effective_key:
        litellm.api_key = effective_key

    kwargs = dict(
        model=resolved_model,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=temperature,
        max_tokens=max_tokens or (2048 if tools else 4096),
        timeout=timeout,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # Hard timeout via thread — litellm's timeout is unreliable for some providers
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(litellm.completion, **kwargs)
        try:
            response = future.result(timeout=timeout + 10)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"No response from {resolved_model} after {timeout + 10}s. "
                f"Is the provider running? For Ollama, check: ollama ps"
            )
    choice = response.choices[0].message

    result = {"content": choice.content or ""}
    if choice.tool_calls:
        import json
        result["tool_calls"] = [
            {
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments,
            }
            for tc in choice.tool_calls
        ]
    return result


# ── Move extraction ──────────────────────────────────────────────────────────

_MOVE_RE = re.compile(
    r"""
    (?:MOVE|move)\s*:\s*            # "MOVE: " or "move: " prefix
    ([a-h][1-8][a-h][1-8][qrbn]?)   # uci: e2e4, e7e8q, g1f3
    """,
    re.VERBOSE | re.IGNORECASE,
)

_RAW_UCI_RE = re.compile(
    r"""
    \b([a-h][1-8][a-h][1-8][qrbn]?)\b
    """,
    re.VERBOSE,
)


def _extract_move_uci(text: str, board: chess.Board) -> Optional[str]:
    """Extract a UCI move from free-text LLM output. Returns None if none found."""
    # First, try the explicit MOVE: marker
    m = _MOVE_RE.search(text)
    if m:
        return m.group(1)

    # Fallback: find any UCI-like token that is legal
    candidates = _RAW_UCI_RE.findall(text)
    for uci in candidates:
        try:
            move = board.parse_uci(uci)
            if move in board.legal_moves:
                return uci
        except ValueError:
            continue
    return None


def _explain_illegal(board: chess.Board, uci: str) -> str:
    """Give a human-readable explanation of why a UCI move is illegal."""
    try:
        move = board.parse_uci(uci)
    except ValueError:
        return (
            f"'{uci}' is not valid UCI notation. "
            f"Use format like e2e4, g1f3, e7e8q (for promotion)."
        )

    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    # Check if from-square has a piece of the right color
    piece = board.piece_at(move.from_square)
    if piece is None:
        return f"No piece on {from_sq}."
    if piece.color != board.turn:
        return f"The piece on {from_sq} is not yours (it's {'white' if piece.color else 'black'})."

    # Check if move is pseudo-legal first
    board_copy = board.copy()
    if board_copy.is_pseudo_legal(move):
        # Pseudo-legal but not fully legal — must leave king in check
        board_copy.push(move)
        return f"After {uci}, your king would be in check."
    else:
        # Not even pseudo-legal — wrong piece movement or blocked path
        target = board.piece_at(move.to_square)
        if target and target.color == piece.color:
            return f"{to_sq} is occupied by your own {target.symbol()}."
        if piece.piece_type == chess.PAWN:
            # Pawn-specific diagnostics
            if chess.square_file(move.from_square) == chess.square_file(move.to_square):
                # Straight push
                if target:
                    return f"{to_sq} is occupied. Pawns can't capture straight ahead."
                # Check if blocked
                intermediate = chess.square(
                    chess.square_file(move.from_square),
                    (chess.square_rank(move.from_square) + chess.square_rank(move.to_square)) // 2
                    if abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2
                    else -1,
                )
                if intermediate >= 0 and board.piece_at(intermediate):
                    return f"The pawn's path is blocked by a piece on {chess.square_name(intermediate)}."
            else:
                if not target:
                    return f"{to_sq} is empty. Pawns can only capture diagonally."
        return f"The {piece.symbol()} on {from_sq} cannot move to {to_sq}."


# ── Player classes ───────────────────────────────────────────────────────────

class LLMPlayer:
    """A chess player backed by any LLM through litellm."""

    def __init__(
        self,
        model: str,
        name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        use_tools: bool = True,
        temperature: float = 0.3,
        timeout: int = 120,
    ):
        self.model = model
        self.name = name or model
        self.api_base = api_base or os.environ.get("LLM_CHESS_API_BASE")
        self.api_key = api_key or os.environ.get("LLM_CHESS_API_KEY")
        self.max_retries = max_retries
        self.use_tools = use_tools
        self.temperature = temperature
        self.timeout = timeout
        self.illegal_count = 0  # stats

    def check_connectivity(self) -> bool:
        """Quick pre-flight check that the model / API is reachable.
        Returns True on success, raises on failure with a helpful message."""
        _, preset_base, preset_key = _resolve_provider(self.model)
        base = self.api_base or preset_base
        key = self.api_key or preset_key

        if base:
            print(f"   🔗 {self.name}: connecting to {base} ...", flush=True)

        # A tiny request just to validate connectivity
        try:
            _call_llm(
                model=self.model,
                system="Respond with exactly the word 'ok'.",
                messages=[],
                temperature=0,
                max_tokens=5,
                timeout=max(15, self.timeout // 4),
                api_base=self.api_base,
                api_key=self.api_key,
            )
        except TimeoutError:
            if base:
                raise ConnectionError(
                    f"Cannot reach {base} for {self.name}. "
                    f"Is the API running? For Ollama: 'ollama serve'"
                )
            raise
        except Exception as e:
            # 401, 404, etc. — report but don't block (might work later)
            print(f"   ⚠ {self.name}: pre-flight warning: {e}")
            return False

        print(f"   ✅ {self.name}: connected", flush=True)
        return True

    def _build_system(self, board: chess.Board, history_text: str) -> str:
        color = "White" if board.turn == chess.WHITE else "Black"
        legal_moves = [board.san(m) for m in board.legal_moves]
        capped = legal_moves[:60]
        suffix = f" ... and {len(legal_moves) - 60} more" if len(legal_moves) > 60 else ""
        return textwrap.dedent(f"""\
        You are playing chess as **{color}**. You are a strong chess player.
        Think carefully about the position, then submit your move.

        ## Current board (FEN)
        {board.fen()}

        ## Legal moves
        {', '.join(capped)}{suffix}

        ## Game history (PGN summary)
        {history_text if history_text else "(opening — first move)"}

        ## Instructions
        1. Reason about the position step by step (material, king safety, piece activity, tactics).
        2. Pick the best legal move.
        3. Output ONLY your move using the make_move tool.""")

    _TOOLS = [{
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Submit your chosen chess move in UCI notation (e.g. e2e4, g1f3, e7e8q for promotion).",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "description": "The move in UCI notation, e.g. e2e4, g1f3, e7e8q",
                    }
                },
                "required": ["move"],
            },
        },
    }]

    def get_move(self, board: chess.Board, history_text: str) -> chess.Move:
        """Get a legal move from the LLM, with retries for illegal moves."""
        messages = []
        system = self._build_system(board, history_text)

        for attempt in range(1, self.max_retries + 1):
            was_retry = attempt > 1
            if DEBUG:
                label = f"retry {attempt}/{self.max_retries}" if was_retry else "thinking"
                print(f"  🤔 {self.name}: {label}...", flush=True)
            try:
                result = _call_llm(
                    model=self.model,
                    system=system,
                    messages=messages,
                    tools=self._TOOLS if self.use_tools else None,
                    temperature=self.temperature,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as e:
                # Network error, API error, rate limit, etc.
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"  ⚠ {self.name}: API error ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                # Exhausted retries — forfeit gracefully instead of crashing
                raise IllegalMoveForfeit(
                    f"{self.name} hit persistent API errors: {e}"
                ) from e

            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            # Debug: show what the model returned
            if DEBUG:
                print(f"  🔍 {self.name} response (attempt {attempt}):")
                if tool_calls:
                    print(f"     tool_calls: {tool_calls}")
                if content:
                    # Show start and end — the move usually comes at the end
                    if len(content) > 400:
                        print(f"     content (start): {content[:200]}")
                        print(f"     content (end):   ...{content[-200:]}")
                    else:
                        print(f"     content: {content}")
                if not content and not tool_calls:
                    print(f"     (empty response — API returned nothing)")

            # Treat completely empty responses as transient API errors
            # (the model didn't refuse — it just returned nothing at all)
            if not content and not tool_calls:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    hint = ""
                    if self.use_tools and attempt == 1:
                        hint = " (model may not support tools — try --no-tools)"
                    print(f"  ⚠ {self.name}: empty API response, retrying in {wait}s...{hint}")
                    time.sleep(wait)
                    # Auto-disable tools for retry 2+ if model returned nothing
                    if attempt >= 2 and self.use_tools:
                        self.use_tools = False
                    continue
                # Final attempt also empty — give up gracefully
                tip = " Try --no-tools if the model doesn't support tool calling." if self.use_tools else ""
                print(f"  ❌ {self.name}: API returned empty responses for all {self.max_retries} attempts.")
                raise IllegalMoveForfeit(
                    f"{self.name} got empty responses from the API "
                    f"{self.max_retries} times in a row.{tip}"
                )

            # Extract UCI
            proposed_uci = None
            used_tools = bool(tool_calls)
            if tool_calls:
                tc = tool_calls[0]
                if tc["name"] == "make_move":
                    proposed_uci = tc["arguments"].get("move", "").strip().lower()
            else:
                proposed_uci = _extract_move_uci(content, board)

            if not proposed_uci:
                # Escalating urgency — by attempt 3 we just demand the UCI
                legal_san = [board.san(m) for m in board.legal_moves][:20]

                if attempt == 1:
                    hint = ""
                    if self.use_tools and not used_tools:
                        hint = " This model may not support tool calling — try --no-tools."
                    error_msg = (
                        f"You did not provide a legal move.{hint}\n"
                        f"Pick one of these: {', '.join(legal_san)}.\n"
                        f"Output a UCI move like e2e4 or g1f3."
                    )
                elif attempt == 2:
                    error_msg = (
                        f"STOP REASONING. Output ONLY a UCI move. No explanation.\n"
                        f"Pick one: {', '.join(legal_san[:10])}"
                    )
                else:
                    error_msg = (
                        f"JUST THE UCI. ONE STRING. Example: {legal_san[0] if legal_san else 'e2e4'}"
                    )

                messages.append({"role": "user", "content": error_msg})
                # Auto-disable tools for subsequent attempts if model ignored them
                if attempt >= 2 and self.use_tools and not used_tools:
                    self.use_tools = False
                continue

            # Validate
            try:
                move = board.parse_uci(proposed_uci)
            except ValueError:
                error_msg = (
                    f"'{proposed_uci}' is not valid UCI notation. "
                    f"Use format like e2e4, g1f3, or e7e8q (promotion). "
                    f"Try again."
                )
                messages.append({"role": "user", "content": error_msg})
                continue

            if move in board.legal_moves:
                return move
            else:
                self.illegal_count += 1
                explanation = _explain_illegal(board, proposed_uci)
                legal_san = [board.san(m) for m in board.legal_moves][:30]
                error_msg = (
                    f"Illegal move: {explanation}\n"
                    f"Some legal moves: {', '.join(legal_san)}.\n"
                    f"Please pick a legal move."
                )
                messages.append({"role": "user", "content": error_msg})

        # Exhausted retries
        raise IllegalMoveForfeit(
            f"{self.name} failed to produce a legal move after "
            f"{self.max_retries} attempts. Forfeiting."
        )


class HumanPlayer:
    """A human player via terminal input."""

    def __init__(self, name: str = "Human"):
        self.name = name
        self.illegal_count = 0

    def get_move(self, board: chess.Board, history_text: str) -> chess.Move:
        legal_san = [board.san(m) for m in board.legal_moves]

        while True:
            try:
                raw = input(f"\n  Your move ({board.san(board.move_stack[-1]) if board.move_stack else 'first move'}): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                sys.exit(0)

            # Try UCI first, then SAN
            try:
                move = board.parse_uci(raw.lower())
                if move in board.legal_moves:
                    return move
            except ValueError:
                pass

            try:
                move = board.parse_san(raw)
                if move in board.legal_moves:
                    return move
            except ValueError:
                pass

            self.illegal_count += 1
            print(f"  ❌ '{raw}' is not a legal move.")
            print(f"  Legal moves: {', '.join(legal_san[:20])}"
                  f"{'...' if len(legal_san) > 20 else ''}")


class RandomPlayer:
    """Plays a random legal move — useful as a baseline."""

    def __init__(self, name: str = "Random"):
        self.name = name
        self.illegal_count = 0

    def get_move(self, board: chess.Board, history_text: str) -> chess.Move:
        import random
        return random.choice(list(board.legal_moves))


class StockfishPlayer:
    """Plays using the Stockfish chess engine."""

    # Common install locations checked in order when no path is given
    _KNOWN_PATHS = [
        "stockfish",                                    # in PATH
        "/usr/games/stockfish",                         # Debian/Ubuntu apt
        "/usr/bin/stockfish",                           # some distros
        "/usr/local/bin/stockfish",                     # manual install
        os.path.expanduser("~/.local/bin/stockfish"),   # pip/user install
    ]

    def __init__(
        self,
        name: str = "Stockfish",
        skill_level: int = 20,
        think_time: float = 0.1,
        binary_path: str | None = None,
    ):
        self.name = name
        self.skill_level = skill_level
        self.think_time = think_time
        self.binary_path = binary_path or self._find_binary()
        self.illegal_count = 0
        self._engine = None

    @classmethod
    def _find_binary(cls) -> str:
        """Find the stockfish binary, trying PATH and known locations."""
        import shutil
        for path in cls._KNOWN_PATHS:
            if shutil.which(path):
                return path
        raise FileNotFoundError(
            "stockfish not found. Install it (apt install stockfish) or pass --stockfish-path"
        )

    def _get_engine(self):
        if self._engine is None:
            import chess.engine
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self.binary_path,
                timeout=30.0,  # generous startup timeout for slow systems
            )
            self._engine.configure({"Skill Level": self.skill_level})
        return self._engine

    def get_move(self, board: chess.Board, history_text: str) -> chess.Move:
        limit = chess.engine.Limit(time=self.think_time)
        result = self._get_engine().play(board, limit)
        return result.move

    def close(self):
        if self._engine:
            self._engine.quit()
            self._engine = None


# ── Exceptions ───────────────────────────────────────────────────────────────

class IllegalMoveForfeit(Exception):
    """Raised when a player exhausts retries for illegal moves."""
    pass


# ── Board rendering ──────────────────────────────────────────────────────────

def render_board(board: chess.Board, last_move: Optional[chess.Move] = None) -> str:
    """Render the board as a compact text diagram with colored pieces."""
    # Unicode chess symbols
    PIECES = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    }

    lines = []
    lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    for rank in range(7, -1, -1):
        row = f"{rank + 1} │"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            bg = ""
            if last_move and sq in (last_move.from_square, last_move.to_square):
                bg = "\033[43m"  # yellow highlight
            symbol = PIECES[piece.symbol()] if piece else ' '
            reset = "\033[0m" if bg else ""
            row += f" {bg}{symbol}{reset} │"
        lines.append(row)
        if rank > 0:
            lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("    a   b   c   d   e   f   g   h")
    return "\n".join(lines)


# ── Match orchestrator ───────────────────────────────────────────────────────

class ChessMatch:
    """Orchestrates a game between two players."""

    def __init__(
        self,
        white,
        black,
        delay: float = 1.0,
        event: str = "LLM Chess Match",
        round_name: str = "1",
    ):
        self.white = white
        self.black = black
        self.delay = delay  # seconds between moves for readability
        self.board = chess.Board()
        self.pgn_game = chess.pgn.Game()
        self.pgn_game.headers["Event"] = event
        self.pgn_game.headers["Site"] = "LLM Chess CLI"
        self.pgn_game.headers["Date"] = datetime.date.today().isoformat()
        self.pgn_game.headers["Round"] = round_name
        self.pgn_game.headers["White"] = white.name
        self.pgn_game.headers["Black"] = black.name
        self.node = self.pgn_game

    def _history_text(self) -> str:
        """Return a compact PGN summary of moves so far."""
        if not self.board.move_stack:
            return ""
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        return self.pgn_game.accept(exporter)

    def _cleanup(self):
        """Close any engine-based players."""
        for p in (self.white, self.black):
            if hasattr(p, "close"):
                p.close()

    def play(self) -> str:
        """Play the game until conclusion. Returns result string ('1-0', '0-1', '1/2-1/2')."""
        move_num = 0

        while not self.board.is_game_over(claim_draw=True):
            player = self.white if self.board.turn == chess.WHITE else self.black
            move_num += 1

            # Print board before each move
            if self.board.move_stack:
                print(f"\n--- Move {self.board.fullmove_number}"
                      f"{'.' if self.board.turn == chess.WHITE else '...'} "
                      f"({player.name}) ---")
            else:
                print(f"\n--- Game start: {self.white.name} (White) vs "
                      f"{self.black.name} (Black) ---")
            print(render_board(self.board, self.board.move_stack[-1] if self.board.move_stack else None))

            # Get move
            try:
                move = player.get_move(self.board, self._history_text())
            except IllegalMoveForfeit as e:
                print(f"\n🏳 {e}")
                self._cleanup()
                if self.board.turn == chess.WHITE:
                    self.pgn_game.headers["Result"] = "0-1"
                    return "0-1"
                else:
                    self.pgn_game.headers["Result"] = "1-0"
                    return "1-0"

            san = self.board.san(move)
            print(f"  ▶ {player.name}: {san} ({move.uci()})")

            # Push move to board and PGN
            self.board.push(move)
            self.node = self.node.add_variation(move)

            if self.delay > 0:
                time.sleep(self.delay)

        # Game over
        self._cleanup()
        print(f"\n{'=' * 40}")
        print(render_board(self.board, self.board.move_stack[-1] if self.board.move_stack else None))
        result = self.board.result(claim_draw=True)
        final_result = result if result != "*" else "1/2-1/2"
        outcome_map = {
            "1-0": f"{self.white.name} (White) wins!",
            "0-1": f"{self.black.name} (Black) wins!",
            "1/2-1/2": "Draw!",
        }
        print(f"\n🏁 Game over: {outcome_map.get(final_result, final_result)}")

        # Termination reason
        if self.board.is_checkmate():
            print("   Checkmate.")
        elif self.board.is_stalemate():
            print("   Stalemate.")
        elif self.board.is_insufficient_material():
            print("   Draw by insufficient material.")
        elif self.board.is_fifty_moves():
            print("   Draw by fifty-move rule.")
        elif self.board.is_repetition():
            print("   Draw by threefold repetition.")

        self.pgn_game.headers["Result"] = final_result
        return result

    def save_pgn(self, path: str):
        """Save the game to a PGN file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            print(self.pgn_game, file=f)
        print(f"\n📄 PGN saved to {path}")

    def print_stats(self):
        """Print per-player stats."""
        for p in (self.white, self.black):
            if hasattr(p, "illegal_count"):
                print(f"   {p.name}: {p.illegal_count} illegal move(s)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_player(spec: str) -> tuple[str, dict]:
    """
    Parse a player spec string into (type, kwargs).

    Types:
      human          — HumanPlayer
      random         — RandomPlayer
      model_name     — LLMPlayer with that model
      ollama/model   — LLMPlayer with Ollama
      openai/gpt-4o  — LLMPlayer with litellm prefix
    """
    spec = spec.strip()

    if spec.lower() == "human":
        return "human", {"name": "Human"}

    if spec.lower() == "random":
        return "random", {"name": "Random"}

    if spec.lower() == "stockfish":
        return "stockfish", {"name": "Stockfish"}

    # Otherwise, treat as an LLM model string
    return "llm", {"model": spec, "name": spec}


def main():
    parser = argparse.ArgumentParser(
        description="LLM Chess — pit LLMs (or humans) against each other",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Environment variables:
          OPENCODE_GO_API_KEY   API key for opencode-go/ provider
          OPENCODE_ZEN_API_KEY  API key for opencode-zen/ provider
          OPENROUTER_API_KEY    API key for openrouter/ provider
          OPENAI_API_KEY        API key for openai/ provider
          LLM_CHESS_DEBUG=1     Show model responses (debug illegal moves)

        Examples:
          python chess_llm.py
          python chess_llm.py --white human
          python chess_llm.py --white openai/gpt-4o --black anthropic/claude-sonnet-4
          python chess_llm.py --model groq/llama-3.3-70b  # both sides
          python chess_llm.py --delay 0  # no pause between moves
        """),
    )
    parser.add_argument(
        "--white", "-w", default="opencode-go/deepseek-v4-pro",
        help="White player: 'human', 'random', 'stockfish', or model name "
             "(default: opencode-go/deepseek-v4-pro)",
    )
    parser.add_argument(
        "--black", "-b", default="opencode-go/deepseek-v4-pro",
        help="Black player: 'human', 'random', 'stockfish', or model name "
             "(default: opencode-go/deepseek-v4-pro)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Use same model for both white and black (overrides --white/--black)",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=1.0,
        help="Delay between moves in seconds (default: 1.0, use 0 for max speed)",
    )
    parser.add_argument(
        "--pgn", "-p", default=None,
        help="PGN output file (default: auto-generated in ./games/)",
    )
    parser.add_argument(
        "--no-tools", action="store_true",
        help="Disable tool calling — use text parsing instead (for models that don't support tools)",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    parser.add_argument(
        "--retries", "-r", type=int, default=3,
        help="Max illegal-move retries before forfeit (default: 3)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="API timeout in seconds (default: 120, increase for slow local models)",
    )
    parser.add_argument(
        "--stockfish-skill", type=int, default=20,
        help="Stockfish skill level 0-20 (default: 20)",
    )
    parser.add_argument(
        "--stockfish-time", type=float, default=0.1,
        help="Stockfish think time in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--stockfish-path", default=None,
        help="Path to stockfish binary (default: auto-detect from PATH or common locations)",
    )
    parser.add_argument(
        "--api-base",
        help="Custom API base URL (or set LLM_CHESS_API_BASE env var)",
    )
    parser.add_argument(
        "--api-key",
        help="Custom API key (or set LLM_CHESS_API_KEY env var)",
    )
    parser.add_argument(
        "--log", "-l", nargs="?", const="auto", default=None,
        help="Tee output to a log file in .logs/ (auto-names by date; or give a path)",
    )

    args = parser.parse_args()

    # ── Tee output to log file ──────────────────────────────────────────
    _log_file = None
    if args.log:
        if args.log == "auto":
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".logs")
            os.makedirs(log_dir, exist_ok=True)
            args.log = os.path.join(log_dir, f"game_{ts}.txt")
        _log_file = open(args.log, "w", buffering=1)  # line-buffered
        _original_stdout = sys.stdout
        class _Tee:
            def write(self, data):
                _original_stdout.write(data)
                _log_file.write(data)
            def flush(self):
                _original_stdout.flush()
                _log_file.flush()
        class _TeeStderr:
            def write(self, data):
                _REAL_STDERR.write(data)
                _log_file.write(data)
            def flush(self):
                _REAL_STDERR.flush()
                _log_file.flush()
        sys.stdout = _Tee()
        sys.stderr = _TeeStderr()
        print(f"📝 Logging to {args.log}")

    if args.model:
        args.white = args.black = args.model

    # Build players
    def build_player(spec: str, color: str):
        ptype, kwargs = parse_player(spec)
        if ptype == "human":
            return HumanPlayer(name=f"Human ({color})")
        elif ptype == "random":
            return RandomPlayer(name=f"Random ({color})")
        elif ptype == "stockfish":
            return StockfishPlayer(
                name=f"Stockfish {args.stockfish_skill} ({color})",
                skill_level=args.stockfish_skill,
                think_time=args.stockfish_time,
                binary_path=args.stockfish_path,
            )
        else:
            name = f"{kwargs['model']} ({color})"
            return LLMPlayer(
                model=kwargs["model"],
                name=name,
                api_base=args.api_base,
                api_key=args.api_key,
                max_retries=args.retries,
                use_tools=not args.no_tools,
                temperature=args.temperature,
                timeout=args.timeout,
            )

    white = build_player(args.white, "White")
    black = build_player(args.black, "Black")

    # Pre-flight connectivity checks for LLM players
    for p in (white, black):
        if hasattr(p, "check_connectivity"):
            p.check_connectivity()

    print(f"\n♟  LLM Chess Match")
    print(f"   White: {white.name}")
    print(f"   Black: {black.name}")
    print(f"   Delay: {args.delay}s  |  Retries: {args.retries}  |  "
          f"Tools: {'on' if not args.no_tools else 'off'}")

    match = ChessMatch(white, black, delay=args.delay)
    result = match.play()

    # Save PGN
    pgn_path = args.pgn
    if not pgn_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        w_name = white.name.replace("/", "_").replace(" ", "_")
        b_name = black.name.replace("/", "_").replace(" ", "_")
        pgn_path = f"games/{ts}_{w_name}_vs_{b_name}.pgn"
    match.save_pgn(pgn_path)
    match.print_stats()

    return result


if __name__ == "__main__":
    main()
