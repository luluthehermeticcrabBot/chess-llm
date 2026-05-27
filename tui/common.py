"""Shared styles, colors, and constants for the chess TUI."""

from textual.color import Color


# ── Board colors ──────────────────────────────────────────────────────────
LIGHT_SQUARE = Color.parse("#f0d9b5")
DARK_SQUARE = Color.parse("#b58863")
SELECTED_SQUARE = Color.parse("#829769")       # green highlight
LEGAL_MOVE_DOT = Color.parse("#829769")        # darker green for move targets
LAST_MOVE_LIGHT = Color.parse("#cdd26a")       # yellow tint
LAST_MOVE_DARK = Color.parse("#aaa23a")
CURSOR_BORDER = Color.parse("#ffff00")         # bright yellow cursor

# ── Theme colors ──────────────────────────────────────────────────────────
BG = Color.parse("#1a1a2e")                    # dark navy background
SURFACE = Color.parse("#16213e")               # slightly lighter surface
ACCENT = Color.parse("#0f3460")                # blue accent
HIGHLIGHT = Color.parse("#e94560")             # red/pink highlight
TEXT_PRIMARY = Color.parse("#eaeaea")
TEXT_SECONDARY = Color.parse("#a0a0a0")
SUCCESS = Color.parse("#4ecca3")               # green
WARNING = Color.parse("#ffc107")               # amber
ERROR = Color.parse("#e94560")                 # red
DRAW_COLOR = Color.parse("#aaaaaa")            # grey for draws

# ── Unicode chess pieces ──────────────────────────────────────────────────
PIECE_MAP = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
}

# ── TUI CSS ───────────────────────────────────────────────────────────────
TUI_CSS = """
Screen {
    background: #1a1a2e;
}

#header {
    height: 3;
    padding: 0 1;
    background: #0f3460;
    color: #eaeaea;
    text-style: bold;
}

#progress-bar {
    height: 3;
    padding: 0 1;
}

#active-games {
    height: 6;
    padding: 0 1;
    border: solid #a0a0a0;
}

#elo-table {
    height: 12;
    padding: 0 1;
    border: solid #a0a0a0;
}

#matchup-grid {
    height: 1fr;
    padding: 0 1;
    border: solid #a0a0a0;
}

.footer {
    height: 1;
    background: #0f3460;
    color: #a0a0a0;
    text-style: italic;
}

DataTable {
    background: #16213e;
    color: #eaeaea;
}

DataTable > .datatable--header {
    background: #0f3460;
    color: #eaeaea;
    text-style: bold;
}
"""
