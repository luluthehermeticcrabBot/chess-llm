"""Chess board widget for Textual TUI."""

from typing import Optional

import chess
from textual.widget import Widget
from textual.strip import Strip

from .common import (
    PIECE_MAP, LIGHT_SQUARE, DARK_SQUARE, SELECTED_SQUARE,
    LAST_MOVE_LIGHT, LAST_MOVE_DARK, CURSOR_BORDER, LEGAL_MOVE_DOT,
    BG, TEXT_PRIMARY,
)


class BoardWidget(Widget):
    """Renders an 8×8 chess board with Unicode pieces."""

    DEFAULT_CSS = """
    BoardWidget {
        width: 23;
        height: 10;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        board: chess.Board | None = None,
        flipped: bool = False,
        selected_square: Optional[int] = None,
        legal_moves: Optional[set[int]] = None,
        last_move: Optional[chess.Move] = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.board = board or chess.Board()
        self.flipped = flipped
        self.selected_square = selected_square
        self.legal_moves = legal_moves or set()
        self.last_move = last_move

    def render_line(self, y: int) -> Strip:
        """Render one line of the board widget."""
        if y >= 10:
            return Strip.blank(23)

        if y == 9:
            # Column labels: "   a  b  c  d  e  f  g  h"
            cols = "abcdefgh"
            if self.flipped:
                cols = cols[::-1]
            text = "    " + "  ".join(cols)
            return Strip(text).style(f"bold {TEXT_PRIMARY.hex}")

        # Board rows (y=0 → rank 8, y=7 → rank 1 when not flipped)
        rank_idx = y if not self.flipped else 7 - y
        rank = 8 - rank_idx

        parts = [f" {rank} "]  # rank label with padding

        for file_idx in range(8):
            col = file_idx if not self.flipped else 7 - file_idx
            square = chess.square(col, rank_idx)
            piece = self.board.piece_at(square)

            # Square styling
            is_light = (rank_idx + col) % 2 == 0
            base_color = LIGHT_SQUARE if is_light else DARK_SQUARE

            # Last move highlight
            if self.last_move and square in (self.last_move.from_square,
                                              self.last_move.to_square):
                base_color = LAST_MOVE_LIGHT if is_light else LAST_MOVE_DARK

            # Selected square
            if square == self.selected_square:
                base_color = SELECTED_SQUARE

            # Piece or empty
            if piece:
                symbol = PIECE_MAP[piece.symbol()]
                char = f" {symbol} "
            elif square in self.legal_moves:
                char = " · "  # legal move dot
            else:
                char = "   "

            bg_hex = base_color.hex
            fg = "#000000" if is_light or square == self.selected_square else "#ffffff"

            # Cursor border on selected square
            if square == self.selected_square:
                parts.append(f"[{fg} on {CURSOR_BORDER.hex}]{char}[/]")
            else:
                parts.append(f"[{fg} on {bg_hex}]{char}[/]")

        return Strip("".join(parts))

    def update_state(
        self,
        board: chess.Board | None = None,
        selected_square: Optional[int] = None,
        legal_moves: Optional[set[int]] = None,
        last_move: Optional[chess.Move] = None,
    ):
        """Update the board state and refresh."""
        if board is not None:
            self.board = board
        self.selected_square = selected_square
        self.legal_moves = legal_moves or set()
        if last_move is not None:
            self.last_move = last_move
        self.refresh()
