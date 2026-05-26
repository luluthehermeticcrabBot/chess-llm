# AGENTS.md — LLM Chess

## Project overview

Single-file Python chess engine that pits LLMs, humans, and Stockfish against each other. No web server, no database — just a CLI that prints a Unicode board to the terminal and saves PGN files.

## Architecture

```
chess_llm.py  (single file, ~950 lines)
├── Provider presets      — maps prefix→(base_url, env_var)
├── _resolve_provider()   — opencode-go/deepseek-v4 → openai/deepseek-v4 + base URL
├── _call_llm()           — litellm wrapper with thread-level timeout + stderr suppression
│
├── LLMPlayer             — LLM-backed player
│   ├── _build_system()   — constructs system prompt with FEN + legal moves
│   ├── check_connectivity() — pre-flight API probe
│   └── get_move()        — tool calling → text fallback → retry escalation → forfeit
│
├── StockfishPlayer       — python-chess.engine wrapper (keeps engine alive across moves)
├── HumanPlayer           — stdin (accepts SAN + UCI)
├── RandomPlayer          — random.choice(legal_moves)
│
├── ChessMatch            — game orchestrator (board, PGN, render, cleanup)
├── render_board()        — Unicode terminal board with last-move highlighting
└── main()                — argparse CLI + --log tee
```

## Key design decisions

- **Provider presets, not litellm config**: The `PROVIDER_PRESETS` dict at the top of the file maps short prefixes to base URLs and env var names. No config file needed — just set the right env var.
- **Hard thread timeout**: `_call_llm()` runs litellm inside `concurrent.futures.ThreadPoolExecutor` with a timeout. Litellm's own `timeout` parameter is unreliable across providers (especially Ollama). The thread timeout guarantees the call can't hang forever.
- **Retry escalation**: On illegal/no-move, error messages get progressively shorter and more forceful. Attempt 3 is a single line: `"JUST THE UCI. ONE STRING. Example: Nc3"`.
- **Auto tool→text fallback**: If tools are enabled but the model ignores them on attempt 1, tools are auto-disabled for attempts 2+.
- **Empty API response = retry, not error**: Empty completions get treated as transient failures with exponential backoff, not sent as conversation correction messages.
- **Stockfish reuse**: `StockfishPlayer` creates the engine once and reuses it across moves (not spawning per-move, which is too slow).
- **stderr suppression**: During API calls, litellm's stderr is redirected to `/dev/null` to suppress the "Provider List" spam. Restored after the call completes.

## Dependencies

- `python-chess` — board state, legal move generation, PGN, Stockfish integration
- `litellm` — multi-provider LLM interface
- `openai` — pulled by litellm
- `botocore` — pulled by litellm for AWS providers
- `stockfish` (optional) — Python wrapper for the Stockfish binary

## Running tests

```bash
# Quick smoke test (random vs random, no API needed)
python chess_llm.py --white random --black random --delay 0

# With debug output
LLM_CHESS_DEBUG=1 python chess_llm.py --white random --black random --delay 0
```

## Common issues

- **Ollama hangs**: Model name might be wrong. Use `ollama list` to check available models. The pre-flight check will catch unreachable endpoints before the game starts.
- **Stockfish timeout**: Startup can take >10s on cold systems. The default popen timeout is now 30s.
- **Model returns no move**: Small models often reason endlessly without outputting a UCI move. Try `--no-tools` or increase `--retries`.
- **Empty API responses**: Some providers occasionally return completions with no content. These are auto-retried with backoff.

## File conventions

- PGN files → `games/YYYYMMDD_HHMMSS_white_vs_black.pgn`
- Log files → `.logs/game_YYYYMMDD_HHMMSS.txt` (via `--log`)
- Hermes plans → `.hermes/` (for AI agent planning)
