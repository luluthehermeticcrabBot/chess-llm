# ♟ LLM Chess

Pit language models against each other — or against humans and Stockfish — in a game of chess. Every game auto-saves as a standard PGN file.

## Quickstart

```bash
# Install
uv pip install -e .

# Set API keys for your providers
export OPENCODE_GO_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-v1-...

# Play!
python chess_llm.py --white opencode-go/deepseek-v4-pro --black stockfish
```

## Players

| Spec | Description |
|---|---|
| `opencode-go/model` | OpenCode Go API (`$OPENCODE_GO_API_KEY`) |
| `opencode-zen/model` | OpenCode Zen API (`$OPENCODE_ZEN_API_KEY`) |
| `openrouter/model` | OpenRouter API (`$OPENROUTER_API_KEY`) |
| `openai/model` | OpenAI API (`$OPENAI_API_KEY`) |
| `anthropic/model` | Anthropic API (`$ANTHROPIC_API_KEY`) |
| `groq/model` | Groq API (`$GROQ_API_KEY`) |
| `ollama/model` | Local Ollama (no key needed) |
| `stockfish` | Stockfish 18 chess engine |
| `human` | Terminal input (SAN or UCI notation) |
| `random` | Random legal moves (baseline) |

## Usage

```bash
# LLM vs LLM
python chess_llm.py --white opencode-go/deepseek-v4-pro --black openrouter/anthropic/claude-sonnet-4

# Same model both sides
python chess_llm.py --model opencode-go/deepseek-v4-pro

# Human vs Stockfish
python chess_llm.py --white human --black stockfish

# LLM vs Stockfish (benchmark!)
python chess_llm.py --white opencode-go/kimi-k2.5 --black stockfish

# Speed mode — no delay between moves
python chess_llm.py --model opencode-go/deepseek-v4-pro --delay 0

# Debug mode — see model responses
LLM_CHESS_DEBUG=1 python chess_llm.py --model opencode-go/deepseek-v4-pro

# Auto-log to file
python chess_llm.py --model opencode-go/deepseek-v4-pro --log
```

All games auto-save as PGN in `./games/` — load them in Lichess or Chess.com for analysis.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--white`, `-w` | `opencode-go/deepseek-v4-pro` | White player |
| `--black`, `-b` | `opencode-go/deepseek-v4-pro` | Black player |
| `--model`, `-m` | — | Use same model for both sides |
| `--delay`, `-d` | `1.0` | Seconds between moves (`0` = max speed) |
| `--no-tools` | off | Disable tool calling, use text parsing instead |
| `--retries`, `-r` | `3` | Illegal-move retries before forfeit |
| `--timeout` | `120` | API timeout in seconds |
| `--temperature`, `-t` | `0.3` | LLM temperature |
| `--log`, `-l` | off | Tee output to `.logs/` (auto-names by date) |
| `--pgn`, `-p` | auto | PGN output path |

**Stockfish flags:** `--stockfish-skill 0-20`, `--stockfish-time 0.1`, `--stockfish-path`

**API flags:** `--api-base`, `--api-key` (override provider defaults)

## Environment

| Variable | Used by |
|---|---|
| `OPENCODE_GO_API_KEY` | `opencode-go/` provider |
| `OPENCODE_ZEN_API_KEY` | `opencode-zen/` provider |
| `OPENROUTER_API_KEY` | `openrouter/` provider |
| `OPENAI_API_KEY` | `openai/` provider |
| `ANTHROPIC_API_KEY` | `anthropic/` provider |
| `GROQ_API_KEY` | `groq/` provider |
| `LLM_CHESS_DEBUG` | Set to `1` for verbose model output |

## How it works

**Move validation pipeline:**

```
LLM thinks → calls make_move(uci) → python-chess validates:
  ✓ Legal → push move, next turn
  ✗ Illegal → "That move is illegal because… Legal options: …"
              → retry (up to 3x) → forfeit if still no legal move
```

- **Tool calling** (default): models use a `make_move` tool — clean, reliable extraction
- **Text fallback**: parses UCI tokens from free text — useful for models without tool support
- **Empty response handling**: transient API hiccups (empty completions) get exponential backoff retries
- **Hard timeout**: all API calls run in a thread with a timeout — no more hangs on unresponsive Ollama

**Pre-flight check:** before the match starts, each LLM player sends a tiny probe to validate connectivity.

## Project files

```
chess-llm/
├── chess_llm.py       # The engine (single file, ~950 lines)
├── pyproject.toml     # uv / pip project config
├── requirements.txt   # pip fallback
├── games/             # Auto-saved PGN files
├── .logs/             # Logs (--log flag)
└── .hermes/           # Hermes agent plans
```
