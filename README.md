# Hothand Analysis

A small exploratory script for collecting NCAA Division I mens basketball play-by-play data and extracting per-player shot sequences to investigate "hot hand" patterns.

**Quick Start**

- Requirements: Python 3.8+ (uses only stdlib modules).
- Run: 

```bash
python3 hothand.py
```

**Files**

- [hothand.py](hothand.py) — main script that downloads scoreboard and play-by-play JSON, parses events, and summarizes player shot sequences.

**What this project does (high level)**

- Fetches daily scoreboards from an NCAA API and filters games by conference.
- Downloads play-by-play JSON for each selected game.
- Parses event descriptions to build per-player shot sequences (two-point and three-point makes/misses).
- Produces simple console summaries useful for exploratory analysis of hot-hand patterns.

**How `hothand.py` works (key functions)**

- `get_game_ids()`
  - Opens a single HTTPS connection to the configured API host.
  - Iterates date-by-date between `SEASON_START` and `SEASON_END`.
  - Requests the scoreboard endpoint for each date and parses JSON.
  - Filters games by `conferenceSeo` (currently looks for `big-12`).
  - Accumulates `gameID`s and returns the list.

- `get_pbp_data()`
  - Calls `get_game_ids()` and then requests `/game/{gameID}/play-by-play` for each ID.
  - Parses play-by-play JSON and returns a list of per-game payloads.

- `get_all_player_stats_of_game(pbp_data)`
  - Walks `pbp_data['periods']` and each period's `playbyplayStats`.
  - Extracts a player label (from `firstName`/`lastName` fields when present, otherwise heuristics on `eventDescription`).
  - Filters for shot events and records sequences as `two_point_sequence` and `three_point_sequence` (0 = make, 1 = miss).
  - Uses simple deduplication heuristics (`skip_counter`) because some events are duplicated in the payload.
  - Returns a mapping player -> shot-sequence dict.

- `main()`
  - Orchestrates downloading pbp data, calling the parser for each game, collecting summaries, and printing compact console reports.

**Configuration and constants**

- `API_HOST` — the API host used by the script.
- `SEASON_START`, `SEASON_END` — date range scanned for scoreboards.
- `MAX_TEST_GAMES` — caps the number of games to download for fast testing.

These values are defined near the top of `hothand.py` and are safe to change for different experiments.

**Assumptions & limitations**

- The parser relies on payload field names and human-readable `eventDescription` text. This is brittle and may break if the API changes wording or structure.
- The conference filter is hardcoded (`big-12`). Change or parameterize it to analyze other conferences.
- No retries, rate-limiting, or robust HTTP error handling are implemented.
- The script prints to console; it does not persist results to disk (but it's easy to add CSV/JSON export).

**Suggestions for improvement**

- Replace `http.client` with `requests` for simpler HTTP handling and built-in timeouts/retries.
- Add structured logging via the `logging` module instead of `print`.
- Improve name/entity extraction (use regexes or map roster IDs where available).
- Add persistence (CSV/JSON) and unit tests for the parsing logic.
- Parameterize conference/date range and add command-line flags.
