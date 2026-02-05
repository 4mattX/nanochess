# Chess Data - Quick Start

Get high-quality chess training data in 3 steps:

## 1. Fetch Data

### Easiest (Interactive Menu)
```bash
bash dev/quick_fetch.sh
```

### Common Commands

**Note:** The script now automatically uses game outcomes (win/draw/loss) as value targets when you don't request position-specific evaluations. This trains both policy and value heads!

```bash
# Quick test (1K positions, uses game outcomes for value head)
python dev/fetch_chess_data.py --source lichess-elite --max-games 500 --max-positions 1000 --time-control rapid

# Small dataset (50K positions)
python dev/fetch_chess_data.py --source lichess-elite --max-games 2000 --max-positions 50000 --time-control rapid

# Production dataset (500K positions)
python dev/fetch_chess_data.py --source lichess-elite --max-games 10000 --max-positions 500000 --time-control rapid

# Magnus Carlsen only
python dev/fetch_chess_data.py --source lichess-user --username DrNykterstein --max-games 5000
```

## 2. Inspect Data

```bash
python dev/inspect_data.py
```

Shows:
- Number of positions
- Eval coverage
- Move distribution
- Quality checks
- Sample positions

## 3. Train

```bash
# Using TUI
python -m nanochess

# Or directly
python -m scripts.pra_train --depth=8 --n-embd=192 --device-batch-size=256
```

## Pro Tips

1. **Start small** - Test with 1K positions first
2. **Use rapid/blitz** - More games available, still high quality at 2500+ level
3. **Game outcomes work great** - Win/loss/draw signals train the value head effectively
4. **Check data quality** - Run `inspect_data.py` before training
5. **Mix sources** - Combine elite + specific players

## Value Head Training

Two approaches:

**Game Outcomes (Recommended - Fast & Easy)**
- Uses win/loss/draw results from completed games
- Automatically enabled when you don't use `--with-eval`
- Works with any game database
- This is how AlphaZero and modern engines train!

**Position Evaluations (Slower - Needs Analysis)**
- Uses Stockfish evaluations for each position
- Add `--with-eval` flag (requires pre-analyzed games)
- More precise but analyzed games are rare on Lichess API
- Better: use `--source lichess-db` for pre-analyzed database dumps

## Common Players

| Username | Player | Rating |
|----------|--------|--------|
| `DrNykterstein` | Magnus Carlsen | 2800+ |
| `Hikaru` | Hikaru Nakamura | 2750+ |
| `FabianoCaruana` | Fabiano Caruana | 2750+ |
| `DanielNaroditsky` | Daniel Naroditsky | 2600+ |
| `gmwesleysoso` | Wesley So | 2750+ |

## Troubleshooting

### No data fetched?
- Check internet connection
- Try different username
- Wait 1 min (rate limiting)

### Out of memory?
- Use `--sample-rate 0.5` (50% of positions)
- Use `--max-positions 100000`
- Process smaller batches

### Missing dependencies?
```bash
pip install requests chess pyarrow
```

## Full Documentation

See [DATA_FETCHING.md](DATA_FETCHING.md) for complete guide.
