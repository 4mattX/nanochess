# Chess Data Fetching Guide

Streamlined tools for acquiring high-quality chess data from top players for PRA training.

## Quick Start

The fastest way to get started:

```bash
# Interactive menu with presets
bash dev/quick_fetch.sh
```

This gives you options like:
1. **Quick test** - 1K positions in ~1 min (for testing)
2. **Small dataset** - 50K positions in ~10 min
3. **Medium dataset** - 500K positions in ~1 hour
4. **Large dataset** - 2M positions in ~3 hours
5. **Magnus Carlsen games only** - All available games from the current World Champion
6. **Lichess DB monthly** - Full monthly database (10+ GB, millions of games)

## Manual Usage

For more control, use the `fetch_chess_data.py` script directly:

### Lichess Elite Players (Recommended)

Fetches games from top 100 rated players on Lichess (2700+ ELO):

```bash
python dev/fetch_chess_data.py \
    --source lichess-elite \
    --max-games 10000 \
    --time-control classical \
    --with-eval
```

**Pros:**
- Free API, no authentication
- Includes computer evaluations
- High quality games (2500+ rating)
- Fast (~10K games in 10 minutes)

**Cons:**
- Rate limited (slower for very large datasets)

### Specific Lichess User

Fetch games from a specific strong player:

```bash
# Magnus Carlsen
python dev/fetch_chess_data.py \
    --source lichess-user \
    --username DrNykterstein \
    --max-games 5000

# Hikaru Nakamura
python dev/fetch_chess_data.py \
    --source lichess-user \
    --username Hikaru \
    --max-games 5000

# Daniel Naroditsky
python dev/fetch_chess_data.py \
    --source lichess-user \
    --username DanielNaroditsky \
    --max-games 5000
```

**Strong player usernames:**
- `DrNykterstein` - Magnus Carlsen (World Champion)
- `Hikaru` - Hikaru Nakamura (Super GM, streamer)
- `FabianoCaruana` - Fabiano Caruana (Super GM)
- `DanielNaroditsky` - Daniel Naroditsky (GM, educator)
- `gmwesleysoso` - Wesley So (Super GM)
- `GMHKCHESSIFY` - Levon Aronian (Super GM)

### Lichess Database Archives

Download complete monthly databases (largest dataset option):

```bash
python dev/fetch_chess_data.py \
    --source lichess-db \
    --month 2024-01 \
    --sample-rate 0.1  # Use 10% of positions
```

**Pros:**
- Massive dataset (10M+ games per month)
- Pre-annotated with Stockfish evaluations
- Best for large-scale training

**Cons:**
- Very large downloads (10+ GB compressed)
- Requires `zstd` decompression
- Takes hours to download and process

**Available at:** https://database.lichess.org/

### Chess.com Titled Players

Fetch from Chess.com GMs:

```bash
python dev/fetch_chess_data.py \
    --source chesscom-titled \
    --max-games 5000
```

**Note:** Chess.com API is more limited and doesn't include evaluations by default.

## Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Data source (lichess-elite/lichess-user/lichess-db/chesscom-titled) | lichess-elite |
| `--username` | Specific Lichess username | - |
| `--max-games` | Maximum games to fetch | 10000 |
| `--max-positions` | Maximum positions to extract (-1 = all) | -1 |
| `--month` | Month for lichess-db (YYYY-MM) | - |
| `--time-control` | blitz/rapid/classical/all | classical |
| `--min-rating` | Minimum player rating | 2500 |
| `--with-eval` | Include engine evaluations | True |
| `--output-dir` | Output directory | ~/.cache/nanochat/pra_data |
| `--sample-rate` | Sampling rate (0-1) | 1.0 |
| `--keep-pgn` | Keep intermediate PGN file | False |

## Output Format

All sources automatically convert to the parquet format required by PRA:

```
~/.cache/nanochat/pra_data/
├── shard_00000.parquet  # Training data
├── shard_00001.parquet  # Training data
├── ...
└── shard_00010.parquet  # Validation data (last shard)
```

Each parquet file contains:
- `fen` - FEN position string
- `best_move` - UCI move (e.g., "e2e4")
- `eval` - Centipawn evaluation (0 if not available)

## Recommended Workflows

### Quick Experimentation
```bash
# Small, fast dataset for testing
bash dev/quick_fetch.sh
# Select option 1 (Quick test)
```

### Production Training Run
```bash
# Medium dataset with high-quality games
python dev/fetch_chess_data.py \
    --source lichess-elite \
    --max-games 5000 \
    --max-positions 1000000 \
    --time-control classical
```

### Maximum Quality Dataset
```bash
# Combine multiple strong players
for player in DrNykterstein Hikaru FabianoCaruana; do
    python dev/fetch_chess_data.py \
        --source lichess-user \
        --username $player \
        --max-games 2000 \
        --keep-pgn
done

# Then consolidate all PGN files
python dev/prepare_chess_positions.py \
    --input ~/.cache/nanochat/pra_data/*.pgn \
    --output-dir ~/.cache/nanochat/pra_data \
    --with-eval
```

### Massive Scale Training
```bash
# Download full monthly database
python dev/fetch_chess_data.py \
    --source lichess-db \
    --month 2024-01 \
    --sample-rate 0.05  # 5% of all positions = ~500K-1M

# Or multiple months
for month in 2024-01 2024-02 2024-03; do
    python dev/fetch_chess_data.py \
        --source lichess-db \
        --month $month \
        --sample-rate 0.02
done
```

## Data Quality Considerations

### Rating Thresholds
- **2500+** - Strong masters and GMs (default)
- **2700+** - Super GMs and elite players
- **2300+** - FICS Masters, titled players

### Time Controls
- **Classical** (15+ minutes) - Best quality, deep thinking
- **Rapid** (10-15 minutes) - Good balance
- **Blitz** (3-5 minutes) - More tactical, less strategic
- **All** - Maximum variety

### Position Filtering
The data preparation script automatically:
- Skips opening book moves (first 10 plies by default)
- Stops at move 200 (endgames can be repetitive)
- Filters out illegal positions
- Removes duplicate positions

## Troubleshooting

### "python-chess not found"
```bash
pip install chess
```

### "requests not found"
```bash
pip install requests
```

### "No games fetched"
- Check internet connection
- Try a different username
- Verify the month exists for lichess-db
- Check if API is rate limiting (wait a minute)

### "Out of memory during conversion"
- Use `--sample-rate 0.5` to process 50% of positions
- Use `--max-positions 100000` to limit output size
- Process smaller batches with `--max-games`

### Lichess DB files are .zst compressed
```bash
# Install zstd
sudo apt-get install zstd  # Ubuntu/Debian
brew install zstd          # macOS

# Decompress if needed
zstd -d lichess_db_standard_rated_2024-01.pgn.zst
```

## Integration with Training

After fetching data, verify it's ready:

```bash
# Check files
ls -lh ~/.cache/nanochat/pra_data

# Count positions
python -c "
import pyarrow.parquet as pq
import glob
total = sum(len(pq.read_table(f)) for f in glob.glob('~/.cache/nanochat/pra_data/*.parquet'))
print(f'Total positions: {total:,}')
"
```

Then start training:

```bash
# Using TUI
python -m nanochess

# Or directly
python -m scripts.pra_train \
    --depth=8 \
    --n-embd=192 \
    --device-batch-size=256 \
    --total-batch-size=16384 \
    --num-iterations=10000
```

## Advanced: Custom Data Sources

To add your own PGN files:

```bash
# If you have custom PGN files
python dev/prepare_chess_positions.py \
    --input /path/to/your/games.pgn \
    --output-dir ~/.cache/nanochat/pra_data \
    --with-eval  # if your PGN has eval annotations
```

Supported PGN formats:
- Standard PGN
- Lichess format with `[%eval X.XX]` annotations
- Stockfish format with `{+1.23/20}` annotations
- Chess.com exports

## Performance Tips

1. **Use classical time control** - Best quality games
2. **Enable evaluations** - Critical for value head training
3. **Sample large databases** - Use `--sample-rate 0.1` for 10% of positions
4. **Target 500K-1M positions** - Good balance for training
5. **Mix sources** - Combine elite + specific players for diversity

## Data Ethics

- Lichess data is freely available under Creative Commons CC0
- Chess.com data requires respecting their API terms
- Rate limit API calls appropriately
- Don't redistribute large datasets without permission

## Next Steps

After fetching data:
1. Verify dataset size: `ls -lh ~/.cache/nanochat/pra_data`
2. Check sample positions: `python -c "import pyarrow.parquet as pq; print(pq.read_table('~/.cache/nanochat/pra_data/shard_00000.parquet').to_pandas().head())"`
3. Start training: `python -m nanochess`
4. Monitor training: Check W&B dashboard

For more information:
- **Lichess API:** https://lichess.org/api
- **Lichess Database:** https://database.lichess.org/
- **Chess.com API:** https://www.chess.com/news/view/published-data-api
