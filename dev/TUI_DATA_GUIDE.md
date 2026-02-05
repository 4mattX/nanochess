# How the TUI Sees Your Downloaded Data

## Quick Answer

**Yes!** The TUI automatically detects your downloaded data. Once you fetch data, it will instantly show up in the TUI.

## The Flow

```
1. FETCH DATA
   ↓
   bash dev/quick_fetch.sh
   OR
   python dev/fetch_chess_data.py --source lichess-elite --max-games 10000
   ↓
   Data saved to: ~/.cache/nanochat/pra_data/
   ├── shard_00000.parquet  (training)
   ├── shard_00001.parquet  (training)
   └── shard_00002.parquet  (validation)

2. VERIFY (optional)
   ↓
   python dev/inspect_data.py
   ↓
   Shows:
   - ✓ GOOD 500,000 positions
   - ✓ GOOD Eval coverage 98.5%
   - ✓ GOOD High move diversity

3. LAUNCH TUI
   ↓
   python -m nanochess
   ↓
   TUI automatically detects data at startup

4. TUI DISPLAYS
   ↓
   Status bar shows: "📊 500,000 positions"
   Dataset panel shows:
   - Train: 450,000 positions
   - Val: 50,000 positions
   - Shards: 3 files
```

## What You'll See in the TUI

### Status Bar (Bottom of Screen)
```
🖥️ NVIDIA RTX 4070 | 💾 1.2/16GB | 📊 500,000 positions
```

The status bar automatically shows:
- **GPU info** - Your hardware
- **VRAM usage** - Memory consumption
- **Dataset size** - Total positions detected

### Dataset Panel (Training Screen)

```
┌─ DATA SOURCE ────────────────────┐
│                                  │
│  ○ Local PGN                     │
│  ● Prepared Data    ← Selected   │
│  ○ Lichess DB                    │
│                                  │
│  Path: ~/.cache/nanochat/pra_data│
│                                  │
│  Train:  450,000 positions       │
│  Val:    50,000 positions        │
│  Shards: 3 files                 │
│                                  │
├─ PRESETS ───────────────────────┤
│                                  │
│  ○ Instant Test                  │
│  ○ Quick Test                    │
│  ● RTX 4070 (16GB) ← Selected    │
│  ○ 8xH100 Speedrun               │
│  ○ CPU Debug                     │
│                                  │
└──────────────────────────────────┘
```

### What Happens Automatically

1. **On TUI startup**: Scans `~/.cache/nanochat/pra_data/`
2. **Detects parquet files**: Counts total positions
3. **Updates display**: Shows train/val split
4. **Refreshes status bar**: Shows dataset size
5. **Ready to train**: All configs use this data

## Selection Process

The TUI doesn't require you to "select" the dataset - it **automatically uses** whatever data is in `~/.cache/nanochat/pra_data/`.

### Data Source Radio Buttons

These are for **future functionality** (not yet implemented):

- **○ Local PGN** - Will process PGN files on-the-fly
- **● Prepared Data** - Uses your fetched parquet files (current mode)
- **○ Lichess DB** - Will download database archives

For now, just fetch data with the scripts, and the TUI will automatically use it.

## Preset Selection

The **PRESETS** section lets you quickly configure training parameters:

```bash
Press F5 or click a preset:

○ Instant Test        # 5 iterations, 2 layers, ~10 seconds
○ Quick Test          # 100 iterations, 4 layers, ~5 minutes
● RTX 4070 (16GB)     # 10K iterations, 8 layers, ~2 hours
○ 8xH100 Speedrun     # 50K iterations, 8 layers, ~4 hours
○ CPU Debug           # 10 iterations, CPU-only testing
```

When you select a preset:
1. All training parameters update automatically
2. Command preview refreshes
3. Time/cost estimates update

## Complete Workflow Example

```bash
# Terminal 1: Fetch data
$ bash dev/quick_fetch.sh
> Select option 2 (Small dataset)
> Downloading from Lichess elite players...
> Processing games...
> Converting to parquet...
✓ Done! 50,000 positions ready

# Terminal 2: Verify data
$ python dev/inspect_data.py
NANOCHESS DATA INSPECTION
Data directory: ~/.cache/nanochat/pra_data
Parquet files: 2

  TRAIN shard_00000.parquet    45,034 positions
  VAL   shard_00001.parquet     5,003 positions

TOTAL: 50,037 positions

✓ GOOD 50,037 positions - sufficient for initial training
✓ GOOD Eval coverage 98.5% - value head will train well
✓ GOOD High move diversity (2,341 unique moves)

# Terminal 3: Launch TUI
$ python -m nanochess

# You see:
┌─ NANOCHESS ──────────────────────┐
│                                  │
│  [T] Train Model                 │
│  [E] Evaluate Model              │
│  [I] Interactive Play            │
│  [Q] Quit                        │
│                                  │
└──────────────────────────────────┘
🖥️ NVIDIA RTX 4070 | 💾 0.5/16GB | 📊 50,037 positions
                                          ^^^^^^^ Your data!

# Press T for training screen:
# Dataset panel automatically shows:
#   Train: 45,034 positions
#   Val:   5,003 positions
#   Shards: 2 files
```

## Troubleshooting

### "📊 Not found" in status bar

**Cause**: No data in `~/.cache/nanochat/pra_data/`

**Fix**:
```bash
bash dev/quick_fetch.sh
```

### "📊 0 positions" in status bar

**Cause**: Empty or corrupted parquet files

**Fix**:
```bash
# Remove old data
rm -rf ~/.cache/nanochat/pra_data

# Fetch fresh data
bash dev/quick_fetch.sh
```

### Dataset panel shows "No data found"

**Cause**: Path mismatch or missing files

**Fix**:
```bash
# Check if data exists
ls -lh ~/.cache/nanochat/pra_data

# If empty, fetch data
bash dev/quick_fetch.sh

# Verify with inspector
python dev/inspect_data.py
```

### TUI doesn't update after fetching data

**Cause**: TUI was already running when you fetched data

**Fix**:
```bash
# Exit and restart TUI
# Press Q to quit
# Then restart: python -m nanochess
```

The TUI only scans for data at startup, so restart it after fetching new data.

## Data Management

### Check current dataset
```bash
python dev/inspect_data.py
```

### Replace dataset
```bash
# Remove old data
rm -rf ~/.cache/nanochat/pra_data/*.parquet

# Fetch new data
bash dev/quick_fetch.sh
```

### Add more data
```bash
# Fetch additional data (appends to existing)
python dev/fetch_chess_data.py \
    --source lichess-user \
    --username Hikaru \
    --max-games 1000

# Then re-process all PGN together
python dev/prepare_chess_positions.py \
    --input ~/.cache/nanochat/pra_data/*.pgn \
    --output-dir ~/.cache/nanochat/pra_data
```

### Use different data location

Set the environment variable before launching TUI:

```bash
export NANOCHAT_BASE_DIR=/path/to/custom/location
python -m nanochess
```

TUI will look for data in `/path/to/custom/location/pra_data/`

## Summary

✓ **Automatic detection** - TUI scans on startup
✓ **No manual selection** - Uses whatever's in the data folder
✓ **Live status display** - Status bar shows dataset size
✓ **Detailed info panel** - Shows train/val split
✓ **Preset integration** - All presets use the same data

Just fetch data once, launch TUI, and start training!
