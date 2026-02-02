#!/bin/bash

# PRA (Piece-Routed Attention) training script for RTX 4070 (16GB VRAM)
# Hardware: 64GB RAM, RTX 4070 16GB, i7
#
# Usage:
#   bash runs/pra_rtx4070.sh
#
# For wandb logging:
#   WANDB_RUN=pra_chess bash runs/pra_rtx4070.sh

set -e  # Exit on error

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochess"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Configuration for RTX 4070 16GB VRAM
#
# PRA Model VRAM Usage (approximate):
#   - n_layer=8, n_embd=256:  ~2.5M params, ~500MB model
#   - Fixed 64-token sequence (vs 2048 for language)
#   - Can use large batch sizes: 256-1024 comfortably
#   - Total estimated: <2GB even with batch_size=256
#
# This leaves plenty of headroom on 16GB VRAM!

# Model architecture
DEPTH=8              # Number of transformer layers
N_EMBD=192           # Embedding dimension (must be divisible by N_HEAD)
N_HEAD=12            # Attention heads (knight*2 + bishop*2 + rook*2 + queen + king + pawn + free*3)

# Batch sizes - can be quite large due to fixed 64-token sequence
DEVICE_BATCH_SIZE=256
TOTAL_BATCH_SIZE=16384  # ~64 gradient accumulation steps with batch 256

# Training iterations (adjust based on dataset size)
# For 1M positions: ~60 epochs with batch 16384
NUM_ITERATIONS=10000

# Loss weights
POLICY_WEIGHT=1.0   # Move prediction loss
VALUE_WEIGHT=0.1    # Position evaluation loss

# Evaluation frequency
EVAL_EVERY=500
SAVE_EVERY=1000

# -----------------------------------------------------------------------------
# Data preparation
# You need to provide PGN files with chess games

PRA_DATA_DIR="$NANOCHAT_BASE_DIR/pra_data"
PGN_DIR="$NANOCHAT_BASE_DIR/pgn_files"

echo "Checking for training data..."
if [ ! -d "$PRA_DATA_DIR" ] || [ -z "$(ls -A $PRA_DATA_DIR 2>/dev/null)" ]; then
    echo ""
    echo "No PRA training data found!"
    echo ""
    echo "To prepare data, download PGN files and run:"
    echo "  python dev/prepare_chess_positions.py --input YOUR_PGN_FILE.pgn --output-dir $PRA_DATA_DIR"
    echo ""
    echo "Example with Lichess data (requires python-chess):"
    echo "  pip install chess"
    echo "  # Download from https://database.lichess.org/"
    echo "  python dev/prepare_chess_positions.py \\"
    echo "    --input lichess_db_standard_rated_2024-01.pgn.zst \\"
    echo "    --output-dir $PRA_DATA_DIR \\"
    echo "    --with-eval \\"
    echo "    --max-positions 1000000"
    echo ""

    # Create a small synthetic dataset for testing
    echo "Creating small test dataset for demo..."
    mkdir -p "$PRA_DATA_DIR"

    python3 << 'EOF'
import os
import pyarrow as pa
import pyarrow.parquet as pq

# Create minimal test data
positions = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", 35),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5", 25),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "g1f3", 30),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6", 20),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5", 40),
] * 200  # 1000 positions

data_dir = os.path.expanduser("~/.cache/nanochess/pra_data")

# Train shard
table = pa.table({
    'fen': [p[0] for p in positions[:900]],
    'best_move': [p[1] for p in positions[:900]],
    'eval': [p[2] for p in positions[:900]],
})
pq.write_table(table, os.path.join(data_dir, "shard_00000.parquet"))

# Val shard
table = pa.table({
    'fen': [p[0] for p in positions[900:]],
    'best_move': [p[1] for p in positions[900:]],
    'eval': [p[2] for p in positions[900:]],
})
pq.write_table(table, os.path.join(data_dir, "shard_00001.parquet"))

print(f"Created test data in {data_dir}")
EOF

fi

# -----------------------------------------------------------------------------
# PRA model training (single GPU, no torchrun needed for RTX 4070)

echo ""
echo "Starting PRA model training..."
echo "Model: depth=$DEPTH, n_embd=$N_EMBD, n_head=$N_HEAD"
echo "Batch: device=$DEVICE_BATCH_SIZE, total=$TOTAL_BATCH_SIZE"
echo ""

python -m scripts.pra_train \
    --depth=$DEPTH \
    --n-embd=$N_EMBD \
    --n-head=$N_HEAD \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --num-iterations=$NUM_ITERATIONS \
    --policy-weight=$POLICY_WEIGHT \
    --value-weight=$VALUE_WEIGHT \
    --eval-every=$EVAL_EVERY \
    --save-every=$SAVE_EVERY \
    --run=$WANDB_RUN \
    --model-tag="pra_d${DEPTH}"

echo ""
echo "Training complete!"
echo ""

# -----------------------------------------------------------------------------
# Evaluation

echo "Running evaluation..."
python -m scripts.pra_eval \
    --model-tag="pra_d${DEPTH}" \
    --device-batch-size=256 \
    --max-positions=10000

echo ""
echo "Done! Model saved to: $NANOCHAT_BASE_DIR/pra_checkpoints/pra_d${DEPTH}"
