# Nanochess

Chess-specific models based on nanochat.

## PRA: Piece-Routed Attention

A chess-specific transformer where attention heads are constrained by piece movement patterns, providing interpretability by construction.

**Architecture:**
- **Input:** 64 tokens (one per board square), each embedded with piece type + position + side to move
- **Attention:** Sparse masks enforce piece movement patterns:
  - 2x Knight heads (L-shape patterns)
  - 2x Bishop heads (diagonal patterns)
  - 2x Rook heads (orthogonal patterns)
  - 1x Queen head (combined patterns)
  - 1x King head (one-step patterns)
  - 1x Pawn head (forward + capture patterns)
  - 3x Free heads (learnable attention patterns)
- **Output:**
  - Policy head: 64x64 move logits (from_square, to_square)
  - Value head: scalar evaluation in [-1, 1]

**Key Features:**
- Fixed 64-token sequence (vs 2048 for language) = much larger batch sizes possible
- Piece-specific attention masks precomputed as buffers
- Uses `F.scaled_dot_product_attention` with explicit masks
- ~6M parameters with default config (d8/n_embd=192)
- Runs comfortably on RTX 4070 (16GB VRAM)

## Hardware Requirements

- 16GB+ RAM recommended
- RTX 4070 (16GB VRAM) or better
- CPU training possible but slow

## Environment Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync --extra gpu
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## PRA Model Training

### 1. Data Preparation

PRA requires position/move pairs in parquet format:

```bash
# Convert PGN to parquet
python dev/prepare_chess_positions.py \
    --input games.pgn \
    --output-dir ~/.cache/nanochess/pra_data \
    --with-eval  # Extract engine evaluations if present
```

**Parquet schema:**
| Column | Type | Description |
|--------|------|-------------|
| fen | string | FEN position string |
| best_move | string | UCI move (e.g., "e2e4") |
| eval | int32 | Centipawns evaluation |

**Data sources:**
- [Lichess Database](https://database.lichess.org/) - games with engine evals
- FICS databases
- chess.com exports

Example with Lichess data:
```bash
# Download from https://database.lichess.org/
python dev/prepare_chess_positions.py \
    --input lichess_db_standard_rated_2024-01.pgn.zst \
    --output-dir ~/.cache/nanochess/pra_data \
    --with-eval \
    --max-positions 1000000 \
    --sample-rate 0.1
```

### 2. Training

```bash
# Full training script for RTX 4070
bash runs/pra_rtx4070.sh

# Or run directly with custom options
python -m scripts.pra_train \
    --depth=8 \
    --n-embd=192 \
    --device-batch-size=256 \
    --total-batch-size=16384 \
    --num-iterations=10000 \
    --eval-every=500 \
    --run=my_experiment  # wandb logging
```

**Key parameters:**
| Parameter | Default | Notes |
|-----------|---------|-------|
| --depth | 8 | Transformer layers |
| --n-embd | 192 | Embedding dimension (must divide by n_head) |
| --n-head | 12 | Attention heads (fixed allocation) |
| --device-batch-size | 256 | Per-GPU batch size |
| --total-batch-size | 16384 | Effective batch (gradient accumulation) |
| --policy-weight | 1.0 | Move prediction loss weight |
| --value-weight | 0.1 | Position evaluation loss weight |

### 3. Resuming Training

```bash
# Resume from a specific checkpoint
python -m scripts.pra_train \
    --resume-from-step=5000 \
    --model-tag=pra_d8 \
    --num-iterations=20000
```

### 4. Evaluation

```bash
python -m scripts.pra_eval \
    --model-tag pra_d8 \
    --max-positions 50000
```

**Metrics:**
- Top-1 / Top-5 move accuracy
- Value MAE (mean absolute error vs engine eval)
- Per-piece-type accuracy breakdown

### Loss Function

The training combines two losses:
```
total_loss = policy_weight * policy_loss + value_weight * value_loss
```

- **Policy loss:** Cross-entropy over 64×64=4096 possible moves
- **Value loss:** MSE between predicted and target evaluation (normalized to [-1, 1])

## Files Overview

### New PRA Files

| File | Description |
|------|-------------|
| nanochat/chess_utils.py | FEN parsing, UCI conversion, piece masks |
| nanochat/pra.py | PRA model architecture |
| nanochat/chess_dataloader.py | Position/move dataloader |
| dev/prepare_chess_positions.py | PGN to parquet converter |
| scripts/pra_train.py | PRA training script |
| scripts/pra_eval.py | PRA evaluation script |
| runs/pra_rtx4070.sh | RTX 4070 training config |

### Board Encoding

- **Pieces:** 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
- **Squares:** a1=0, b1=1, ..., h8=63 (row-major from white's perspective)
- **Moves:** UCI format (e.g., "e2e4", "g1f3", "e7e8q")

## VRAM Usage

PRA on RTX 4070 (16GB):
- Model (d8, n_embd=192, ~6M params): ~25MB weights
- Optimizer state (Muon+Adam): ~75MB
- Batch size 256 activations: ~200MB
- Gradient accumulation buffers: ~100MB
- Total: <1GB (lots of headroom for larger batches or models)

**Scaling options:**
| Config | Params | Notes |
|--------|--------|-------|
| d4, n_embd=144 | 2.4M | Quick experiments |
| d8, n_embd=192 | 6.1M | Default |
| d8, n_embd=288 | 13.6M | Larger model |
| d12, n_embd=288 | 17.6M | Still fits 16GB |

## TODO

- [ ] Acquire large chess dataset (Lichess, FICS)
- [ ] Initial PRA training run
- [ ] Analyze learned free attention patterns
- [ ] Add promotion support (extend to 64x64x5 output)
- [ ] Legal move masking during inference
