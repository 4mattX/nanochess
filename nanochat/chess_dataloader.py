"""
Distributed dataloaders for chess position/move pairs.

Loads parquet files with columns:
- fen: FEN string of the position
- best_move: UCI move string (e.g., "e2e4")
- eval: Engine evaluation in centipawns (optional)

Yields batches of:
- board: (B, 64) tensor of piece indices
- side_to_move: (B,) tensor of 0 (black) or 1 (white)
- from_square: (B,) tensor of source square indices
- to_square: (B,) tensor of destination square indices
- value: (B,) tensor of normalized evaluations in [-1, 1]
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info, get_base_dir
from nanochat.chess_utils import fen_to_position, move_to_indices

import os
import glob


def list_chess_parquet_files():
    """List all chess data parquet files in the cache directory."""
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "pra_data")
    pattern = os.path.join(data_dir, "shard_*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def normalize_eval(centipawns: int, cap: float = 1000.0) -> float:
    """Normalize centipawn evaluation to [-1, 1] range.

    Uses tanh-like scaling where ±1000cp maps to approximately ±0.76.

    Args:
        centipawns: Evaluation in centipawns (positive = white advantage)
        cap: Soft cap value for scaling

    Returns:
        Normalized evaluation in [-1, 1]
    """
    # Tanh-style scaling: eval / (|eval| + cap)
    # This gives smooth scaling where large evals asymptote to ±1
    return centipawns / (abs(centipawns) + cap)


def _position_batches(split, resume_state_dict, batch_size):
    """
    Infinite iterator over position batches from parquet files.

    Handles DDP sharding and approximate resume. Each yield is:
    (positions, state_dict) where positions is a list of (fen, move, eval) tuples
    and state_dict tracks position for resumption.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_chess_parquet_files()
    if len(parquet_paths) == 0:
        raise RuntimeError("No chess parquet files found. Run dev/prepare_chess_positions.py first.")

    # Split: all but last file for train, last file for val
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    if len(parquet_paths) == 0:
        raise RuntimeError(f"No parquet files for split '{split}'. Need at least 2 files.")

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)

            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                # Get columns
                fens = rg.column('fen').to_pylist()
                moves = rg.column('best_move').to_pylist()
                # Eval is optional
                if 'eval' in rg.schema.names:
                    evals = rg.column('eval').to_pylist()
                else:
                    evals = [0] * len(fens)  # Default to 0 (equal position)

                # Batch positions
                positions = list(zip(fens, moves, evals))
                for i in range(0, len(positions), batch_size):
                    batch = positions[i:i + batch_size]
                    yield batch, (pq_idx, rg_idx, epoch)

                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def chess_distributed_data_loader_with_state(B, split, device="cuda", resume_state_dict=None):
    """
    Distributed dataloader for chess positions.

    Args:
        B: Batch size
        split: "train" or "val"
        device: Target device for tensors
        resume_state_dict: Optional state dict for resuming

    Yields:
        (board, side_to_move, from_sq, to_sq, value, state_dict)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    batches = _position_batches(split, resume_state_dict, B)
    use_cuda = device == "cuda"

    # Pre-allocate buffers
    board_buffer = torch.empty((B, 64), dtype=torch.long)
    side_buffer = torch.empty(B, dtype=torch.long)
    from_buffer = torch.empty(B, dtype=torch.long)
    to_buffer = torch.empty(B, dtype=torch.long)
    value_buffer = torch.empty(B, dtype=torch.float32)

    # GPU buffers
    board_gpu = torch.empty((B, 64), dtype=torch.long, device=device)
    side_gpu = torch.empty(B, dtype=torch.long, device=device)
    from_gpu = torch.empty(B, dtype=torch.long, device=device)
    to_gpu = torch.empty(B, dtype=torch.long, device=device)
    value_gpu = torch.empty(B, dtype=torch.float32, device=device)

    pq_idx, rg_idx, epoch = 0, 0, 1

    while True:
        positions, (pq_idx, rg_idx, epoch) = next(batches)
        actual_batch_size = len(positions)

        # Parse positions into tensors
        for i, (fen, move, eval_cp) in enumerate(positions):
            board, white_to_move = fen_to_position(fen)
            from_sq, to_sq = move_to_indices(move)
            value = normalize_eval(eval_cp)

            board_buffer[i] = board
            side_buffer[i] = 1 if white_to_move else 0
            from_buffer[i] = from_sq
            to_buffer[i] = to_sq
            value_buffer[i] = value

        # Handle partial batches (pad with zeros, they'll be ignored in loss)
        if actual_batch_size < B:
            board_buffer[actual_batch_size:] = 0
            side_buffer[actual_batch_size:] = 0
            from_buffer[actual_batch_size:] = 0
            to_buffer[actual_batch_size:] = 0
            value_buffer[actual_batch_size:] = 0

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch,
                     "actual_batch_size": actual_batch_size}

        # Copy to GPU
        board_gpu.copy_(board_buffer, non_blocking=use_cuda)
        side_gpu.copy_(side_buffer, non_blocking=use_cuda)
        from_gpu.copy_(from_buffer, non_blocking=use_cuda)
        to_gpu.copy_(to_buffer, non_blocking=use_cuda)
        value_gpu.copy_(value_buffer, non_blocking=use_cuda)

        yield board_gpu[:actual_batch_size], side_gpu[:actual_batch_size], \
              from_gpu[:actual_batch_size], to_gpu[:actual_batch_size], \
              value_gpu[:actual_batch_size], state_dict


def chess_distributed_data_loader(B, split, device="cuda"):
    """Helper that omits state_dict from yields."""
    for board, side, from_sq, to_sq, value, state_dict in chess_distributed_data_loader_with_state(
        B, split, device=device, resume_state_dict=None
    ):
        yield board, side, from_sq, to_sq, value


def get_num_positions(split="train"):
    """Count total positions in the dataset for a given split."""
    parquet_paths = list_chess_parquet_files()
    if len(parquet_paths) == 0:
        return 0

    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    total = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        total += pf.metadata.num_rows
    return total
