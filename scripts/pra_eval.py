"""
Evaluate PRA (Piece-Routed Attention) model for chess.

Usage:
    python -m scripts.pra_eval
    python -m scripts.pra_eval --model-tag pra_d8 --step 5000

Metrics:
- Top-1/Top-5 move accuracy
- Value MAE vs engine eval
- Per-piece-type accuracy breakdown
"""

import argparse
import os
from collections import defaultdict
from contextlib import nullcontext

import torch

from nanochat.pra import PRA, PRAConfig
from nanochat.chess_dataloader import chess_distributed_data_loader, get_num_positions
from nanochat.chess_utils import (
    get_piece_on_square, get_base_piece_type, index_to_algebraic
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
)
from nanochat.checkpoint_manager import load_checkpoint


def find_last_step(checkpoint_dir):
    """Find the latest checkpoint step in a directory."""
    import glob
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    steps = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    return max(steps)


def load_pra_model(checkpoint_dir, step, device):
    """Load a PRA model from checkpoint."""
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Handle torch.compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config_kwargs = meta_data["model_config"]
    model_config = PRAConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = PRA(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    return model, meta_data


def evaluate_accuracy(model, val_loader, num_positions, device, device_batch_size, autocast_ctx):
    """Evaluate move prediction accuracy."""
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Per-piece stats
    piece_correct = defaultdict(int)
    piece_total = defaultdict(int)

    # Value prediction stats
    value_abs_error = 0
    value_count = 0

    with torch.no_grad():
        for i in range(num_positions // device_batch_size):
            board, side, from_sq, to_sq, value = next(val_loader)

            with autocast_ctx:
                result = model(board, side)

            B = board.size(0)
            policy_flat = result['policy_logits'].view(B, -1)  # (B, 4096)
            target_move = from_sq * 64 + to_sq

            # Top-1
            pred_top1 = policy_flat.argmax(dim=-1)
            top1_correct = (pred_top1 == target_move)
            correct_top1 += top1_correct.sum().item()

            # Top-5
            _, pred_top5 = policy_flat.topk(5, dim=-1)
            top5_correct = (pred_top5 == target_move.unsqueeze(1)).any(dim=1)
            correct_top5 += top5_correct.sum().item()

            # Per-piece accuracy
            for j in range(B):
                from_sq_j = from_sq[j].item()
                piece = get_piece_on_square(board[j], from_sq_j)
                piece_type = get_base_piece_type(piece)
                piece_total[piece_type] += 1
                if top1_correct[j].item():
                    piece_correct[piece_type] += 1

            # Value error
            pred_value = result['value']
            value_abs_error += (pred_value - value).abs().sum().item()
            value_count += B

            total += B

            if (i + 1) % 100 == 0:
                print0(f"  Evaluated {(i + 1) * device_batch_size} positions...")

    results = {
        'top1_acc': correct_top1 / total,
        'top5_acc': correct_top5 / total,
        'value_mae': value_abs_error / value_count,
        'total_positions': total,
        'piece_accuracy': {},
    }

    # Per-piece accuracy
    for piece_type in sorted(piece_total.keys()):
        if piece_total[piece_type] > 0:
            acc = piece_correct[piece_type] / piece_total[piece_type]
            results['piece_accuracy'][piece_type] = {
                'accuracy': acc,
                'correct': piece_correct[piece_type],
                'total': piece_total[piece_type],
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PRA chess model")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--model-tag", type=str, default=None, help="model directory name")
    parser.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    parser.add_argument("--device-batch-size", type=int, default=256, help="batch size for evaluation")
    parser.add_argument("--max-positions", type=int, default=50000, help="max positions to evaluate")
    args = parser.parse_args()

    # Device setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, ddp_rank, _, _, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Find checkpoint
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "pra_checkpoints")

    if args.model_tag:
        checkpoint_dir = os.path.join(checkpoints_dir, args.model_tag)
    else:
        # Find largest model
        if not os.path.exists(checkpoints_dir):
            print0("ERROR: No PRA checkpoints found. Train a model first.")
            return
        model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
        if not model_tags:
            print0("ERROR: No PRA checkpoints found.")
            return
        # Pick most recently modified
        model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
        checkpoint_dir = os.path.join(checkpoints_dir, model_tags[0])
        print0(f"Using checkpoint: {model_tags[0]}")

    # Find step
    step = args.step if args.step > 0 else find_last_step(checkpoint_dir)
    if step is None:
        print0(f"ERROR: No checkpoints found in {checkpoint_dir}")
        return
    print0(f"Loading checkpoint step {step}")

    # Load model
    model, meta_data = load_pra_model(checkpoint_dir, step, device)
    print0(f"Model config: {meta_data['model_config']}")
    print0(f"Parameters: {model.num_params():,}")

    # Dataset info
    num_val = get_num_positions("val")
    print0(f"Validation positions available: {num_val:,}")
    num_eval = min(args.max_positions, num_val)
    print0(f"Evaluating on {num_eval:,} positions")

    # Evaluate
    val_loader = chess_distributed_data_loader(args.device_batch_size, "val", device=device)
    results = evaluate_accuracy(model, val_loader, num_eval, device, args.device_batch_size, autocast_ctx)

    # Print results
    print0("\n" + "=" * 60)
    print0("PRA Model Evaluation Results")
    print0("=" * 60)
    print0(f"Total positions evaluated: {results['total_positions']:,}")
    print0(f"Top-1 Accuracy: {results['top1_acc']:.4f} ({results['top1_acc']*100:.2f}%)")
    print0(f"Top-5 Accuracy: {results['top5_acc']:.4f} ({results['top5_acc']*100:.2f}%)")
    print0(f"Value MAE: {results['value_mae']:.4f}")

    print0("\nPer-Piece Accuracy:")
    print0("-" * 40)
    for piece_type, stats in sorted(results['piece_accuracy'].items()):
        print0(f"  {piece_type:8s}: {stats['accuracy']:.4f} ({stats['correct']:5d}/{stats['total']:5d})")

    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    main()
