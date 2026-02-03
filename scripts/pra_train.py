"""
Train PRA (Piece-Routed Attention) model for chess. From root directory:

python -m scripts.pra_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.pra_train

For single GPU with limited VRAM:
python -m scripts.pra_train --depth=8 --n-embd=256 --device-batch-size=64 --num-iterations=5000
"""

import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.pra import PRA, PRAConfig
from nanochat.chess_dataloader import (
    chess_distributed_data_loader_with_state,
    chess_distributed_data_loader,
    get_num_positions,
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops
)
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train PRA chess model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=8, help="number of transformer layers")
parser.add_argument("--n-embd", type=int, default=192, help="embedding dimension (must be divisible by n_head)")
parser.add_argument("--n-head", type=int, default=12, help="number of attention heads")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=10000, help="number of optimization steps")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=256, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=16384, help="total batch size (positions)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay for Muon optimizer")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup-ratio", type=float, default=0.02, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.3, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step")
# Loss weights
parser.add_argument("--policy-weight", type=float, default=1.0, help="weight for policy loss")
parser.add_argument("--value-weight", type=float, default=0.1, help="weight for value loss")
# Evaluation
parser.add_argument("--eval-every", type=int, default=500, help="evaluate every N steps (-1 = disable)")
parser.add_argument("--eval-positions", type=int, default=10000, help="number of positions to evaluate")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochess-pra", name=args.run, config=user_config)

# -----------------------------------------------------------------------------
# Dataset info
num_train_positions = get_num_positions("train")
num_val_positions = get_num_positions("val")
print0(f"Training positions: {num_train_positions:,}")
print0(f"Validation positions: {num_val_positions:,}")

if num_train_positions == 0:
    print0("ERROR: No training data found. Run dev/prepare_chess_positions.py first.")
    exit(1)

# -----------------------------------------------------------------------------
# Gradient accumulation
positions_per_fwdbwd = args.device_batch_size * ddp_world_size
assert args.total_batch_size % positions_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // positions_per_fwdbwd
print0(f"Positions / micro-batch: {positions_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates
batch_lr_scale = 1.0
reference_batch_size = 2**14  # 16384
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(
    n_layer=args.depth,
    n_embd=args.n_embd,
    n_head=args.n_head,
)
with torch.device("meta"):
    model_config = PRAConfig(**model_config_kwargs)
    model = PRA(model_config)
model.to_empty(device=device)
model.init_weights()

# Resume if requested
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"pra_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "pra_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model

# Fix TF32 configuration for compilation (use new API consistently)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = torch.compile(model, dynamic=False)

# Parameter counts
num_params = orig_model.num_params()
print0(f"Total parameters: {num_params:,}")
num_flops_per_position = orig_model.estimate_flops()
print0(f"Estimated FLOPs per position: {num_flops_per_position:e}")

num_iterations = args.num_iterations
total_positions = args.total_batch_size * num_iterations
print0(f"Total training positions: {total_positions:,}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=args.weight_decay,
    adam_betas=adam_betas,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data.get("dataloader_state_dict")
train_loader = chess_distributed_data_loader_with_state(
    args.device_batch_size, "train", device=device, resume_state_dict=dataloader_resume_state_dict
)
build_val_loader = lambda: chess_distributed_data_loader(
    args.device_batch_size, "val", device=device
)

# Prefetch first batch
board, side, from_sq, to_sq, value, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Weight decay scheduler
def get_weight_decay(it):
    return args.weight_decay * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Loop state
if not resuming:
    step = 0
    val_policy_acc = None
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_policy_acc = meta_data.get("val_policy_acc")
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Evaluation function
def evaluate_model(model, val_loader, num_positions):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_positions // args.device_batch_size):
            board, side, from_sq, to_sq, value = next(val_loader)
            with autocast_ctx:
                result = model(
                    board, side, from_sq, to_sq, value,
                    policy_weight=args.policy_weight, value_weight=args.value_weight
                )

            total_loss += result['loss'].item()
            total_policy_loss += result['policy_loss'].item()
            if 'value_loss' in result:
                total_value_loss += result['value_loss'].item()

            # Compute top-1 and top-5 accuracy
            B = board.size(0)
            policy_flat = result['policy_logits'].view(B, -1)  # (B, 4096)
            target_move = from_sq * 64 + to_sq

            # Top-1
            pred_top1 = policy_flat.argmax(dim=-1)
            correct_top1 += (pred_top1 == target_move).sum().item()

            # Top-5
            _, pred_top5 = policy_flat.topk(5, dim=-1)
            correct_top5 += (pred_top5 == target_move.unsqueeze(1)).any(dim=1).sum().item()

            total += B

    model.train()

    n_batches = num_positions // args.device_batch_size
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches if total_value_loss > 0 else 0,
        'top1_acc': correct_top1 / total,
        'top5_acc': correct_top5 / total,
    }

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_position * args.total_batch_size * step

    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        val_loader = build_val_loader()
        with autocast_ctx:
            eval_results = evaluate_model(orig_model, val_loader, args.eval_positions)

        val_policy_acc = eval_results['top1_acc']
        print0(f"Step {step:05d} | Val loss: {eval_results['loss']:.4f} | "
               f"Top-1: {eval_results['top1_acc']:.4f} | Top-5: {eval_results['top5_acc']:.4f}")

        if eval_results['loss'] < min_val_loss:
            min_val_loss = eval_results['loss']

        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/loss": eval_results['loss'],
            "val/policy_loss": eval_results['policy_loss'],
            "val/value_loss": eval_results['value_loss'],
            "val/top1_acc": eval_results['top1_acc'],
            "val/top5_acc": eval_results['top5_acc'],
        })

    # Save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_policy_acc": val_policy_acc,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            result = model(
                board, side, from_sq, to_sq, value,
                policy_weight=args.policy_weight, value_weight=args.value_weight
            )
            loss = result['loss']

        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        board, side, from_sq, to_sq, value, dataloader_state_dict = next(train_loader)

    # Optimizer step
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    pos_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_position * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    if step > 10:
        total_training_time += dt

    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""

    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.4f} | "
           f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | pos/sec: {pos_per_sec:,} | mfu: {mfu:.2f} | "
           f"epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/pos_per_sec": pos_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })

    # State update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # GC management
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# -----------------------------------------------------------------------------
# Final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_policy_acc is not None:
    print0(f"Final top-1 accuracy: {val_policy_acc:.4f}")
    print0(f"Minimum validation loss: {min_val_loss:.4f}")

# Cleanup
wandb_run.finish()
compute_cleanup()
