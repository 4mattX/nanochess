"""Utility functions for the Nanochess TUI."""

import os
import glob
from dataclasses import dataclass
from typing import Optional

import pyarrow.parquet as pq


@dataclass
class GPUInfo:
    """Information about the current GPU setup."""
    available: bool
    device_type: str  # cuda, mps, cpu
    device_name: str
    vram_total_gb: float
    vram_used_gb: float
    cuda_version: str
    pytorch_version: str


@dataclass
class DatasetInfo:
    """Information about the prepared dataset."""
    path: str
    num_train_positions: int
    num_val_positions: int
    num_shards: int
    exists: bool


def get_base_dir() -> str:
    """Get the base directory for nanochat data."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        return os.environ.get("NANOCHAT_BASE_DIR")
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    return os.path.join(cache_dir, "nanochat")


def get_gpu_info() -> GPUInfo:
    """Detect GPU information for display in the TUI."""
    import torch

    pytorch_version = torch.__version__

    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = props.total_memory / (1024**3)

        # Get current memory usage
        vram_used_gb = torch.cuda.memory_allocated(0) / (1024**3)

        # Get CUDA version
        cuda_version = torch.version.cuda or "N/A"

        return GPUInfo(
            available=True,
            device_type=device_type,
            device_name=device_name,
            vram_total_gb=vram_total_gb,
            vram_used_gb=vram_used_gb,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
        )
    elif torch.backends.mps.is_available():
        return GPUInfo(
            available=True,
            device_type="mps",
            device_name="Apple Silicon",
            vram_total_gb=0,
            vram_used_gb=0,
            cuda_version="N/A",
            pytorch_version=pytorch_version,
        )
    else:
        return GPUInfo(
            available=False,
            device_type="cpu",
            device_name="CPU Only",
            vram_total_gb=0,
            vram_used_gb=0,
            cuda_version="N/A",
            pytorch_version=pytorch_version,
        )


def get_dataset_info() -> DatasetInfo:
    """Get information about the prepared chess dataset."""
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "pra_data")
    pattern = os.path.join(data_dir, "shard_*.parquet")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        return DatasetInfo(
            path=data_dir,
            num_train_positions=0,
            num_val_positions=0,
            num_shards=0,
            exists=False,
        )

    # Train: all but last file, Val: last file
    train_files = files[:-1] if len(files) > 1 else []
    val_files = files[-1:] if len(files) > 0 else []

    num_train = 0
    for path in train_files:
        try:
            pf = pq.ParquetFile(path)
            num_train += pf.metadata.num_rows
        except Exception:
            pass

    num_val = 0
    for path in val_files:
        try:
            pf = pq.ParquetFile(path)
            num_val += pf.metadata.num_rows
        except Exception:
            pass

    return DatasetInfo(
        path=data_dir,
        num_train_positions=num_train,
        num_val_positions=num_val,
        num_shards=len(files),
        exists=len(files) > 0,
    )


def format_number(n: int) -> str:
    """Format a number with commas for display."""
    return f"{n:,}"


def shorten_path(path: str, max_len: int = 30) -> str:
    """Shorten a path for display by replacing the home directory with ~."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]

    if len(path) <= max_len:
        return path

    # Truncate from the beginning
    return "..." + path[-(max_len - 3):]


# Training presets for different hardware configurations
PRESETS = {
    "Instant Test": {
        "depth": 2,
        "n_embd": 60,
        "n_head": 12,
        "num_iterations": 5,
        "device_batch_size": 16,
        "total_batch_size": 32,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "weight_decay": 0.1,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.0,
        "warmdown_ratio": 0.0,
        "final_lr_frac": 1.0,
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "eval_every": 2,
        "eval_positions": 100,
        "save_every": -1,
    },
    "Quick Test": {
        "depth": 4,
        "n_embd": 144,
        "n_head": 12,
        "num_iterations": 100,
        "device_batch_size": 64,
        "total_batch_size": 256,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "weight_decay": 0.1,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.02,
        "warmdown_ratio": 0.3,
        "final_lr_frac": 0.0,
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "eval_every": 50,
        "eval_positions": 1000,
        "save_every": -1,
    },
    "RTX 4070 (16GB)": {
        "depth": 8,
        "n_embd": 192,
        "n_head": 12,
        "num_iterations": 10000,
        "device_batch_size": 256,
        "total_batch_size": 16384,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "weight_decay": 0.1,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.02,
        "warmdown_ratio": 0.3,
        "final_lr_frac": 0.0,
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "eval_every": 500,
        "eval_positions": 10000,
        "save_every": -1,
    },
    "8xH100 Speedrun": {
        "depth": 8,
        "n_embd": 288,
        "n_head": 12,
        "num_iterations": 50000,
        "device_batch_size": 512,
        "total_batch_size": 65536,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "weight_decay": 0.1,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.02,
        "warmdown_ratio": 0.3,
        "final_lr_frac": 0.0,
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "eval_every": 1000,
        "eval_positions": 10000,
        "save_every": 5000,
    },
    "CPU Debug": {
        "depth": 2,
        "n_embd": 96,
        "n_head": 12,
        "num_iterations": 10,
        "device_batch_size": 8,
        "total_batch_size": 8,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "weight_decay": 0.1,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.02,
        "warmdown_ratio": 0.3,
        "final_lr_frac": 0.0,
        "policy_weight": 1.0,
        "value_weight": 0.1,
        "eval_every": 5,
        "eval_positions": 100,
        "save_every": -1,
    },
}


# Default training configuration (matches pra_train.py defaults)
DEFAULT_CONFIG = {
    "depth": 8,
    "n_embd": 192,
    "n_head": 12,
    "num_iterations": 10000,
    "device_batch_size": 256,
    "total_batch_size": 16384,
    "embedding_lr": 0.3,
    "matrix_lr": 0.02,
    "weight_decay": 0.1,
    "adam_beta1": 0.8,
    "adam_beta2": 0.95,
    "warmup_ratio": 0.02,
    "warmdown_ratio": 0.3,
    "final_lr_frac": 0.0,
    "policy_weight": 1.0,
    "value_weight": 0.1,
    "eval_every": 500,
    "eval_positions": 10000,
    "save_every": -1,
    "resume_from_step": -1,
    "run_name": "dummy",
    "model_tag": "",
    "device_type": "",
    "wandb_enabled": False,
}


def build_training_command(config: dict) -> list[str]:
    """Build the training command from configuration."""
    cmd = ["python", "-u", "-m", "scripts.pra_train"]  # -u for unbuffered output

    # Add all parameters
    if config.get("depth", 8) != 8:
        cmd.extend([f"--depth={config['depth']}"])
    else:
        cmd.extend([f"--depth={config.get('depth', 8)}"])

    cmd.extend([
        f"--n-embd={config.get('n_embd', 192)}",
        f"--n-head={config.get('n_head', 12)}",
        f"--num-iterations={config.get('num_iterations', 10000)}",
        f"--device-batch-size={config.get('device_batch_size', 256)}",
        f"--total-batch-size={config.get('total_batch_size', 16384)}",
        f"--embedding-lr={config.get('embedding_lr', 0.3)}",
        f"--matrix-lr={config.get('matrix_lr', 0.02)}",
        f"--weight-decay={config.get('weight_decay', 0.1)}",
        f"--adam-beta1={config.get('adam_beta1', 0.8)}",
        f"--adam-beta2={config.get('adam_beta2', 0.95)}",
        f"--warmup-ratio={config.get('warmup_ratio', 0.02)}",
        f"--warmdown-ratio={config.get('warmdown_ratio', 0.3)}",
        f"--final-lr-frac={config.get('final_lr_frac', 0.0)}",
        f"--policy-weight={config.get('policy_weight', 1.0)}",
        f"--value-weight={config.get('value_weight', 0.1)}",
        f"--eval-every={config.get('eval_every', 500)}",
        f"--eval-positions={config.get('eval_positions', 10000)}",
        f"--save-every={config.get('save_every', -1)}",
    ])

    # Optional parameters
    if config.get("resume_from_step", -1) != -1:
        cmd.append(f"--resume-from-step={config['resume_from_step']}")

    run_name = config.get("run_name", "dummy")
    if run_name and run_name != "dummy" and config.get("wandb_enabled", False):
        cmd.append(f"--run={run_name}")

    model_tag = config.get("model_tag", "")
    if model_tag:
        cmd.append(f"--model-tag={model_tag}")

    device_type = config.get("device_type", "")
    if device_type:
        cmd.append(f"--device-type={device_type}")

    return cmd


def format_command_preview(cmd: list[str], max_width: int = 40) -> str:
    """Format a command for preview display."""
    lines = []
    lines.append(cmd[0] + " " + cmd[1] + " \\")
    lines.append("  " + cmd[2] + " \\")

    for arg in cmd[3:]:
        lines.append(f"  {arg} \\")

    # Remove trailing backslash from last line
    if lines:
        lines[-1] = lines[-1].rstrip(" \\")

    return "\n".join(lines)
