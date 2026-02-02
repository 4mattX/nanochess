"""
PRA (Piece-Routed Attention) model for chess.

A chess-specific transformer where attention heads are constrained by piece movement
patterns, providing interpretability by construction.

Architecture:
- Input: 64 tokens (one per board square), each embedded with piece type + position
- Attention: Sparse masks enforce piece movement patterns
  - 2× Knight heads (L-shape patterns)
  - 2× Bishop heads (diagonal patterns)
  - 2× Rook heads (orthogonal patterns)
  - 1× Queen head (combined diagonal + orthogonal)
  - 1× King head (one-step patterns)
  - 1× Pawn head (forward + capture patterns)
  - 3× Free heads (learnable attention patterns)
- Output: Policy head (64×64 move logits) + Value head (position evaluation)

Notable features:
- No positional embeddings (position is encoded via piece placement)
- Functional RMSNorm (no learnable params)
- No bias in linear layers
- Piece-specific attention masks registered as buffers
- Uses F.scaled_dot_product_attention (not Flash Attention) for explicit mask support
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.chess_utils import (
    compute_knight_mask, compute_bishop_mask, compute_rook_mask,
    compute_queen_mask, compute_king_mask, compute_pawn_mask,
    NUM_PIECE_TYPES,
)


@dataclass
class PRAConfig:
    """Configuration for PRA model."""
    n_layer: int = 8           # Number of transformer layers
    n_embd: int = 192          # Embedding dimension (must be divisible by n_head)
    n_head: int = 12           # Total number of attention heads
    # Head allocation: knight(2) + bishop(2) + rook(2) + queen(1) + king(1) + pawn(1) + free(3) = 12
    n_knight_heads: int = 2
    n_bishop_heads: int = 2
    n_rook_heads: int = 2
    n_queen_heads: int = 1
    n_king_heads: int = 1
    n_pawn_heads: int = 1
    n_free_heads: int = 3
    # Board constants
    n_squares: int = 64        # 8x8 board
    n_piece_types: int = 13    # 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk


def norm(x):
    """Purely functional rmsnorm with no learnable params."""
    return F.rms_norm(x, (x.size(-1),))


class SquareEmbedding(nn.Module):
    """Embedding layer for chess board squares.

    Each square is embedded with:
    - Piece type embedding (what piece is on this square)
    - Position embedding (which square this is)
    - Side to move embedding (whose turn it is)
    """
    def __init__(self, config: PRAConfig):
        super().__init__()
        self.config = config

        # Piece type embedding: 13 types (empty + 6 white + 6 black)
        self.piece_embed = nn.Embedding(config.n_piece_types, config.n_embd)

        # Position embedding: 64 squares
        self.position_embed = nn.Embedding(config.n_squares, config.n_embd)

        # Side to move embedding: 0=black, 1=white
        self.side_embed = nn.Embedding(2, config.n_embd)

    def forward(self, board: torch.Tensor, side_to_move: torch.Tensor) -> torch.Tensor:
        """Embed the board state.

        Args:
            board: (B, 64) tensor of piece indices (0-12)
            side_to_move: (B,) tensor of 0 (black) or 1 (white)

        Returns:
            (B, 64, n_embd) embedded board
        """
        B = board.size(0)
        device = board.device

        # Piece embeddings
        piece_emb = self.piece_embed(board)  # (B, 64, n_embd)

        # Position embeddings (same for all boards in batch)
        positions = torch.arange(64, device=device)
        pos_emb = self.position_embed(positions)  # (64, n_embd)

        # Side to move embedding (broadcast to all squares)
        side_emb = self.side_embed(side_to_move)  # (B, n_embd)
        side_emb = side_emb.unsqueeze(1)  # (B, 1, n_embd)

        # Combine embeddings
        x = piece_emb + pos_emb + side_emb  # (B, 64, n_embd)
        return x


class PieceRoutedAttention(nn.Module):
    """Multi-head attention with piece-specific sparse masks.

    Heads are allocated to piece types:
    - Knight heads: L-shape attention patterns
    - Bishop heads: diagonal attention patterns
    - Rook heads: orthogonal attention patterns
    - Queen heads: combined patterns
    - King heads: one-step patterns
    - Pawn heads: forward/capture patterns
    - Free heads: learnable attention patterns
    """
    def __init__(self, config: PRAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        # Validate head allocation
        total_allocated = (config.n_knight_heads + config.n_bishop_heads +
                          config.n_rook_heads + config.n_queen_heads +
                          config.n_king_heads + config.n_pawn_heads +
                          config.n_free_heads)
        assert total_allocated == config.n_head, f"Head allocation mismatch: {total_allocated} != {config.n_head}"

        # QKV projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Learnable attention pattern for free heads
        # Logits that will be passed through sigmoid to get soft mask
        self.free_attn_logits = nn.Parameter(torch.zeros(config.n_free_heads, 64, 64))

        # Store head allocation for indexing
        self.head_ranges = self._compute_head_ranges(config)

    def _compute_head_ranges(self, config):
        """Compute start/end indices for each head type."""
        idx = 0
        ranges = {}
        for name, count in [('knight', config.n_knight_heads),
                           ('bishop', config.n_bishop_heads),
                           ('rook', config.n_rook_heads),
                           ('queen', config.n_queen_heads),
                           ('king', config.n_king_heads),
                           ('pawn', config.n_pawn_heads),
                           ('free', config.n_free_heads)]:
            ranges[name] = (idx, idx + count)
            idx += count
        return ranges

    def _build_attention_mask(self, device, dtype):
        """Build the full attention mask for all heads.

        Returns:
            (n_head, 64, 64) attention bias tensor (0 where allowed, -inf where blocked)
        """
        n_head = self.n_head
        mask = torch.zeros(n_head, 64, 64, device=device, dtype=dtype)

        # Get piece movement masks
        knight_mask = compute_knight_mask().to(device)
        bishop_mask = compute_bishop_mask().to(device)
        rook_mask = compute_rook_mask().to(device)
        queen_mask = compute_queen_mask().to(device)
        king_mask = compute_king_mask().to(device)
        pawn_mask = compute_pawn_mask().to(device)

        # Apply piece-specific masks to corresponding heads
        def apply_mask(head_range, piece_mask):
            start, end = head_range
            for h in range(start, end):
                # 0 where allowed, -inf where not allowed
                mask[h] = torch.where(piece_mask, torch.zeros(1, device=device, dtype=dtype),
                                     torch.full((1,), float('-inf'), device=device, dtype=dtype))

        apply_mask(self.head_ranges['knight'], knight_mask)
        apply_mask(self.head_ranges['bishop'], bishop_mask)
        apply_mask(self.head_ranges['rook'], rook_mask)
        apply_mask(self.head_ranges['queen'], queen_mask)
        apply_mask(self.head_ranges['king'], king_mask)
        apply_mask(self.head_ranges['pawn'], pawn_mask)

        # Free heads use learnable soft masks
        free_start, free_end = self.head_ranges['free']
        # Sigmoid gives values in (0, 1), scale to attention weights
        # Use log-space trick: log(sigmoid(x)) for numerical stability in softmax
        free_mask_probs = torch.sigmoid(self.free_attn_logits)  # (n_free, 64, 64)
        # Convert to additive bias: log(p) where p is the attention weight multiplier
        # Small epsilon for numerical stability
        eps = 1e-6
        free_mask_bias = torch.log(free_mask_probs + eps)  # (n_free, 64, 64)
        mask[free_start:free_end] = free_mask_bias.to(dtype)

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with piece-routed attention.

        Args:
            x: (B, 64, n_embd) input tensor

        Returns:
            (B, 64, n_embd) output tensor
        """
        B, T, C = x.size()
        assert T == 64, f"Expected 64 squares, got {T}"

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, 64, head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK norm for stability
        q, k = norm(q), norm(k)

        # Build attention mask (includes piece patterns + learnable free heads)
        attn_mask = self._build_attention_mask(x.device, x.dtype)  # (n_head, 64, 64)

        # Scaled dot-product attention with mask
        # Note: we use F.scaled_dot_product_attention for efficiency but with explicit mask
        # The mask is additive (added to attention scores before softmax)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP block with relu^2 activation."""
    def __init__(self, config: PRAConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class PRABlock(nn.Module):
    """Pre-norm transformer block with piece-routed attention."""
    def __init__(self, config: PRAConfig, layer_idx: int):
        super().__init__()
        self.attn = PieceRoutedAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class PolicyHead(nn.Module):
    """Policy head for move prediction.

    Predicts (from_square, to_square) as a 64x64 distribution over moves.
    Uses a bilinear-style architecture: separate projections for "from" and "to",
    then combines them.
    """
    def __init__(self, config: PRAConfig):
        super().__init__()
        self.config = config

        # Project each square's representation for "from" and "to" roles
        self.from_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.to_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute move logits.

        Args:
            x: (B, 64, n_embd) board representation

        Returns:
            (B, 64, 64) move logits where [b, i, j] = logit for moving from i to j
        """
        # Project for from/to roles
        from_repr = self.from_proj(x)  # (B, 64, n_embd)
        to_repr = self.to_proj(x)      # (B, 64, n_embd)

        # Compute pairwise scores via dot product
        # (B, 64, n_embd) @ (B, n_embd, 64) -> (B, 64, 64)
        logits = torch.bmm(from_repr, to_repr.transpose(1, 2))

        # Scale by sqrt(d) for stability
        logits = logits / (self.config.n_embd ** 0.5)

        return logits


class ValueHead(nn.Module):
    """Value head for position evaluation.

    Predicts a scalar evaluation in [-1, 1] where:
    - +1 = white winning
    - -1 = black winning
    - 0 = equal position
    """
    def __init__(self, config: PRAConfig):
        super().__init__()
        self.config = config

        # Pool over all squares and project to scalar
        self.fc1 = nn.Linear(config.n_embd * 64, config.n_embd, bias=False)
        self.fc2 = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute position evaluation.

        Args:
            x: (B, 64, n_embd) board representation

        Returns:
            (B,) scalar evaluation in [-1, 1]
        """
        B = x.size(0)
        x = x.view(B, -1)  # (B, 64 * n_embd)
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        x = torch.tanh(x)  # Squash to [-1, 1]
        return x.squeeze(-1)


class PRA(nn.Module):
    """Piece-Routed Attention model for chess.

    Full model combining:
    - Square embedding layer
    - Stack of PRA transformer blocks
    - Policy head for move prediction
    - Value head for position evaluation
    """
    def __init__(self, config: PRAConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = SquareEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([PRABlock(config, i) for i in range(config.n_layer)])

        # Output heads
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)

    def get_device(self):
        return self.embedding.piece_embed.weight.device

    @torch.no_grad()
    def init_weights(self):
        """Initialize all model weights."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # Uniform bound for same std as normal

        # Embedding layers
        torch.nn.init.normal_(self.embedding.piece_embed.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.embedding.position_embed.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.embedding.side_embed.weight, mean=0.0, std=1.0)

        # Transformer blocks
        for block in self.blocks:
            # Attention
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # Free attention logits: init to 0 means sigmoid(0)=0.5 uniform attention
            torch.nn.init.zeros_(block.attn.free_attn_logits)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Policy head
        torch.nn.init.uniform_(self.policy_head.from_proj.weight, -s, s)
        torch.nn.init.uniform_(self.policy_head.to_proj.weight, -s, s)

        # Value head
        # Use smaller init for value head since output goes through tanh
        torch.nn.init.uniform_(self.value_head.fc1.weight, -s, s)
        torch.nn.init.normal_(self.value_head.fc2.weight, mean=0.0, std=0.01)

        # Cast embeddings to bf16 if on CUDA
        if self.embedding.piece_embed.weight.device.type == "cuda":
            self.embedding.piece_embed.to(dtype=torch.bfloat16)
            self.embedding.position_embed.to(dtype=torch.bfloat16)
            self.embedding.side_embed.to(dtype=torch.bfloat16)

    def forward(self, board: torch.Tensor, side_to_move: torch.Tensor,
                target_from: torch.Tensor = None, target_to: torch.Tensor = None,
                target_value: torch.Tensor = None,
                policy_weight: float = 1.0, value_weight: float = 0.1) -> dict:
        """Forward pass.

        Args:
            board: (B, 64) tensor of piece indices
            side_to_move: (B,) tensor of 0 (black) or 1 (white)
            target_from: (B,) tensor of source square indices (for training)
            target_to: (B,) tensor of destination square indices (for training)
            target_value: (B,) tensor of position evaluations in [-1, 1] (for training)
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss

        Returns:
            dict with 'policy_logits', 'value', and optionally 'loss', 'policy_loss', 'value_loss'
        """
        # Embed the board
        x = self.embedding(board, side_to_move)  # (B, 64, n_embd)
        x = norm(x)

        # Transform through blocks
        for block in self.blocks:
            x = block(x)
        x = norm(x)

        # Compute outputs
        policy_logits = self.policy_head(x)  # (B, 64, 64)
        value = self.value_head(x)            # (B,)

        result = {
            'policy_logits': policy_logits,
            'value': value,
        }

        # Compute loss if targets provided
        if target_from is not None and target_to is not None:
            B = board.size(0)
            # Flatten policy to (B, 4096) and compute target indices
            policy_flat = policy_logits.view(B, -1)  # (B, 4096)
            target_move = target_from * 64 + target_to  # (B,)
            policy_loss = F.cross_entropy(policy_flat, target_move)
            result['policy_loss'] = policy_loss

            if target_value is not None:
                value_loss = F.mse_loss(value, target_value)
                result['value_loss'] = value_loss
                result['loss'] = policy_weight * policy_loss + value_weight * value_loss
            else:
                result['loss'] = policy_loss

        return result

    def estimate_flops(self):
        """Estimate FLOPs per position (forward + backward)."""
        n_layer = self.config.n_layer
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        n_squares = 64

        # Per-layer attention: 4 projections (Q, K, V, out) @ (64, n_embd) x (n_embd, n_embd)
        attn_proj_flops = 4 * n_squares * n_embd * n_embd
        # Attention scores: (64, n_head, head_dim) @ (n_head, head_dim, 64)
        head_dim = n_embd // n_head
        attn_score_flops = n_head * n_squares * head_dim * n_squares * 2  # Q@K and scores@V
        # MLP: (64, n_embd) x (n_embd, 4*n_embd) + (64, 4*n_embd) x (4*n_embd, n_embd)
        mlp_flops = n_squares * n_embd * 4 * n_embd * 2

        per_layer_flops = attn_proj_flops + attn_score_flops + mlp_flops
        transformer_flops = n_layer * per_layer_flops

        # Embedding: negligible (just lookups)
        # Policy head: bilinear (64, n_embd) @ (n_embd, 64) = 64 * n_embd * 64
        policy_flops = n_squares * n_embd * n_squares * 2  # from and to projections
        # Value head: (64*n_embd, n_embd) + (n_embd, 1)
        value_flops = n_squares * n_embd * n_embd + n_embd

        total_forward = transformer_flops + policy_flops + value_flops
        # Backward is ~2x forward
        return total_forward * 3  # forward + backward

    def num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def setup_optimizer(self, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0,
                       adam_betas=(0.8, 0.95)):
        """Set up the MuonAdamW optimizer.

        Following nanochat patterns:
        - AdamW for embeddings and small params
        - Muon for matrix (2D) params in transformer blocks
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Collect parameters by type
        embedding_params = list(self.embedding.parameters())
        matrix_params = []
        free_attn_params = []

        for block in self.blocks:
            # Attention matrices
            matrix_params.extend([block.attn.c_q.weight, block.attn.c_k.weight,
                                 block.attn.c_v.weight, block.attn.c_proj.weight])
            # Free attention logits (treat as scalars/small params)
            free_attn_params.append(block.attn.free_attn_logits)
            # MLP matrices
            matrix_params.extend([block.mlp.c_fc.weight, block.mlp.c_proj.weight])

        # Head params
        head_params = [self.policy_head.from_proj.weight, self.policy_head.to_proj.weight,
                      self.value_head.fc1.weight, self.value_head.fc2.weight]

        # Scale LR for model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR for AdamW params by {dmodel_lr_scale:.6f}")

        # Build param groups
        param_groups = [
            # AdamW for embeddings
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # AdamW for free attention logits (small, scalar-like)
            dict(kind='adamw', params=free_attn_params, lr=0.1 * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]

        # Muon for matrix params grouped by shape
        all_matrix_params = matrix_params + head_params
        for shape in sorted({p.shape for p in all_matrix_params}):
            group_params = [p for p in all_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer
