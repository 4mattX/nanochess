# Piece-Routed Attention (PRA) Architecture

This document provides a technical explanation of the Piece-Routed Attention architecture for chess move prediction.

## Table of Contents

1. [Core Concept](#core-concept)
2. [Attention Mask Design](#attention-mask-design)
3. [Model Components](#model-components)
4. [Forward Pass](#forward-pass)
5. [Dynamic Masking](#dynamic-masking)
6. [Interpretability](#interpretability)

---

## Core Concept

### Standard Multi-Head Attention

In standard attention, each head can attend to any position:

```text
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Piece-Routed Attention

In PRA, piece-specific heads have sparse attention masks that restrict which positions can attend to which:

```text
Attention(Q, K, V, M) = softmax(QK^T / √d_k + M) V
```

Where `M[i,j]` is:
- `0` if square j is reachable from square i by this piece type
- `-∞` otherwise

### Example: Knight Head

For a knight head, the mask for square e4 (index 28):

```text
Reachable squares: d2, f2, c3, g3, c5, g5, d6, f6

Mask visualization (from e4):
  a b c d e f g h
8 · · · · · · · ·
7 · · · · · · · ·
6 · · · ■ · ■ · ·
5 · · ■ · · · ■ ·
4 · · · · ★ · · ·
3 · · ■ · · · ■ ·
2 · · · ■ · ■ · ·
1 · · · · · · · ·
```

---

## Attention Mask Design

### Piece Movement Masks

Each piece type has a pre-computed 64×64 attention mask.

#### Knight Mask

```python
def get_knight_moves(sq):
    offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
```

#### Bishop Mask

```python
def get_bishop_moves(sq):
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
```

#### Rook Mask

```python
def get_rook_moves(sq):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
```

#### Queen Mask

Union of bishop and rook masks.

#### King Mask

Adjacent squares only (8 directions).

#### Pawn Mask

Forward diagonal captures and forward movement squares (both colors).

### Self-Attention

All masks include self-attention (`M[i,i] = 0`).

---

## Model Components

### Square Embedding (`SquareEmbedding`)

```python
class SquareEmbedding(nn.Module):
    def forward(self, board):
        piece_emb = self.piece_embedding(board)
        pos_emb = self.position_embedding(positions)
        rank_file_emb = concat(rank_emb, file_emb)

        combined = concat(piece_emb + pos_emb, rank_file_emb)
        return self.proj(combined)
```

### Piece-Routed Attention Head (`PieceRoutedAttentionHead`)

```python
class PieceRoutedAttentionHead(nn.Module):
    def __init__(self, mask):
        self.mask = mask
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn_scores = q @ k.T / sqrt(head_size)
        attn_scores = attn_scores + self.mask
        attn_weights = softmax(attn_scores)
        return attn_weights @ v
```

### Multi-Head Attention (`PieceRoutedMultiHeadAttention`)

```text
Total heads =
2 knight
2 bishop
2 rook
1 queen
1 king
1 pawn
3 free
```

```python
class PieceRoutedMultiHeadAttention(nn.Module):
    def forward(self, x):
        outputs = []

        for head in self.piece_heads:
            outputs.append(head(x))

        for head in self.free_heads:
            outputs.append(head(x))

        return self.proj(concat(outputs))
```

### PRA Block (`PRABlock`)

```text
x = x + PRA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Policy Head (`PolicyHead`)

```python
class PolicyHead(nn.Module):
    def forward(self, x):
        from_emb = self.from_proj(x)
        to_emb = self.to_proj(x)
        return bilinear(from_emb, to_emb)
```

### Value Head (`ValueHead`)

```python
class ValueHead(nn.Module):
    def forward(self, x):
        x_flat = x.view(B, 64 * d_model)
        return tanh(MLP(x_flat))
```

---

## Forward Pass

```python
def forward(board, is_white_turn):
    x = square_embedding(board)
    x = x + side_embedding(is_white_turn).unsqueeze(1)

    for block in blocks:
        x = block(x)

    x = layer_norm(x)
    move_logits = policy_head(x)
    value = value_head(x)

    return {
        "move_logits": move_logits,
        "value": value
    }
```

---

## Dynamic Masking

```python
def compute_dynamic_bishop_mask(sq, board, static_mask):
    mask = static_mask.clone()

    for ray in get_bishop_rays(sq):
        blocked = False
        for target in ray:
            if blocked:
                mask[target] = -inf
            elif board[target] != EMPTY:
                blocked = True

    return mask
```

---

## Interpretability

```python
attention = model.get_attention_maps(board, is_white_turn)

attention["block_0"]["knight_0"]
```

Each piece-specific head is structurally constrained to its movement pattern by design.
