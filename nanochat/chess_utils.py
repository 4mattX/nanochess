"""
Chess domain utilities for PRA (Piece-Routed Attention) model.

Provides:
- FEN parsing to board tensors
- UCI move conversion (e.g., "e2e4" <-> (12, 28))
- Precomputed 64x64 attention masks for each piece type
- Square/piece constants and encoding

Board encoding:
- 13 piece types: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
- Square indexing: a1=0, b1=1, ..., h8=63 (row-major from white's perspective)
"""

import torch
from functools import lru_cache

# Piece encoding: 0=empty, 1-6=white (PNBRQK), 7-12=black (pnbrqk)
EMPTY = 0
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = 1, 2, 3, 4, 5, 6
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = 7, 8, 9, 10, 11, 12

# Map FEN characters to piece indices
FEN_TO_PIECE = {
    'P': WHITE_PAWN, 'N': WHITE_KNIGHT, 'B': WHITE_BISHOP,
    'R': WHITE_ROOK, 'Q': WHITE_QUEEN, 'K': WHITE_KING,
    'p': BLACK_PAWN, 'n': BLACK_KNIGHT, 'b': BLACK_BISHOP,
    'r': BLACK_ROOK, 'q': BLACK_QUEEN, 'k': BLACK_KING,
}

# Map piece indices back to FEN characters
PIECE_TO_FEN = {v: k for k, v in FEN_TO_PIECE.items()}
PIECE_TO_FEN[EMPTY] = '.'

# File and rank letters for UCI conversion
FILES = 'abcdefgh'
RANKS = '12345678'

# Number of piece types including empty
NUM_PIECE_TYPES = 13


def square_to_index(file: int, rank: int) -> int:
    """Convert (file, rank) to square index. file=0-7 (a-h), rank=0-7 (1-8)."""
    return rank * 8 + file


def index_to_square(idx: int) -> tuple[int, int]:
    """Convert square index to (file, rank)."""
    return idx % 8, idx // 8


def index_to_algebraic(idx: int) -> str:
    """Convert square index to algebraic notation (e.g., 0 -> 'a1')."""
    file, rank = index_to_square(idx)
    return FILES[file] + RANKS[rank]


def algebraic_to_index(sq: str) -> int:
    """Convert algebraic notation to square index (e.g., 'a1' -> 0)."""
    file = FILES.index(sq[0])
    rank = RANKS.index(sq[1])
    return square_to_index(file, rank)


def move_to_indices(uci_move: str) -> tuple[int, int]:
    """Convert UCI move string to (from_square, to_square) indices.

    Args:
        uci_move: Move in UCI format, e.g., "e2e4" or "e7e8q" (with promotion)

    Returns:
        (from_sq, to_sq) tuple of square indices
    """
    from_sq = algebraic_to_index(uci_move[:2])
    to_sq = algebraic_to_index(uci_move[2:4])
    return from_sq, to_sq


def indices_to_move(from_sq: int, to_sq: int, promotion: str = '') -> str:
    """Convert (from_square, to_square) indices to UCI move string.

    Args:
        from_sq: Source square index
        to_sq: Destination square index
        promotion: Optional promotion piece ('q', 'r', 'b', 'n')

    Returns:
        Move in UCI format
    """
    return index_to_algebraic(from_sq) + index_to_algebraic(to_sq) + promotion


def fen_to_position(fen: str) -> tuple[torch.Tensor, bool]:
    """Parse FEN string to board tensor and side to move.

    Args:
        fen: FEN string (can be just the board part or full FEN)

    Returns:
        (board, white_to_move) where board is (64,) tensor of piece indices
    """
    parts = fen.split()
    board_fen = parts[0]
    white_to_move = parts[1] == 'w' if len(parts) > 1 else True

    board = torch.zeros(64, dtype=torch.long)

    rank = 7  # Start from rank 8 (index 7)
    file = 0

    for char in board_fen:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            sq_idx = square_to_index(file, rank)
            board[sq_idx] = FEN_TO_PIECE[char]
            file += 1

    return board, white_to_move


def position_to_fen(board: torch.Tensor, white_to_move: bool = True) -> str:
    """Convert board tensor back to FEN string (board part only).

    Args:
        board: (64,) tensor of piece indices
        white_to_move: True if white to move

    Returns:
        FEN string (board part and side to move)
    """
    fen_parts = []

    for rank in range(7, -1, -1):
        empty_count = 0
        rank_str = ''
        for file in range(8):
            sq_idx = square_to_index(file, rank)
            piece = board[sq_idx].item()
            if piece == EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += PIECE_TO_FEN[piece]
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_parts.append(rank_str)

    board_fen = '/'.join(fen_parts)
    side = 'w' if white_to_move else 'b'
    return f"{board_fen} {side}"


# ---------------------------------------------------------------------------
# Piece movement mask computation
# ---------------------------------------------------------------------------

def _compute_knight_moves(sq: int) -> list[int]:
    """Get all squares a knight can reach from sq."""
    file, rank = index_to_square(sq)
    moves = []
    deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for df, dr in deltas:
        new_file, new_rank = file + df, rank + dr
        if 0 <= new_file < 8 and 0 <= new_rank < 8:
            moves.append(square_to_index(new_file, new_rank))
    return moves


def _compute_bishop_moves(sq: int) -> list[int]:
    """Get all squares a bishop can reach from sq (sliding diagonals)."""
    file, rank = index_to_square(sq)
    moves = []
    for df, dr in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        new_file, new_rank = file + df, rank + dr
        while 0 <= new_file < 8 and 0 <= new_rank < 8:
            moves.append(square_to_index(new_file, new_rank))
            new_file += df
            new_rank += dr
    return moves


def _compute_rook_moves(sq: int) -> list[int]:
    """Get all squares a rook can reach from sq (sliding orthogonals)."""
    file, rank = index_to_square(sq)
    moves = []
    for df, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_file, new_rank = file + df, rank + dr
        while 0 <= new_file < 8 and 0 <= new_rank < 8:
            moves.append(square_to_index(new_file, new_rank))
            new_file += df
            new_rank += dr
    return moves


def _compute_queen_moves(sq: int) -> list[int]:
    """Get all squares a queen can reach from sq (rook + bishop)."""
    return _compute_rook_moves(sq) + _compute_bishop_moves(sq)


def _compute_king_moves(sq: int) -> list[int]:
    """Get all squares a king can reach from sq (one step any direction)."""
    file, rank = index_to_square(sq)
    moves = []
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            new_file, new_rank = file + df, rank + dr
            if 0 <= new_file < 8 and 0 <= new_rank < 8:
                moves.append(square_to_index(new_file, new_rank))
    return moves


def _compute_pawn_moves(sq: int, white: bool = True) -> list[int]:
    """Get all squares a pawn can reach from sq (including captures).

    Note: This includes all potential pawn moves (forward, double forward, captures)
    for attention purposes. The model will learn which are valid in context.
    """
    file, rank = index_to_square(sq)
    moves = []
    direction = 1 if white else -1
    start_rank = 1 if white else 6

    # Forward one
    new_rank = rank + direction
    if 0 <= new_rank < 8:
        moves.append(square_to_index(file, new_rank))

        # Forward two from starting rank
        if rank == start_rank:
            new_rank2 = rank + 2 * direction
            if 0 <= new_rank2 < 8:
                moves.append(square_to_index(file, new_rank2))

    # Captures (diagonal)
    for df in [-1, 1]:
        new_file = file + df
        new_rank = rank + direction
        if 0 <= new_file < 8 and 0 <= new_rank < 8:
            moves.append(square_to_index(new_file, new_rank))

    return moves


@lru_cache(maxsize=1)
def compute_knight_mask() -> torch.Tensor:
    """Compute 64x64 attention mask for knight movement patterns.

    Returns:
        Boolean tensor where mask[i, j] = True if knight on i can reach j
    """
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_knight_moves(sq):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_bishop_mask() -> torch.Tensor:
    """Compute 64x64 attention mask for bishop movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_bishop_moves(sq):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_rook_mask() -> torch.Tensor:
    """Compute 64x64 attention mask for rook movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_rook_moves(sq):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_queen_mask() -> torch.Tensor:
    """Compute 64x64 attention mask for queen movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_queen_moves(sq):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_king_mask() -> torch.Tensor:
    """Compute 64x64 attention mask for king movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_king_moves(sq):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_pawn_mask_white() -> torch.Tensor:
    """Compute 64x64 attention mask for white pawn movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_pawn_moves(sq, white=True):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_pawn_mask_black() -> torch.Tensor:
    """Compute 64x64 attention mask for black pawn movement patterns."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for sq in range(64):
        for target in _compute_pawn_moves(sq, white=False):
            mask[sq, target] = True
    return mask


@lru_cache(maxsize=1)
def compute_pawn_mask() -> torch.Tensor:
    """Compute combined 64x64 attention mask for pawn movement patterns (both colors).

    For simplicity in the PRA model, we combine white and black pawn patterns.
    The model can learn color-specific behavior from the piece embeddings.
    """
    return compute_pawn_mask_white() | compute_pawn_mask_black()


@lru_cache(maxsize=1)
def get_all_piece_masks() -> dict[str, torch.Tensor]:
    """Get all piece movement masks as a dictionary.

    Returns:
        Dict mapping piece type names to 64x64 boolean masks
    """
    return {
        'knight': compute_knight_mask(),
        'bishop': compute_bishop_mask(),
        'rook': compute_rook_mask(),
        'queen': compute_queen_mask(),
        'king': compute_king_mask(),
        'pawn': compute_pawn_mask(),
    }


def mask_to_attention_bias(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert boolean mask to attention bias (0 where allowed, -inf where blocked).

    Args:
        mask: Boolean tensor where True = allowed attention
        dtype: Output dtype

    Returns:
        Tensor with 0 where mask is True, -inf where mask is False
    """
    return torch.where(mask, torch.zeros(1, dtype=dtype), torch.full((1,), float('-inf'), dtype=dtype))


# ---------------------------------------------------------------------------
# Utility functions for evaluation
# ---------------------------------------------------------------------------

def get_piece_on_square(board: torch.Tensor, sq: int) -> int:
    """Get the piece type on a given square."""
    return board[sq].item()


def get_piece_type_name(piece: int) -> str:
    """Get human-readable piece type name."""
    names = {
        EMPTY: 'empty',
        WHITE_PAWN: 'white_pawn', BLACK_PAWN: 'black_pawn',
        WHITE_KNIGHT: 'white_knight', BLACK_KNIGHT: 'black_knight',
        WHITE_BISHOP: 'white_bishop', BLACK_BISHOP: 'black_bishop',
        WHITE_ROOK: 'white_rook', BLACK_ROOK: 'black_rook',
        WHITE_QUEEN: 'white_queen', BLACK_QUEEN: 'black_queen',
        WHITE_KING: 'white_king', BLACK_KING: 'black_king',
    }
    return names.get(piece, 'unknown')


def is_white_piece(piece: int) -> bool:
    """Check if piece is a white piece."""
    return 1 <= piece <= 6


def is_black_piece(piece: int) -> bool:
    """Check if piece is a black piece."""
    return 7 <= piece <= 12


def get_base_piece_type(piece: int) -> str:
    """Get base piece type regardless of color."""
    if piece in (WHITE_PAWN, BLACK_PAWN):
        return 'pawn'
    elif piece in (WHITE_KNIGHT, BLACK_KNIGHT):
        return 'knight'
    elif piece in (WHITE_BISHOP, BLACK_BISHOP):
        return 'bishop'
    elif piece in (WHITE_ROOK, BLACK_ROOK):
        return 'rook'
    elif piece in (WHITE_QUEEN, BLACK_QUEEN):
        return 'queen'
    elif piece in (WHITE_KING, BLACK_KING):
        return 'king'
    return 'empty'
