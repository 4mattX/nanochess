"""
Prepare chess positions for PRA (Piece-Routed Attention) training.

Converts chess games to position/move pairs in parquet format:
- fen: FEN string of the position
- best_move: UCI move string (e.g., "e2e4")
- eval: Position evaluation in centipawns OR game outcome (±10000 for win/loss, 0 for draw)

Input sources:
- PGN files with or without engine annotations
- Lichess database exports (with eval comments)

Usage:
    # Use game outcomes as value targets (recommended)
    python dev/prepare_chess_positions.py --input games.pgn --use-game-outcome --output-dir ~/.cache/nanochess/pra_data

    # Extract position-specific evaluations from annotated games
    python dev/prepare_chess_positions.py --input lichess_db.pgn --with-eval --output-dir ~/.cache/nanochess/pra_data

    # Use both: position evals when available, game outcomes as fallback
    python dev/prepare_chess_positions.py --input games.pgn --with-eval --use-game-outcome --output-dir ~/.cache/nanochess/pra_data
"""

import argparse
import os
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class Position:
    """A single chess position with move and optional evaluation."""
    fen: str
    best_move: str  # UCI format
    eval_cp: Optional[int] = None  # centipawns, None if not available


def extract_result_from_headers(headers: list[str]) -> Optional[str]:
    """Extract game result from PGN headers.

    Args:
        headers: List of PGN header lines

    Returns:
        Result string: "1-0", "0-1", "1/2-1/2", or None
    """
    for header in headers:
        if header.startswith('[Result'):
            match = re.search(r'\[Result\s+"([^"]+)"\]', header)
            if match:
                return match.group(1)
    return None


def game_result_to_value(result: str, white_to_move: bool) -> Optional[int]:
    """Convert game result to centipawn value from current player's perspective.

    Args:
        result: Game result ("1-0", "0-1", "1/2-1/2", "*")
        white_to_move: Whether it's white's turn in this position

    Returns:
        Value in centipawns: +10000 for win, -10000 for loss, 0 for draw, None for unknown
    """
    if result == "*":
        return None  # Unknown outcome

    # Determine outcome from white's perspective
    if result == "1-0":
        white_value = 10000  # White wins
    elif result == "0-1":
        white_value = -10000  # Black wins
    elif result == "1/2-1/2":
        white_value = 0  # Draw
    else:
        return None

    # Convert to current player's perspective
    return white_value if white_to_move else -white_value


def parse_pgn_games(pgn_path: str) -> Iterator[tuple[list[str], list[str]]]:
    """Parse a PGN file and yield (headers, moves) for each game.

    Args:
        pgn_path: Path to PGN file

    Yields:
        (headers, move_tokens) where headers is list of "[Tag value]" strings
        and move_tokens is list of move/comment tokens
    """
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split into games (each game starts with [Event)
    games = re.split(r'\n\n(?=\[Event)', content)

    for game in games:
        game = game.strip()
        if not game:
            continue

        lines = game.split('\n')
        headers = []
        moves_lines = []
        in_moves = False

        for line in lines:
            line = line.strip()
            if line.startswith('['):
                headers.append(line)
            elif line:
                in_moves = True
                moves_lines.append(line)

        if moves_lines:
            moves_text = ' '.join(moves_lines)
            # Tokenize moves and comments
            tokens = re.findall(r'\{[^}]*\}|[^\s{}]+', moves_text)
            yield headers, tokens


def san_to_uci(san: str, board) -> Optional[str]:
    """Convert SAN move to UCI format using python-chess.

    Args:
        san: Move in Standard Algebraic Notation (e.g., "e4", "Nf3")
        board: python-chess Board object

    Returns:
        UCI move string or None if invalid
    """
    try:
        import chess
        move = board.parse_san(san)
        return move.uci()
    except:
        return None


def extract_eval_from_comment(comment: str) -> Optional[int]:
    """Extract centipawn evaluation from a comment.

    Supports formats:
    - Lichess: {[%eval 0.35]} or {[%eval #5]} (mate in 5)
    - Stockfish: {+0.35/20} or {-1.50/25}

    Args:
        comment: Comment string (with braces)

    Returns:
        Evaluation in centipawns or None
    """
    # Lichess format: [%eval X.XX] or [%eval #N]
    match = re.search(r'\[%eval\s+([+-]?\d+\.?\d*)\]', comment)
    if match:
        return int(float(match.group(1)) * 100)

    # Mate score: [%eval #N] -> large value
    match = re.search(r'\[%eval\s+#([+-]?\d+)\]', comment)
    if match:
        mate_in = int(match.group(1))
        # Return large value with sign
        return 10000 if mate_in > 0 else -10000

    # Stockfish format: +X.XX/depth or -X.XX/depth
    match = re.search(r'([+-]?\d+\.?\d*)/\d+', comment)
    if match:
        return int(float(match.group(1)) * 100)

    return None


def extract_positions_from_game(headers: list[str], tokens: list[str],
                                with_eval: bool = False,
                                use_game_outcome: bool = False,
                                min_ply: int = 10,
                                max_ply: int = 200) -> Iterator[Position]:
    """Extract positions from a parsed game.

    Args:
        headers: PGN headers
        tokens: Move/comment tokens
        with_eval: Whether to extract evaluations from comments
        use_game_outcome: Whether to use game outcome as eval (win=+1, draw=0, loss=-1)
        min_ply: Minimum ply (half-move) to start extracting
        max_ply: Maximum ply to extract

    Yields:
        Position objects
    """
    try:
        import chess
    except ImportError:
        raise ImportError("python-chess required: pip install chess")

    # Extract game result for outcome-based evaluation
    game_result = extract_result_from_headers(headers) if use_game_outcome else None

    board = chess.Board()
    ply = 0
    pending_eval = None

    for token in tokens:
        # Skip move numbers and results
        if re.match(r'^\d+\.+$', token):
            continue
        if token in ('1-0', '0-1', '1/2-1/2', '*'):
            continue

        # Handle comments (may contain eval)
        if token.startswith('{'):
            if with_eval:
                pending_eval = extract_eval_from_comment(token)
            continue

        # Try to parse as a move
        uci = san_to_uci(token, board)
        if uci is None:
            continue

        # Extract position if within ply range
        if min_ply <= ply <= max_ply:
            fen = board.fen()

            # Determine evaluation: prefer position-specific eval, fallback to game outcome
            eval_cp = pending_eval
            if eval_cp is None and game_result:
                eval_cp = game_result_to_value(game_result, board.turn)

            yield Position(fen=fen, best_move=uci, eval_cp=eval_cp)

        # Make the move
        try:
            board.push_san(token)
            ply += 1
            pending_eval = None
        except:
            break

        if ply > max_ply:
            break


def process_pgn_file(pgn_path: str, with_eval: bool = False,
                    use_game_outcome: bool = False,
                    min_ply: int = 10, max_ply: int = 200,
                    max_positions: int = -1,
                    sample_rate: float = 1.0) -> list[Position]:
    """Process a PGN file and extract positions.

    Args:
        pgn_path: Path to PGN file
        with_eval: Whether to extract evaluations from comments
        use_game_outcome: Whether to use game outcome as eval
        min_ply: Minimum ply to extract
        max_ply: Maximum ply to extract
        max_positions: Maximum positions to extract (-1 = unlimited)
        sample_rate: Fraction of positions to sample (0-1)

    Returns:
        List of Position objects
    """
    positions = []
    games_processed = 0

    print(f"Processing {pgn_path}...")

    for headers, tokens in parse_pgn_games(pgn_path):
        for pos in extract_positions_from_game(headers, tokens, with_eval, use_game_outcome, min_ply, max_ply):
            # Random sampling
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue

            positions.append(pos)

            if max_positions > 0 and len(positions) >= max_positions:
                print(f"  Reached max positions: {max_positions}")
                return positions

        games_processed += 1
        if games_processed % 10000 == 0:
            print(f"  Processed {games_processed} games, {len(positions)} positions")

    print(f"  Total: {games_processed} games, {len(positions)} positions")
    return positions


def positions_to_parquet(positions: list[Position], output_path: str):
    """Save positions to a parquet file.

    Args:
        positions: List of Position objects
        output_path: Output parquet file path
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required: pip install pyarrow")

    # Build columns
    fens = [p.fen for p in positions]
    moves = [p.best_move for p in positions]
    evals = [p.eval_cp if p.eval_cp is not None else 0 for p in positions]

    table = pa.table({
        'fen': fens,
        'best_move': moves,
        'eval': evals,
    })

    pq.write_table(table, output_path, compression='snappy')
    print(f"Wrote {len(positions)} positions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare chess positions for PRA training")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input PGN file or directory of PGN files")
    parser.add_argument("--output-dir", "-o", type=str,
                       default=os.path.expanduser("~/.cache/nanochat/pra_data"),
                       help="Output directory for parquet files")
    parser.add_argument("--with-eval", action="store_true",
                       help="Extract evaluations from comments (Lichess format)")
    parser.add_argument("--use-game-outcome", action="store_true",
                       help="Use game outcome (win/draw/loss) as evaluation signal")
    parser.add_argument("--min-ply", type=int, default=10,
                       help="Minimum ply to start extracting (default: 10)")
    parser.add_argument("--max-ply", type=int, default=200,
                       help="Maximum ply to extract (default: 200)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--max-positions", type=int, default=-1,
                       help="Maximum positions to extract (-1 = all)")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                       help="Fraction of positions to sample (default: 1.0)")
    parser.add_argument("--positions-per-shard", type=int, default=100000,
                       help="Positions per parquet shard (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all PGN files
    input_path = Path(args.input)
    if input_path.is_file():
        pgn_files = [input_path]
    else:
        pgn_files = list(input_path.glob("**/*.pgn"))

    print(f"Found {len(pgn_files)} PGN file(s)")

    # Extract positions from all files
    all_positions = []
    for pgn_file in pgn_files:
        remaining = args.max_positions - len(all_positions) if args.max_positions > 0 else -1
        positions = process_pgn_file(
            str(pgn_file),
            with_eval=args.with_eval,
            use_game_outcome=args.use_game_outcome,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            max_positions=remaining,
            sample_rate=args.sample_rate,
        )
        all_positions.extend(positions)

        if args.max_positions > 0 and len(all_positions) >= args.max_positions:
            break

    print(f"\nTotal positions collected: {len(all_positions)}")

    if len(all_positions) == 0:
        print("No positions found!")
        return

    # Shuffle
    print("Shuffling positions...")
    random.shuffle(all_positions)

    # Split into train/val
    val_size = int(len(all_positions) * args.val_ratio)
    val_positions = all_positions[:val_size]
    train_positions = all_positions[val_size:]

    print(f"Train positions: {len(train_positions)}, Val positions: {len(val_positions)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Clear existing parquet files
    for f in Path(args.output_dir).glob("shard_*.parquet"):
        f.unlink()
        print(f"Removed old file: {f}")

    # Write train shards
    shard_idx = 0
    for i in range(0, len(train_positions), args.positions_per_shard):
        shard_positions = train_positions[i:i + args.positions_per_shard]
        output_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
        positions_to_parquet(shard_positions, output_path)
        shard_idx += 1

    # Write val shard (MUST be last file alphabetically for dataloader)
    val_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
    positions_to_parquet(val_positions, val_path)

    print(f"\nDone! Created {shard_idx + 1} shard(s) in {args.output_dir}")
    print(f"Train shards: shard_00000.parquet to shard_{shard_idx-1:05d}.parquet")
    print(f"Val shard: shard_{shard_idx:05d}.parquet")

    # Print eval stats if available
    evals_present = sum(1 for p in all_positions if p.eval_cp is not None)
    print(f"\nPositions with eval: {evals_present} ({100*evals_present/len(all_positions):.1f}%)")


if __name__ == "__main__":
    main()
