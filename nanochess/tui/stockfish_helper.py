"""Optional Stockfish integration for chess evaluation."""

import subprocess
from typing import Optional
import chess


class StockfishHelper:
    """Helper class to interface with Stockfish engine."""

    def __init__(self, stockfish_path: str = "stockfish"):
        """Initialize Stockfish helper.

        Args:
            stockfish_path: Path to stockfish binary (default: "stockfish" from PATH)
        """
        self.stockfish_path = stockfish_path
        self.process: Optional[subprocess.Popen] = None
        self.available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Stockfish is available."""
        try:
            result = subprocess.run(
                [self.stockfish_path, "--help"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start(self) -> bool:
        """Start Stockfish engine.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.available:
            return False

        try:
            self.process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Initialize UCI mode
            self._send_command("uci")
            self._wait_for("uciok")
            self._send_command("isready")
            self._wait_for("readyok")

            return True
        except Exception:
            self.process = None
            return False

    def stop(self) -> None:
        """Stop Stockfish engine."""
        if self.process:
            try:
                self._send_command("quit")
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None

    def _send_command(self, command: str) -> None:
        """Send a command to Stockfish."""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _wait_for(self, expected: str, timeout: float = 5.0) -> None:
        """Wait for expected output from Stockfish."""
        if not self.process or not self.process.stdout:
            return

        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if expected in line:
                return

    def evaluate_position(self, board: chess.Board, depth: int = 15) -> Optional[float]:
        """Evaluate a chess position using Stockfish.

        Args:
            board: Chess board position
            depth: Search depth (default: 15)

        Returns:
            Evaluation in pawns (positive = white advantage), or None if unavailable
        """
        if not self.process:
            if not self.start():
                return None

        try:
            # Set up position
            fen = board.fen()
            self._send_command(f"position fen {fen}")

            # Analyze
            self._send_command(f"go depth {depth}")

            # Read evaluation
            eval_cp = None
            eval_mate = None

            while True:
                if not self.process or not self.process.stdout:
                    return None

                line = self.process.stdout.readline().strip()

                if "bestmove" in line:
                    break

                if "score cp" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "cp" and i + 1 < len(parts):
                            eval_cp = int(parts[i + 1])
                            break

                if "score mate" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "mate" and i + 1 < len(parts):
                            eval_mate = int(parts[i + 1])
                            break

            # Convert to pawns
            if eval_mate is not None:
                # Mate in N: large eval with sign
                return 100.0 if eval_mate > 0 else -100.0
            elif eval_cp is not None:
                # Centipawns to pawns
                return eval_cp / 100.0

            return None

        except Exception:
            return None

    def get_best_move(self, board: chess.Board, depth: int = 15) -> Optional[tuple[chess.Move, int]]:
        """Get the best move and evaluation for a position.

        Args:
            board: Chess board position
            depth: Search depth (default: 15)

        Returns:
            Tuple of (best_move, eval_cp) or None if unavailable
            eval_cp is in centipawns from white's perspective
        """
        if not self.process:
            if not self.start():
                return None

        try:
            # Set up position
            fen = board.fen()
            self._send_command(f"position fen {fen}")

            # Analyze
            self._send_command(f"go depth {depth}")

            # Read evaluation and best move
            eval_cp = None
            eval_mate = None
            best_move_str = None

            while True:
                if not self.process or not self.process.stdout:
                    return None

                line = self.process.stdout.readline().strip()

                if "bestmove" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move_str = parts[1]
                    break

                if "score cp" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "cp" and i + 1 < len(parts):
                            eval_cp = int(parts[i + 1])
                            break

                if "score mate" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "mate" and i + 1 < len(parts):
                            eval_mate = int(parts[i + 1])
                            break

            if not best_move_str:
                return None

            # Parse move
            try:
                best_move = chess.Move.from_uci(best_move_str)
            except ValueError:
                return None

            # Convert evaluation to centipawns
            if eval_mate is not None:
                # Mate in N: use large value with sign
                eval_cp = 10000 if eval_mate > 0 else -10000
            elif eval_cp is None:
                eval_cp = 0

            return (best_move, eval_cp)

        except Exception:
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def get_move_with_jitter(
        self,
        board: chess.Board,
        min_depth: int = 8,
        max_depth: int = 15,
        temperature: float = 0.2,
    ) -> Optional[chess.Move]:
        """Get a move from Stockfish with randomness to avoid repetitive games.

        The jitter is achieved by:
        1. Randomly varying the search depth between min_depth and max_depth
        2. Optionally selecting from top moves with temperature-based sampling

        Args:
            board: Chess board position
            min_depth: Minimum search depth (default: 8)
            max_depth: Maximum search depth (default: 15)
            temperature: Randomness in move selection (0 = deterministic, higher = more random)

        Returns:
            A chess move, or None if unavailable
        """
        import random

        if not self.process:
            if not self.start():
                return None

        # Randomly vary depth for non-deterministic search
        depth = random.randint(min_depth, max_depth)

        try:
            # Set up position
            fen = board.fen()
            self._send_command(f"position fen {fen}")

            # Use MultiPV to get top 3 moves for more variety
            self._send_command("setoption name MultiPV value 3")

            # Analyze
            self._send_command(f"go depth {depth}")

            # Read all candidate moves and their evaluations
            candidates = []  # List of (move_str, eval_cp)

            while True:
                if not self.process or not self.process.stdout:
                    return None

                line = self.process.stdout.readline().strip()

                if "bestmove" in line:
                    break

                # Parse each candidate move line
                if "multipv" in line and "score cp" in line:
                    parts = line.split()
                    move_str = None
                    eval_cp = None

                    # Find the move (after "pv")
                    for i, part in enumerate(parts):
                        if part == "pv" and i + 1 < len(parts):
                            move_str = parts[i + 1]
                            break

                    # Find the eval
                    for i, part in enumerate(parts):
                        if part == "cp" and i + 1 < len(parts):
                            eval_cp = int(parts[i + 1])
                            break
                        elif part == "mate" and i + 1 < len(parts):
                            mate_in = int(parts[i + 1])
                            eval_cp = 10000 if mate_in > 0 else -10000
                            break

                    if move_str and eval_cp is not None:
                        candidates.append((move_str, eval_cp))

            # Reset MultiPV to 1
            self._send_command("setoption name MultiPV value 1")

            if not candidates:
                return None

            # If temperature is very low, just pick the best move
            if temperature < 0.01:
                best_move_str = candidates[0][0]
                try:
                    return chess.Move.from_uci(best_move_str)
                except ValueError:
                    return None

            # Apply temperature-based sampling to move selection
            # Convert evals to probabilities using softmax with temperature
            import math

            # Get perspective-correct evaluations (current player wants high eval)
            evals = [cp if board.turn == chess.WHITE else -cp for _, cp in candidates]

            # Apply temperature and compute softmax
            scaled_evals = [e / (temperature * 100) for e in evals]
            max_eval = max(scaled_evals)
            exp_evals = [math.exp(e - max_eval) for e in scaled_evals]
            sum_exp = sum(exp_evals)
            probs = [e / sum_exp for e in exp_evals]

            # Sample from the distribution
            r = random.random()
            cumsum = 0.0
            selected_move_str = candidates[0][0]  # Default to best

            for (move_str, _), prob in zip(candidates, probs):
                cumsum += prob
                if r <= cumsum:
                    selected_move_str = move_str
                    break

            try:
                return chess.Move.from_uci(selected_move_str)
            except ValueError:
                return None

        except Exception:
            # Reset MultiPV on error
            try:
                self._send_command("setoption name MultiPV value 1")
            except:
                pass
            return None


def classify_move(centipawn_loss: int) -> tuple[str, str]:
    """Classify a move based on centipawn loss.

    Similar to chess.com's move classification system.

    Args:
        centipawn_loss: Absolute centipawn loss (positive value)

    Returns:
        Tuple of (classification, symbol)
        - classification: "Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"
        - symbol: Unicode symbol for the classification
    """
    if centipawn_loss <= 10:
        return ("Best", "✓")
    elif centipawn_loss <= 25:
        return ("Excellent", "✓")
    elif centipawn_loss <= 50:
        return ("Good", "•")
    elif centipawn_loss <= 100:
        return ("Inaccuracy", "?!")
    elif centipawn_loss <= 200:
        return ("Mistake", "?")
    else:
        return ("Blunder", "??")


def calculate_move_accuracy(centipawn_loss: int) -> float:
    """Calculate move accuracy percentage from centipawn loss.

    Uses a decay function similar to chess.com's accuracy calculation.

    Args:
        centipawn_loss: Absolute centipawn loss

    Returns:
        Accuracy as a percentage (0-100)
    """
    import math

    # Use exponential decay: accuracy = 100 * exp(-loss / scale)
    # Scale factor determines how quickly accuracy drops
    scale = 100.0

    accuracy = 100.0 * math.exp(-centipawn_loss / scale)
    return max(0.0, min(100.0, accuracy))


def calculate_game_accuracy(centipawn_losses: list[int]) -> float:
    """Calculate overall game accuracy from list of centipawn losses.

    Args:
        centipawn_losses: List of centipawn loss for each move

    Returns:
        Overall accuracy as a percentage (0-100)
    """
    if not centipawn_losses:
        return 100.0

    move_accuracies = [calculate_move_accuracy(loss) for loss in centipawn_losses]
    return sum(move_accuracies) / len(move_accuracies)


def estimate_elo_from_accuracy(accuracy: float) -> int:
    """Estimate Elo rating from move accuracy.

    This is a rough heuristic based on typical chess engine performance.

    Args:
        accuracy: Top-1 move accuracy (0.0 to 1.0)

    Returns:
        Estimated Elo rating
    """
    if accuracy < 0.1:
        return 800  # Beginner
    elif accuracy < 0.2:
        return 1000  # Casual player
    elif accuracy < 0.3:
        return 1200  # Club player
    elif accuracy < 0.4:
        return 1500  # Intermediate
    elif accuracy < 0.5:
        return 1800  # Advanced
    elif accuracy < 0.6:
        return 2000  # Expert
    elif accuracy < 0.7:
        return 2200  # Master
    else:
        return 2400 + int((accuracy - 0.7) * 1000)  # Grandmaster+


def estimate_elo_from_game_accuracy(game_accuracy: float) -> int:
    """Estimate Elo rating from game accuracy percentage.

    Args:
        game_accuracy: Game accuracy percentage (0-100)

    Returns:
        Estimated Elo rating
    """
    # Rough mapping based on typical player accuracy
    if game_accuracy < 40:
        return 800  # Beginner
    elif game_accuracy < 50:
        return 1000  # Novice
    elif game_accuracy < 60:
        return 1200  # Intermediate
    elif game_accuracy < 70:
        return 1500  # Club player
    elif game_accuracy < 75:
        return 1700  # Advanced
    elif game_accuracy < 80:
        return 1900  # Expert
    elif game_accuracy < 85:
        return 2100  # Master
    elif game_accuracy < 90:
        return 2300  # International Master
    elif game_accuracy < 95:
        return 2500  # Grandmaster
    else:
        return 2700  # Super GM
