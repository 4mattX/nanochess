"""Interactive play screen for Nanochess TUI - Watch model play chess."""

import asyncio
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import chess
import chess.pgn
import torch
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, Static, Select, Checkbox

from nanochat.pra import PRA, PRAConfig
from nanochat.chess_utils import fen_to_position, move_to_indices, index_to_algebraic
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir
from nanochess.tui.stockfish_helper import (
    StockfishHelper,
    classify_move,
    calculate_game_accuracy,
    estimate_elo_from_game_accuracy,
)


class ChessBoardWidget(Static):
    """Widget to display a chess board."""

    board_state = reactive(chess.Board())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "chess-board"

    def render(self) -> str:
        """Render the chess board as ASCII art."""
        board = self.board_state
        lines = []

        # Top border
        lines.append("  ┌─────────────────┐")

        # Board squares (from rank 8 to 1)
        for rank in range(7, -1, -1):
            rank_str = f"{rank + 1} │ "
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)

                if piece is None:
                    # Empty square - use checkerboard pattern
                    if (rank + file) % 2 == 0:
                        rank_str += "· "
                    else:
                        rank_str += "  "
                else:
                    # Piece symbols
                    symbol = piece.unicode_symbol()
                    rank_str += f"{symbol} "

            rank_str += "│"
            lines.append(rank_str)

        # Bottom border and file labels
        lines.append("  └─────────────────┘")
        lines.append("    a b c d e f g h")

        return "\n".join(lines)

    def watch_board_state(self, new_board: chess.Board) -> None:
        """Called when board_state changes."""
        self.refresh()


class MoveHistoryWidget(VerticalScroll):
    """Widget to display move history."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "move-history"
        self.moves = []

    def add_move(
        self,
        move_num: int,
        move: chess.Move,
        san: str,
        classification: str = None,
        symbol: str = None,
        cp_loss: int = None,
    ) -> None:
        """Add a move to the history with optional analysis.

        Args:
            move_num: Move number
            move: Chess move object
            san: Move in standard algebraic notation
            classification: Move classification (Best, Good, Inaccuracy, etc.)
            symbol: Symbol for the classification (✓, ?, ??, etc.)
            cp_loss: Centipawn loss for this move
        """
        # Format move text with classification
        if classification and symbol:
            if cp_loss is not None:
                move_text = f"{move_num}. {san} {symbol} ({classification}, -{cp_loss}cp)"
            else:
                move_text = f"{move_num}. {san} {symbol} ({classification})"

            # Color-code based on classification
            if classification in ["Best", "Excellent"]:
                move_text = f"[$success]{move_text}[/]"
            elif classification == "Good":
                move_text = f"[$primary]{move_text}[/]"
            elif classification == "Inaccuracy":
                move_text = f"[$warning]{move_text}[/]"
            elif classification in ["Mistake", "Blunder"]:
                move_text = f"[$error]{move_text}[/]"
        else:
            move_text = f"{move_num}. {san}"

        move_label = Label(move_text)
        move_label.add_class("move-entry")
        self.mount(move_label)
        self.scroll_end(animate=False)


class StatsWidget(Static):
    """Widget to display game statistics."""

    model_eval = reactive(0.0)
    stockfish_eval = reactive(None)
    move_count = reactive(0)
    status_text = reactive("Ready")
    game_accuracy = reactive(None)
    estimated_rating = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "stats-panel"

    def render(self) -> str:
        """Render statistics."""
        lines = ["[$accent]STATISTICS[/]", ""]
        lines.append(f"Move: {self.move_count}")
        lines.append(f"Model eval: {self.model_eval:+.2f}")

        if self.stockfish_eval is not None:
            lines.append(f"Stockfish: {self.stockfish_eval:+.2f}")
        else:
            lines.append("Stockfish: N/A")

        lines.append("")

        if self.game_accuracy is not None:
            lines.append(f"Game Accuracy: {self.game_accuracy:.1f}%")

            # Color-code accuracy
            if self.game_accuracy >= 90:
                accuracy_color = "success"
            elif self.game_accuracy >= 70:
                accuracy_color = "primary"
            elif self.game_accuracy >= 50:
                accuracy_color = "warning"
            else:
                accuracy_color = "error"

            lines[-1] = f"[${accuracy_color}]{lines[-1]}[/]"
        else:
            lines.append("Game Accuracy: N/A")

        if self.estimated_rating is not None:
            lines.append(f"Est. Rating: ~{self.estimated_rating}")
        else:
            lines.append("Est. Rating: N/A")

        lines.append("")
        lines.append(f"Status: {self.status_text}")

        return "\n".join(lines)


class InteractiveScreen(Screen):
    """Interactive play screen - watch the model play chess."""

    CSS = """
    #game-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }

    #left-panel {
        width: 1fr;
        height: 100%;
        padding: 1;
    }

    #right-panel {
        width: 2fr;
        height: 100%;
        padding: 1;
    }

    #chess-board {
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
        content-align: center middle;
    }

    #controls {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    #control-buttons {
        height: auto;
        align: center middle;
    }

    #control-buttons Button {
        margin: 0 1;
    }

    #stats-panel {
        border: solid $primary;
        padding: 1;
        height: auto;
    }

    #move-history {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    .move-entry {
        padding: 0 1;
    }

    #model-selector {
        width: 100%;
        margin-bottom: 1;
    }

    #right-panel > Horizontal {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    #right-panel > Horizontal Label {
        width: 1fr;
    }

    #copy-pgn {
        width: auto;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("space", "toggle_play", "Play/Pause", show=True),
        Binding("r", "reset", "Reset", show=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.board = chess.Board()
        self.model: Optional[PRA] = None
        self.device = None
        self.playing = False
        self.game_speed = 1.0  # seconds between moves
        self.move_count = 0
        self.stockfish: Optional[StockfishHelper] = None
        self.use_stockfish_eval = False  # For evaluation/analysis
        self.opponent_mode = "self-play"  # "self-play" or "stockfish"
        self.centipawn_losses = []  # Track CP loss for each move

    def compose(self) -> ComposeResult:
        """Compose the interactive screen."""
        with Container(id="game-container"):
            # Left panel: Board and controls
            with Vertical(id="left-panel"):
                yield ChessBoardWidget()

                with Container(id="controls"):
                    yield Label("[$accent]CONTROLS[/]")

                    # Model selector
                    yield Select(
                        [(name, name) for name in self._find_checkpoints()],
                        prompt="Select model...",
                        id="model-selector",
                    )

                    # Opponent selector
                    yield Label("Opponent:")
                    yield Select(
                        [
                            ("Self-play (model vs itself)", "self-play"),
                            ("Stockfish (stockfish plays white)", "stockfish"),
                        ],
                        value="self-play",
                        id="opponent-selector",
                    )

                    with Horizontal(id="control-buttons"):
                        yield Button("Load Model", id="load-model", variant="primary")
                        yield Button("Start", id="start-game", variant="success")
                        yield Button("Pause", id="pause-game", variant="warning")
                        yield Button("Reset", id="reset-game", variant="error")

                    yield Checkbox("Enable move analysis", id="stockfish-checkbox")

                yield StatsWidget()

            # Right panel: Move history
            with Vertical(id="right-panel"):
                with Horizontal():
                    yield Label("[$accent]MOVE HISTORY[/]")
                    yield Button("Copy PGN", id="copy-pgn", variant="default")
                yield MoveHistoryWidget()

        yield Footer()

    def _find_checkpoints(self) -> list[str]:
        """Find available model checkpoints."""
        base_dir = get_base_dir()
        checkpoint_base = os.path.join(base_dir, "pra_checkpoints")

        if not os.path.exists(checkpoint_base):
            return ["No checkpoints found"]

        checkpoints = []
        for item in os.listdir(checkpoint_base):
            item_path = os.path.join(checkpoint_base, item)
            if os.path.isdir(item_path):
                # Find latest checkpoint in this directory
                import glob
                pattern = os.path.join(item_path, "model_*.pt")
                files = glob.glob(pattern)
                if files:
                    steps = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
                    max_step = max(steps)
                    checkpoints.append(f"{item} (step {max_step})")

        return checkpoints if checkpoints else ["No checkpoints found"]

    @on(Select.Changed, "#opponent-selector")
    def opponent_changed(self, event: Select.Changed) -> None:
        """Handle opponent selection change."""
        self.opponent_mode = str(event.value)

        if self.opponent_mode == "stockfish":
            # Initialize Stockfish if needed
            if self.stockfish is None:
                self.stockfish = StockfishHelper()

            if not self.stockfish.available:
                self.query_one(StatsWidget).status_text = "Stockfish not found!"
                # Reset to self-play
                self.opponent_mode = "self-play"
                event.select.value = "self-play"
            else:
                self.query_one(StatsWidget).status_text = "Opponent: Stockfish"
        else:
            self.query_one(StatsWidget).status_text = "Opponent: Self-play"

    @on(Checkbox.Changed, "#stockfish-checkbox")
    def stockfish_toggled(self, event: Checkbox.Changed) -> None:
        """Handle Stockfish analysis checkbox toggle."""
        self.use_stockfish_eval = event.value

        if self.use_stockfish_eval:
            # Try to initialize Stockfish
            if self.stockfish is None:
                self.stockfish = StockfishHelper()

            if self.stockfish.available:
                self.query_one(StatsWidget).status_text = "Move analysis enabled"
            else:
                self.use_stockfish_eval = False
                event.checkbox.value = False
                self.query_one(StatsWidget).status_text = "Stockfish not found!"
        else:
            self.query_one(StatsWidget).status_text = "Move analysis disabled"

    @on(Button.Pressed, "#load-model")
    async def load_model_clicked(self) -> None:
        """Load the selected model."""
        select_widget = self.query_one("#model-selector", Select)
        selected = select_widget.value

        if not selected or selected == "No checkpoints found":
            self.query_one(StatsWidget).status_text = "No model selected"
            return

        # Parse checkpoint name and step
        parts = selected.split(" (step ")
        checkpoint_name = parts[0]
        step = int(parts[1].rstrip(")"))

        self.query_one(StatsWidget).status_text = "Loading model..."
        await self._load_model(checkpoint_name, step)

    async def _load_model(self, checkpoint_name: str, step: int) -> None:
        """Load a model from checkpoint."""
        try:
            base_dir = get_base_dir()
            checkpoint_dir = os.path.join(base_dir, "pra_checkpoints", checkpoint_name)

            # Detect device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            # Load checkpoint
            model_data, _, meta_data = load_checkpoint(
                checkpoint_dir, step, self.device, load_optimizer=False
            )

            # Handle torch.compile prefix
            model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

            # Build model
            model_config_kwargs = meta_data["model_config"]
            model_config = PRAConfig(**model_config_kwargs)

            with torch.device("meta"):
                self.model = PRA(model_config)
            self.model.to_empty(device=self.device)
            self.model.init_weights()
            self.model.load_state_dict(model_data, strict=True, assign=True)
            self.model.eval()

            # Convert to float32 for inference (avoid dtype mismatches)
            self.model = self.model.float()

            self.query_one(StatsWidget).status_text = f"Loaded {checkpoint_name} (step {step})"
        except Exception as e:
            self.query_one(StatsWidget).status_text = f"Error: {str(e)}"

    @on(Button.Pressed, "#start-game")
    def start_game_clicked(self) -> None:
        """Start the game."""
        if self.model is None:
            self.query_one(StatsWidget).status_text = "Load a model first!"
            return

        if not self.playing:
            self.playing = True
            self.run_game()

    @on(Button.Pressed, "#pause-game")
    def pause_game_clicked(self) -> None:
        """Pause the game."""
        self.playing = False
        self.query_one(StatsWidget).status_text = "Paused"

    @on(Button.Pressed, "#reset-game")
    def reset_game_clicked(self) -> None:
        """Reset the game."""
        self.playing = False
        self.board.reset()
        self.move_count = 0
        self.centipawn_losses = []
        self.query_one(ChessBoardWidget).board_state = self.board.copy()

        stats = self.query_one(StatsWidget)
        stats.move_count = 0
        stats.model_eval = 0.0
        stats.game_accuracy = None
        stats.estimated_rating = None
        stats.status_text = "Reset"

        # Clear move history
        history = self.query_one(MoveHistoryWidget)
        history.remove_children()

    @work(exclusive=True)
    async def run_game(self) -> None:
        """Run the game loop."""
        stats = self.query_one(StatsWidget)
        board_widget = self.query_one(ChessBoardWidget)
        history_widget = self.query_one(MoveHistoryWidget)

        while self.playing and not self.board.is_game_over():
            # Determine whose turn it is
            is_white_turn = self.board.turn == chess.WHITE

            # Check if it's Stockfish's turn (Stockfish plays white when opponent mode)
            is_stockfish_turn = (
                self.opponent_mode == "stockfish" and is_white_turn
            )

            # Get Stockfish best move BEFORE model moves (for analysis if enabled)
            stockfish_best = None
            eval_before = None
            if self.use_stockfish_eval and self.stockfish and not is_stockfish_turn:
                stockfish_result = self.stockfish.get_best_move(self.board, depth=15)
                if stockfish_result:
                    stockfish_best, eval_before = stockfish_result

            best_move = None
            value = 0.0

            if is_stockfish_turn:
                stats.status_text = "Stockfish thinking..."

                # First move is always random
                if len(self.board.move_stack) == 0:
                    import random
                    legal_moves = list(self.board.legal_moves)
                    best_move = random.choice(legal_moves)
                else:
                    # Subsequent moves: play at full strength (depth 20)
                    stockfish_result = self.stockfish.get_best_move(self.board, depth=20)

                    if stockfish_result:
                        best_move, _ = stockfish_result
                    else:
                        # Fallback to random legal move
                        import random
                        legal_moves = list(self.board.legal_moves)
                        if not legal_moves:
                            break
                        best_move = random.choice(legal_moves)

                # For display purposes, estimate eval
                if self.use_stockfish_eval and self.stockfish:
                    eval_result = self.stockfish.evaluate_position(self.board, depth=10)
                    if eval_result is not None:
                        value = eval_result
                        stats.stockfish_eval = eval_result
            else:
                # Model plays
                stats.status_text = "Model thinking..."

                # Get current position
                fen = self.board.fen()
                board_tensor, is_white = fen_to_position(fen)
                board_tensor = board_tensor.unsqueeze(0).to(self.device)
                side_tensor = torch.tensor([1 if is_white else 0], device=self.device)

                # Get model prediction
                with torch.no_grad():
                    result = self.model(board_tensor, side_tensor)
                    move_logits = result['policy_logits'].view(-1)  # (4096,)
                    value = result['value'].item()

                # Get legal moves and their logits
                legal_moves = list(self.board.legal_moves)
                if not legal_moves:
                    break

                # Find best legal move
                best_logit = float('-inf')

                for move in legal_moves:
                    from_sq = move.from_square
                    to_sq = move.to_square
                    move_idx = from_sq * 64 + to_sq
                    logit = move_logits[move_idx].item()

                    if logit > best_logit:
                        best_logit = logit
                        best_move = move

                if best_move is None:
                    # Fallback to random legal move
                    import random
                    best_move = random.choice(legal_moves)

            # Calculate move quality if Stockfish analysis is enabled and it's the model's turn
            classification = None
            symbol = None
            cp_loss = None

            if stockfish_best and eval_before is not None and not is_stockfish_turn:
                # Analyze model's move
                san = self.board.san(best_move)
                self.board.push(best_move)

                # Get evaluation after model's move
                eval_result = self.stockfish.evaluate_position(self.board, depth=15)
                if eval_result is not None:
                    eval_after = int(eval_result * 100)  # Convert to centipawns

                    # Flip eval if it's black's turn (evals are from white's perspective)
                    if not is_white_turn:
                        eval_before = -eval_before
                        eval_after = -eval_after

                    # Calculate centipawn loss
                    cp_loss = max(0, eval_before - eval_after)
                    self.centipawn_losses.append(cp_loss)

                    # Classify the move
                    classification, symbol = classify_move(cp_loss)

                    # Update game accuracy (only tracks model moves)
                    game_acc = calculate_game_accuracy(self.centipawn_losses)
                    stats.game_accuracy = game_acc
                    stats.estimated_rating = estimate_elo_from_game_accuracy(game_acc)

                    # Update current position eval
                    stats.stockfish_eval = eval_result

                self.move_count += 1
            else:
                # No analysis - just make the move
                san = self.board.san(best_move)
                self.board.push(best_move)
                self.move_count += 1

            # Update display
            board_widget.board_state = self.board.copy()
            stats.move_count = self.move_count
            stats.model_eval = value
            stats.status_text = "Playing..."

            # Add to history with classification
            move_num = (self.move_count + 1) // 2
            history_widget.add_move(move_num, best_move, san, classification, symbol, cp_loss)

            # Wait between moves
            await asyncio.sleep(self.game_speed)

        # Game over
        if self.board.is_game_over():
            result = self.board.result()

            # Final game report
            if self.centipawn_losses and self.opponent_mode == "self-play":
                # Only show accuracy for self-play mode
                game_acc = calculate_game_accuracy(self.centipawn_losses)
                est_rating = estimate_elo_from_game_accuracy(game_acc)
                stats.status_text = f"Game Over: {result} | Accuracy: {game_acc:.1f}% (~{est_rating})"
            elif self.centipawn_losses and self.opponent_mode == "stockfish":
                # Show model's performance against Stockfish
                game_acc = calculate_game_accuracy(self.centipawn_losses)
                est_rating = estimate_elo_from_game_accuracy(game_acc)
                stats.status_text = f"Game Over: {result} | Model accuracy: {game_acc:.1f}% (~{est_rating})"
            else:
                stats.status_text = f"Game Over: {result}"
        else:
            stats.status_text = "Stopped"

        self.playing = False

    def action_toggle_play(self) -> None:
        """Toggle play/pause."""
        if self.playing:
            self.pause_game_clicked()
        else:
            self.start_game_clicked()

    def action_reset(self) -> None:
        """Reset the game."""
        self.reset_game_clicked()

    def action_back(self) -> None:
        """Go back to the main menu."""
        self.playing = False
        if self.stockfish:
            self.stockfish.stop()
        self.app.pop_screen()

    @on(Button.Pressed, "#copy-pgn")
    def copy_pgn_clicked(self) -> None:
        """Copy the current game as PGN to clipboard."""
        try:
            # Create a PGN game from the current board's move stack
            game = chess.pgn.Game()

            # Set headers
            game.headers["Event"] = "Nanochess Interactive Play"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = "?"

            # Set player names based on mode
            if self.opponent_mode == "stockfish":
                game.headers["White"] = "Stockfish"
                game.headers["Black"] = "PRA Model"
            else:
                game.headers["White"] = "PRA Model"
                game.headers["Black"] = "PRA Model"

            # Set result
            if self.board.is_game_over():
                game.headers["Result"] = self.board.result()
            else:
                game.headers["Result"] = "*"

            # Add moves from the board's move stack
            node = game
            temp_board = chess.Board()
            for move in self.board.move_stack:
                node = node.add_variation(move)
                temp_board.push(move)

            # Convert to PGN string
            pgn_string = str(game)

            # Copy to clipboard using subprocess (cross-platform)
            import subprocess
            import sys

            try:
                if sys.platform == "darwin":  # macOS
                    process = subprocess.Popen(
                        ["pbcopy"], stdin=subprocess.PIPE, text=True
                    )
                    process.communicate(pgn_string)
                elif sys.platform == "win32":  # Windows
                    process = subprocess.Popen(
                        ["clip"], stdin=subprocess.PIPE, text=True
                    )
                    process.communicate(pgn_string)
                else:  # Linux/Unix
                    # Try xclip first, then xsel as fallback
                    try:
                        process = subprocess.Popen(
                            ["xclip", "-selection", "clipboard"],
                            stdin=subprocess.PIPE,
                            text=True,
                        )
                        process.communicate(pgn_string)
                    except FileNotFoundError:
                        try:
                            process = subprocess.Popen(
                                ["xsel", "--clipboard", "--input"],
                                stdin=subprocess.PIPE,
                                text=True,
                            )
                            process.communicate(pgn_string)
                        except FileNotFoundError:
                            # Fallback: write to temp file
                            import tempfile
                            fd, path = tempfile.mkstemp(suffix=".pgn", text=True)
                            with os.fdopen(fd, "w") as f:
                                f.write(pgn_string)
                            self.query_one(StatsWidget).status_text = f"PGN saved to {path}"
                            return

                self.query_one(StatsWidget).status_text = "PGN copied to clipboard!"
            except Exception as e:
                # Fallback: write to temp file
                import tempfile
                fd, path = tempfile.mkstemp(suffix=".pgn", text=True)
                with os.fdopen(fd, "w") as f:
                    f.write(pgn_string)
                self.query_one(StatsWidget).status_text = f"PGN saved to {path}"

        except Exception as e:
            self.query_one(StatsWidget).status_text = f"Error copying PGN: {str(e)}"
