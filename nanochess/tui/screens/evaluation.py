"""Evaluation screen for Nanochess TUI - Custom Blue theme."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Center
from textual.screen import Screen
from textual.widgets import Static, Footer



class EvaluationScreen(Screen):
    """The evaluation screen (placeholder)."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
    ]


    def compose(self) -> ComposeResult:
        """Compose the evaluation screen."""
        yield Container(
            Vertical(
                Static("[$accent]   [/]", id="placeholder-icon"),
                Static("[$accent]Model Evaluation[/]", id="placeholder-title"),
                Static("[$primary]This feature is coming soon[/]", id="placeholder-text"),
                Static("[$primary]Press ESC to go back[/]", id="placeholder-hint"),
                id="placeholder-box",
            ),
            id="placeholder-container",
        )
        yield Footer()

    def action_back(self) -> None:
        """Go back to the main menu."""
        self.app.pop_screen()
