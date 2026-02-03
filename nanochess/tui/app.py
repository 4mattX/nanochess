"""Main Textual App class for Nanochess TUI - Dracula theme."""

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from nanochess.tui.screens.main_menu import MainMenuScreen


class NanochessApp(App):
    """The main Nanochess TUI application with Dracula theme."""

    TITLE = "Nanochess"
    SUB_TITLE = "Piece-Routed Attention Training"

    CSS_PATH = Path(__file__).parent / "styles" / "nanochess.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("escape", "back", "Back", show=True),
        Binding("?", "help", "Help", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_command = None

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Use built-in Dracula theme
        self.theme = "dracula"

        self.push_screen(MainMenuScreen())

    def action_back(self) -> None:
        """Go back to the previous screen."""
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def action_help(self) -> None:
        """Show help notification."""
        help_text = """[bold $accent]Nanochess TUI Help[/]

[$accent]Global Keys:[/]
  [$success]q[/]        [$text-muted]- Quit application[/]
  [$success]ESC[/]      [$text-muted]- Go back / Cancel[/]
  [$success]?[/]        [$text-muted]- Show this help[/]

[$accent]Navigation:[/]
  [$success]Tab[/]      [$text-muted]- Next panel/element[/]
  [$success]Shift+Tab[/] [$text-muted]- Previous panel/element[/]
  [$success]arrows[/]   [$text-muted]- Navigate within panels[/]
  [$success]j/k[/]      [$text-muted]- Vim-style up/down[/]
  [$success]Enter[/]    [$text-muted]- Select/Activate[/]
  [$success]Click[/]    [$text-muted]- Select with mouse[/]

[$accent]Training Screen:[/]
  [$success]1/2/3[/]    [$text-muted]- Jump to Dataset/Config/Preview[/]
  [$success]F1[/]       [$text-muted]- Show help[/]
  [$success]F5[/]       [$text-muted]- Cycle presets[/]
  [$success]F10[/]      [$text-muted]- Start training[/]
"""
        self.notify(help_text, title="Help", timeout=15)
