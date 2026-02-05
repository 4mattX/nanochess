"""Main menu screen for Nanochess TUI - Dracula theme."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, Center
from textual.screen import Screen
from textual.widgets import Button, Static, Footer, Label

from nanochess.tui.utils import get_gpu_info
from nanochess.tui.icons import SEPARATOR, GPU, MEMORY, CHIP, CPU, TRAINING, METRICS, PLAY


# Unicode box drawing logo with nerd fonts
def get_logo_simple():
    """Get the simple logo using color palette."""
    return """[bold $text]
    ╭─────────────────────────────────────────────────╮
    │                                                 │
    │           ♟  N A N O C H E S S  ♟               │
    │                                                 │
    │         Piece-Routed Attention Training         │
    │                                                 │
    ╰─────────────────────────────────────────────────╯
[/]
"""

# Big logo for main menu
def get_logo_big():
    """Get the big logo using color palette."""
    return """[bold $text]
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗███████╗███████╗███████╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██║  ██║██╔════╝██╔════╝██╔════╝
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║     ███████║█████╗  ███████╗███████╗
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══╝  ╚════██║╚════██║
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║███████╗███████║███████║
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝

[$primary]Piece-Routed Attention Training[/]
[/]
"""


class MenuButton(Button):
    """A styled menu button with keyboard shortcut indicator."""

    def __init__(
        self,
        label: str,
        shortcut: str,
        id: str | None = None,
        disabled: bool = False,
    ) -> None:
        display_label = f"[$accent bold]{shortcut}[/]  {label}"
        super().__init__(display_label, id=id, disabled=disabled, classes="menu-button")
        self.shortcut = shortcut


class MainMenuScreen(Screen):
    """The main menu screen with Dracula theme styling."""

    BINDINGS = [
        Binding("t", "train", "Train", show=False),
        Binding("e", "evaluate", "Evaluate", show=False),
        Binding("i", "interactive", "Interactive", show=False),
        Binding("q", "quit", "Quit", show=True),
        Binding("up", "focus_previous", "Up", show=False),
        Binding("down", "focus_next", "Down", show=False),
        Binding("k", "focus_previous", "Up", show=False),
        Binding("j", "focus_next", "Down", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Compose the main menu layout."""
        yield Container(
            Vertical(
                Static(get_logo_big(), id="logo"),
                Center(
                    Container(
                        MenuButton(f"{TRAINING}  Train Model", "T", id="btn-train"),
                        MenuButton(f"{METRICS}  Evaluate Model", "E", id="btn-evaluate", disabled=True),
                        MenuButton(f"{PLAY}  Interactive Play", "I", id="btn-interactive"),
                        MenuButton("  Quit", "Q", id="btn-quit"),
                        Static("[$primary]v0.1.0[/]", id="version-info"),
                        id="menu-box",
                    ),
                ),
                Center(
                    Static(
                        "[$text-muted]Use [$accent]arrows[/] or [$accent]j/k[/] to navigate, [$accent]Enter[/] to select[/]",
                        id="help-text",
                    ),
                ),
                id="logo-container",
            ),
            id="main-menu-container",
        )
        yield Static(self._get_system_info(), classes="status-bar")
        yield Footer()

    def _get_system_info(self) -> str:
        """Get system information for the status bar with nerd font icons."""
        try:
            gpu = get_gpu_info()
            if gpu.device_type == "cuda":
                return (
                    f"[$text-muted]{GPU}[/] [$success]{gpu.device_name}[/] "
                    f"[$text-muted]{SEPARATOR}[/] [$text-muted]{MEMORY}[/] [$success]{gpu.vram_total_gb:.0f}GB[/] "
                    f"[$text-muted]{SEPARATOR}[/] [$text-muted]CUDA:[/] [$primary]{gpu.cuda_version}[/] "
                    f"[$text-muted]{SEPARATOR}[/] [$text-muted]PyTorch:[/] [$primary]{gpu.pytorch_version}[/]"
                )
            elif gpu.device_type == "mps":
                return (
                    f"[$text-muted]{CHIP}[/] [$success]Apple Silicon[/] "
                    f"[$text-muted]{SEPARATOR}[/] [$text-muted]PyTorch:[/] [$primary]{gpu.pytorch_version}[/]"
                )
            else:
                return (
                    f"[$text-muted]{CPU}[/] [$text]CPU[/] "
                    f"[$text-muted]{SEPARATOR}[/] [$text-muted]PyTorch:[/] [$primary]{gpu.pytorch_version}[/]"
                )
        except Exception as e:
            return "[$warning]System info unavailable[/]"

    def on_mount(self) -> None:
        """Focus the first button on mount."""
        try:
            self.query_one("#btn-train", Button).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-train":
            self.action_train()
        elif button_id == "btn-evaluate":
            self.action_evaluate()
        elif button_id == "btn-interactive":
            self.action_interactive()
        elif button_id == "btn-quit":
            self.action_quit()

    def action_focus_next(self) -> None:
        """Focus the next focusable widget."""
        self.screen.focus_next()

    def action_focus_previous(self) -> None:
        """Focus the previous focusable widget."""
        self.screen.focus_previous()

    def action_train(self) -> None:
        """Switch to the training screen."""
        from nanochess.tui.screens.training import TrainingScreen
        self.app.push_screen(TrainingScreen())

    def action_evaluate(self) -> None:
        """Switch to the evaluation screen (not implemented yet)."""
        self.app.notify(
            "[$warning]Evaluation screen coming soon![/]",
            title="Not Implemented",
            severity="warning",
        )

    def action_interactive(self) -> None:
        """Switch to the interactive play screen."""
        from nanochess.tui.screens.interactive import InteractiveScreen
        self.app.push_screen(InteractiveScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
