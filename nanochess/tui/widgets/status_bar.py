"""Status bar widget for displaying GPU and system info - Dracula theme."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.widget import Widget

from nanochess.tui.utils import get_gpu_info, get_dataset_info, format_number
from nanochess.tui.icons import SEPARATOR, GPU, MEMORY, DATABASE, CPU, CHIP


class StatusBar(Widget):
    """Status bar showing GPU, memory, and dataset status."""


    def compose(self) -> ComposeResult:
        """Compose the status bar."""
        yield Horizontal(
            Static(self._get_status_text(), id="status-text"),
        )

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.refresh_status()

    def refresh_status(self) -> None:
        """Refresh the status bar information."""
        try:
            status_widget = self.query_one("#status-text", Static)
            status_widget.update(self._get_status_text())
        except Exception:
            pass

    def _get_status_text(self) -> str:
        """Build the status text with Dracula theme colors and nerd font icons."""
        parts = []

        # GPU info with icon
        try:
            gpu = get_gpu_info()
            if gpu.device_type == "cuda":
                parts.append(f"[$text-muted]{GPU}[/] [$success]{gpu.device_name}[/]")
                parts.append(f"[$text-muted]{MEMORY}[/] [$accent]{gpu.vram_used_gb:.1f}[/][$text-muted]/[/][$accent]{gpu.vram_total_gb:.0f}GB[/]")
            elif gpu.device_type == "mps":
                parts.append(f"[$text-muted]{CHIP}[/] [$success]Apple Silicon[/]")
            else:
                parts.append(f"[$text-muted]{CPU}[/] [$text]CPU[/]")
        except Exception as e:
            parts.append(f"[$text-muted]{GPU}[/] [$error]Error[/]")

        # Dataset info with icon
        try:
            data = get_dataset_info()
            if data.exists:
                total = data.num_train_positions + data.num_val_positions
                parts.append(f"[$text-muted]{DATABASE}[/] [$success]{format_number(total)}[/] [$text-muted]positions[/]")
            else:
                parts.append(f"[$text-muted]{DATABASE}[/] [$warning]Not found[/]")
        except Exception:
            parts.append(f"[$text-muted]{DATABASE}[/] [$error]Error[/]")

        return f" [$text-muted]{SEPARATOR}[/] ".join(parts)
